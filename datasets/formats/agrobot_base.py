# -*- coding: utf-8 -*-
import math
import yaml
import csv
from pathlib import Path
from collections import OrderedDict
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import bz2
import pickle

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from pycocotools import mask
from pycocotools.coco import COCO
from contextlib import redirect_stdout
import io

class SequenceDataset(Dataset):

    def __init__(self,
                dataset_file,
                subset,
                class_labels,
                depth_rel_path,
                odometry_rel_file_path,
                frame_window_size,
                mask_robot_path=None,
                preds_rel_path=None,
                max_depth=-1
                ):

      self.dataset_name = Path(dataset_file).stem
      self._root_dir = Path(dataset_file).parent.parent / self.dataset_name


      self.coco_mask = mask
      self.class_labels = class_labels

      self.subset = subset

      assert self.subset == 'train' or self.subset == 'val'

      self.max_depth = max_depth
      
      # Directories and root addresses
      self.annotation_files_dir = self._root_dir / (self.dataset_name + '.json')
      #  Image Sets lists (train, valid, eval)
      self.dataset_config_list_dir = self._root_dir / (self.dataset_name + '.yaml')
      with open(self.dataset_config_list_dir) as fp:
          self.dataset_config = yaml.load(fp, Loader=yaml.FullLoader)

      self.image_sets = self.dataset_config["image_sets"]
      self.stds = self.dataset_config["img_mu"]
      self.means = self.dataset_config["img_std"]

      # initialize COCO api for instance annotations
      f = io.StringIO()
      with redirect_stdout(f):
        self.coco = COCO(self.annotation_files_dir)

      # getting the IDs of images inside the directory
      self.ids = self.coco.getImgIds()
      # Get Categories of loaded Dataset

      self.id_to_class_label = OrderedDict()
      self.cat_ids = set()
      # Get cat ids from dataset with names or supercategories sepecified in te config file
      for id, c in self.coco.cats.items():
        if c['supercategory'] in self.class_labels:
          self.id_to_class_label[id] = self.class_labels.index(c['supercategory'])
          self.cat_ids.add(id)
        elif c['name'] in self.class_labels:
          self.id_to_class_label[id] = self.class_labels.index(c['name'])
          self.cat_ids.add(id)

      # display COCO categories and supercategories
      self.classes = self.coco.loadCats(self.cat_ids)
      self.coco_labels = [cat['name'] for cat in self.classes]

      # For nerf we have a single set for validation, so we can use both val and eval to test
      # self.img_set_ids = self.image_sets['valid'] + self.image_sets['eval']
      self.img_set_ids = self.image_sets['eval']

      def get_img_path_to_ids(ids, remove_edge_frames=False):
        img_path_to_ids = {}
        for md in self.coco.loadImgs(ids):
          im_path = self._root_dir / self.dataset_rel_path(md['path'])
          if remove_edge_frames:
            img_seq_paths = [p for p in sorted(im_path.parent.iterdir()) if p.suffix == im_path.suffix]
            im_seq_pos = img_seq_paths.index(im_path)
            # remove frames too close to the beginning and end of the sequence
            if im_seq_pos < frame_window_size + 1  or len(img_seq_paths) - im_seq_pos < frame_window_size + 1:
              continue   

          img_path_to_ids[im_path] = md['id']

        return img_path_to_ids
      
      self.img_path_to_ids = get_img_path_to_ids(self.img_set_ids, remove_edge_frames=True)
      # remove frames too close to the sequence beginning and end
      self.img_set_ids = [id for id in self.img_path_to_ids.values()]

      self.img_path_to_ids_train = get_img_path_to_ids(self.image_sets['train'])

      self.depth_rel_path = depth_rel_path
      self.preds_rel_path = preds_rel_path

      win_bound = frame_window_size if frame_window_size % 2 == 0 else frame_window_size - 1
    
      # odd frames full as training frames
      self.train_frames_idxs = list(  range(-win_bound-1, win_bound+2,   2))
      # Odd frames as validation and pose-opt only frames
      self.val_frames_idxs = list(range(-win_bound,   win_bound+1, 2))
      self.semantic_val_frame_idx = self.val_frames_idxs.index(0)

      self.odom_file_path = odometry_rel_file_path

      if isinstance(mask_robot_path, (str, Path)):
        self.robot_mask_path = str(mask_robot_path)

      # transformations for tensors
      self.tensorize_img = self.tensorize_depth = transforms.ToTensor()

    ##
    ## get item
    ##
    ######################################################################################
    def __getitem__(self, index):

      
      if self.subset == 'train':
        self.used_frames = self.train_frames_idxs
      elif self.subset in ('val','valid'):
        self.used_frames = self.val_frames_idxs


      imgList, semList, semPredList, semConfList, instList, instPredList, instConfList,\
      depthList, odomList, odomTs, camParams, fileNameList = self.getReprojSequence(index)
      
      data = [{} for _ in range(len(imgList))]

      for i in range(len(data)): data[i]['intrinsics'] = camParams['intrinsics']
      for i in range(len(data)): data[i]['extrinsics'] = camParams['extrinsics']

      # pass odometry to the next frame
      for i, o in enumerate(odomList): data[i]['odom'] = o
      for i, o_ts in enumerate(odomTs): data[i]['odom_ts'] = o_ts

      if hasattr(self, 'robot_mask_path'):
        robot_mask = self.getRobotMask(index)
        robotMaskList = [robot_mask for _ in range(len(imgList))]
        robot_mask_tensors = self.prepareTensors(robotMaskList, 'mask')
        for i, mask in enumerate(robot_mask_tensors): data[i]["robot_mask"] = mask



      # Tensorize image/imgae-list and its mask/mask-list
      imgs_tensors = self.prepareTensors(imgList, 'rgb')
      for i, im in enumerate(imgs_tensors): data[i]["rgb"] = im

      depth_tensors = self.prepareTensors(depthList, 'depth')
      for i, d in enumerate(depth_tensors): data[i]["depth"] = d 

      if len(semList) == 0 or len(instList) == 0:
        raise ValueError('No labels nor predictions were loaded')
      
      # prepare labels and flags for RNN loss computation
      sem_tensors = self.prepareTensors(semList, 'label')
      for i, m in enumerate(sem_tensors): 
        data[i]["semantics"] = m
      
      sem_pred_tensors = self.prepareTensors(semPredList, 'label')
      for i, m in enumerate(sem_pred_tensors): 
        data[i]["semantics_pred"] = m
      
      sem_conf_tensors = self.prepareTensors(semConfList, 'rgb')
      for i, m in enumerate(sem_conf_tensors): 
        data[i]["sem_conf"] = m
      
      inst_tensors = self.prepareTensors(instList, 'label')
      for i, m in enumerate(inst_tensors): 
        data[i]["imap"] = m

      inst_pred_tensors = self.prepareTensors(instPredList, 'label')
      for i, m in enumerate(inst_pred_tensors): 
        data[i]["imap_pred"] = m
      
      inst_conf_tensors = self.prepareTensors(instConfList, 'rgb')
      for i, m in enumerate(inst_conf_tensors): 
        data[i]["inst_conf"] = m

      for i, file_name in enumerate(fileNameList): data[i]["file_names"] = file_name

      return data


    ##
    ## Dataloader utils
    ##
    ######################################################################################

    def prepareTensors(self, imgList, imgType):
      result = []
      img = None
      reveseType = False
      if not type(imgList) == list:
        imgList = [imgList]
        reveseType = True

      for i in imgList:
        # tensorize image i
        if isinstance(i, torch.Tensor):
          img = i
        elif i is None:
          img = torch.Tensor([])
        elif imgType == 'rgb':
          img = i.copy()
          img = self.tensorize_img(img.copy()).float()

        elif imgType == 'label':
          img = i.copy()
          img = torch.from_numpy(np.array(img)).long()

        elif imgType == 'depth' or imgType == 'mask':

          # Resize the img and it label
          img = i.copy()
          img = self.tensorize_depth(img)

        # append image i to list of images !
        if not reveseType:
          result.append(img)
        else:
          result = img

      return result

    def getRobotMask(self, index):
      if not hasattr(self, 'robot_mask_path'):
        raise ValueError('Robot mask was not enabled in the dataloader config file. Add sequencing:robot_mask:enable:True to you .yaml config file')
      img_path = self.getImgPathFromIdx(index)
      return Image.open(img_path.parent.parent / self.robot_mask_path).convert('L')

    def getReprojSequence(self, index):
      return self.getNRealImgLabelPairs(index, return_reproj_data=True)

    def getImageMetadataFromIdx(self,index):
      return self.coco.loadImgs(self.getTragetID(index))[0]

    def getDepthFromIdx(self,index):
      md = self.getImageMetadataFromIdx(index)
      ds_img_path = self.dataset_rel_path(md['path'])
      img_path = self._root_dir / ds_img_path
      return Image.open(img_path.parent / self.depth_rel_path / img_path.name)

    def csv_odom_to_transforms(self, path):
      odom_tfs = {}
      with open(path, mode='r') as f:
        reader = csv.reader(f)
        # get header and change timestamp label name
        header = next(reader)
        header[0] = 'ts'
        # Convert string odometry to numpy transfor matrices
        for row in reader:
          odom = {l: row[i] for i, l in enumerate(header)}
          # Translarion and rotation quaternion as numpy arrays 
          trans = np.array([float(odom[l]) for l in ['tx', 'ty', 'tz']])
          rot = Rotation.from_quat([float(odom[l]) for l in ['qx', 'qy', 'qz', 'qw']]).as_matrix()
          # Build numpy transform matrix
          odom_tf = torch.eye(4)
          odom_tf[0:3, 3] = torch.from_numpy(trans)
          odom_tf[0:3, 0:3] = torch.from_numpy(rot)
          # Add transform to timestamp indexed dictionary
          odom_tfs[odom['ts']] = odom_tf
      
      return odom_tfs

    def getNRealImgLabelPairs(self, index, return_reproj_data=False):
      img_id = self.getTragetID(index)
      # get sorted image paths of all images in the dataset sequence
      img_metadata = self.coco.loadImgs(img_id)[0]
      im_ds_path = self.dataset_rel_path(img_metadata['path'])
      img_path = str(self._root_dir / im_ds_path)

      return self.getNRealImgLabelPairsFromImgPath(img_path, return_reproj_data) 

    def getNRealImgLabelPairsFromImgPath(self, img_path, return_reproj_data=False):
      img_path = Path(img_path)
      img_parent_path = Path(img_path).parent
      img_seq_paths = [p for p in sorted(img_parent_path.iterdir()) if p.suffix == img_path.suffix]
      # index of the central image of the sequence to extract
      seq_idx = img_seq_paths.index(img_path)

      if return_reproj_data:
        odom_path = img_parent_path / self.odom_file_path
        if odom_path.suffix == '.csv':
          # read odometry as a disctionary indexed by [us] timestamp as a string
          odom_from_ts = self.csv_odom_to_transforms(str(img_parent_path / self.odom_file_path))
        elif odom_path.suffix == '.npz':
          metashape_odom = np.load(odom_path)
          tfs = metashape_odom['arr_0']
          tfs[..., :3, 3] *= 0.03
          odom_from_ts = {ts:torch.from_numpy(tf).type(torch.float) for ts,tf in zip(metashape_odom['arr_1'], tfs)} 

        else:
          raise NotImplementedError(f'Unsupported filetype {odom_path}. ',
                                    'Supported types: [csv, npz]')
        
      # Extract images for the sequence relative to the central one,
      # considering the direction of travel
      # The last frame is always idx=0 (aka. central frame)
      imgList = []
      semList = []
      instList = []
      semPredList = []
      semConfList = []
      instPredList = []
      instConfList = []
      depthList = []
      odomList = []
      odomTs = []
      fileNameList = []

      frame_deltas = reversed(sorted(self.used_frames))
      
      img_idxs = [min(len(img_seq_paths)-1,max(0,int(seq_idx - d))) for d in frame_deltas]
      img_paths = [img_seq_paths[idx] for idx in img_idxs]

      if not isinstance(self, InferenceDataset):
        # Remove train frames used to train CNN 
        img_paths = [p for p in img_paths if p not in self.img_path_to_ids_train]

        # Remove unexpected val frames from train sequence
        if self.subset == 'train':
          img_paths = [p for p in img_paths if p not in self.img_path_to_ids]

      if return_reproj_data:
        center_robot_odom = odom_from_ts[str(img_path.name).split('.')[0]]
        # load camera intrinsics and extrinsincs from yaml file
        with open(str(img_parent_path / 'params.yaml'), mode='r') as yml:
          camParams = yaml.load(yml, Loader=yaml.FullLoader)
          camParams = {k:torch.Tensor(v) for k,v in camParams.items()}
        ext, ext_i = (camParams['extrinsics'], camParams['extrinsics'].inverse())

      for i,path in enumerate(img_paths):
        img, sem_pred, inst_pred, sem_conf, inst_conf, file_name = self.getImgPredPairFromPath(path)

        if path == img_path:
          # load label only for the center frame
          _, sem_label, inst_label, _= self.getImgLabelPairFromPath(path)
        else:
          _, sem_label, inst_label, _= self.getImgAndEmptyLabelFromPath(path)

        if return_reproj_data or self.max_depth > 0:
          depth_img = Image.open(path.parent / self.depth_rel_path / path.name)

        # filter masks farther than the specified depth
        if self.max_depth > 0:
          # if not np.all(np.asarray(inst_label) == -1):
          #   inst_label = self.filterMasksWithDepth(inst_label, depth_img)
          #   sem_label = torch.from_numpy(np.array(sem_label))
          #   sem_label[inst_label == 0] = 0
          inst_pred_buffer = inst_pred
          inst_pred = self.filterMasksWithDepth(inst_pred, depth_img)
          flipped_inst_mask = torch.logical_xor(torch.from_numpy(inst_pred_buffer), inst_pred)
          inst_conf[flipped_inst_mask] = 1
          
          sem_pred = torch.from_numpy(np.array(sem_pred))
          sem_pred[inst_pred == 0] = 0
          sem_conf[flipped_inst_mask] = 1


        imgList.append(img)
        semList.append(sem_label)
        semPredList.append(sem_pred)
        semConfList.append(sem_conf)
        instList.append(inst_label)
        instPredList.append(inst_pred)
        instConfList.append(inst_conf)
        fileNameList.append(file_name)

        if return_reproj_data:
          # get depth
          depthList.append(depth_img)
          
          #Transform poses around center sem validation frame
          robot_odom = odom_from_ts[str(path.name).split('.')[0]]
          robot_odom_centered = robot_odom.inverse() @ center_robot_odom
          frame_odom =  ext_i @ robot_odom_centered @ ext
          odomTs.append(str(path.name).split('.')[0])
          odomList.append(frame_odom)
      
      if return_reproj_data:
        return imgList, semList, semPredList, semConfList, instList, instPredList, instConfList, depthList, odomList, odomTs, camParams, fileNameList

      return imgList, semList, semPredList, semConfList, instList, instPredList, instConfList, fileNameList

    def getTragetID(self, index):
      img_set_ids = self.img_set_ids[index]
      img_list_idx = next((index for (index, d) in enumerate(self.coco.dataset["images"]) if d["id"] == img_set_ids), None)
      image_id = self.coco.dataset["images"][img_list_idx]["id"]

      return image_id

    def getImgLabelPairFromPath(self, path):
      # Retrieve annotated image if in the dataset
      if path in self.img_path_to_ids.keys():
        return self.getImgLabelPairFromId(self.img_path_to_ids[path])
      
      return self.getImgAndEmptyLabelFromPath(path)

    def getImgAndEmptyLabelFromPath(self, path):
      # If image is not annotated, return it and an empty mask
      img = Image.open(path).convert('RGB')
      sem_label = Image.fromarray(np.ones(img.size[::-1]) *  -1)
      inst_label = Image.fromarray(np.ones(img.size[::-1]) *  -1)
      return img, sem_label, inst_label, Path(path).name

    def getImgLabelPairFromIdx(self, index):
      img_id = self.getTragetID(index)
      return self.getImgLabelPairFromId(img_id)

    def getImgPredPairFromPath(self, path):
      #  Load BGR image and convert it to RGB
      img_path = self._root_dir / path
      img = Image.open(self._root_dir / path).convert('RGB')
      if 'unet' in self.preds_rel_path:
        semantics, imap, sem_conf, inst_conf = self.getUnetPreds(img_path)
      elif 'maskrcnn' in self.preds_rel_path:
        semantics, imap, sem_conf, inst_conf = self.getMaskrcnnPreds(img_path)
      elif 'deeplab' in self.preds_rel_path:
        semantics, imap, sem_conf, inst_conf = self.getDeeplabPreds(img_path)
      elif 'mask2former' in self.preds_rel_path:
        semantics, imap, sem_conf, inst_conf = self.getMask2FormerPreds(img_path)
      else:
        raise NotImplementedError(f'Load predictions for path name {self.preds_rel_path} not implemented')

      return img, semantics, imap, sem_conf, inst_conf, Path(path).name

    def filterMasksWithDepth(self, mask, depth_img):
      if not isinstance(depth_img,torch.Tensor):
        depth_img = self.prepareTensors(depth_img, 'depth') * 0.001 # depth in [m]
      if not isinstance(mask,torch.Tensor):
        mask = self.prepareTensors(mask, 'label')
      
      
      if depth_img.shape != mask.shape:
        depth_img = torch.nn.functional.interpolate(depth_img[None], mask.shape, mode='bilinear')[0]

      # Get pixels with valid depth below the specified treshold
      valid_ids = mask[torch.logical_and(depth_img[0] <= self.max_depth, depth_img[0]>0)]
      # Count ID and valid ID pixels
      id_counts = torch.bincount(mask.view(-1))
      id_counts_valid = torch.bincount(valid_ids.view(-1))
      id_counts_valid_padded = torch.cat((id_counts_valid, torch.zeros(id_counts.shape[0] - id_counts_valid.shape[0])))
      # Only keep masks with 80% of their pixels fulfilling the condition
      valid_masks = id_counts_valid_padded / id_counts > 0.5

      return torch.where(valid_masks[mask], mask, torch.zeros_like(mask))
    
    def getUnetPreds(self, path):
      with bz2.open(path.parent / self.preds_rel_path / f'{path.stem}.pkl.bz2', "rb") as f:
        # Decompress data from file
        preds = pickle.load(f)
      semantics = Image.fromarray(preds['sem_seg']['preds'].astype(np.int32))
      imap = Image.fromarray(preds['instances']['imap'].astype(np.int32))
      sem_conf = Image.fromarray(preds['sem_seg']['confidence'].squeeze().cpu().numpy())
      inst_conf = sem_conf
      return semantics, imap, sem_conf, inst_conf
    
    def getMaskrcnnPreds(self, path):
      with open(path.parent / self.preds_rel_path / f'{path.stem}.pkl', "rb") as f:
        # Decompress data from file
        preds = pickle.load(f)
      imap = torch.zeros_like(preds['masks'])
      imap[preds['masks'] > 0.5] = 1
      imap = imap.squeeze()
      imap = ((imap.sum(0) > 0) + imap.argmax(0)).numpy().astype(np.int32)
      semantics = (imap > 0).astype(np.int32)
      confidence = preds['masks'].squeeze().max(dim=0).values.numpy()
      confidence[confidence == 0.0] = 0.9
      sem_conf = confidence
      inst_conf =confidence
      return semantics, imap, sem_conf, inst_conf
    
    def getDeeplabPreds(self, path):
      with open(path.parent / self.preds_rel_path / f'{path.stem}.pkl', "rb") as f:
        # Decompress data from file
        preds = pickle.load(f)
      imap = preds['panoptic'][0,1].cpu().numpy()
      semantics = preds['panoptic'][0,0].cpu().numpy()
      confidence = np.ones_like(imap)
      sem_conf = confidence
      inst_conf = confidence
      return semantics, imap, sem_conf, inst_conf
    
    def getMask2FormerPreds(self, path):
      with open(path.parent / self.preds_rel_path / f'{path.stem}.pkl', "rb") as f:
        # Decompress data from file
        preds = pickle.load(f)
      imap = preds[1].cpu().numpy()
      semantics = preds[0].cpu().numpy()
      confidence = preds[2]
      confidence[imap==0] = -confidence[imap==0]
      confidence = torch.sigmoid(confidence).cpu().numpy()
      sem_conf = confidence
      inst_conf = confidence
      return semantics, imap, sem_conf, inst_conf


    def getImgLabelPairFromId(self, img_id):
      # Get meta data of called image
      img_metadata = self.coco.loadImgs(img_id)[0]
      im_ds_path = self.dataset_rel_path(img_metadata['path'])
      #  Load BGR image and convert it to RGB
      img = Image.open(self._root_dir / im_ds_path).convert('RGB')
      # Creat the mask with loaded annotations with same size as RGB image
      sem_label = Image.fromarray(self.generateMask(img_metadata))
      inst_label = Image.fromarray(self.generateInstanceMasks(img_metadata))
      return img, sem_label, inst_label, Path(im_ds_path).name

    def generateMask(self, img_metadata):

      anns_ids = self.coco.getAnnIds(imgIds=img_metadata['id'], catIds=self.cat_ids, iscrowd=None)
      anns = self.coco.loadAnns(anns_ids)

      mask = np.zeros((img_metadata['height'], img_metadata['width'])).astype(np.uint32)
      for ann in anns:
        if not ann['segmentation']:
          continue
        ann_mask = self.coco.annToMask(ann)
        mask *= not(ann_mask).all()
        mask += ann_mask * self.id_to_class_label[ann["category_id"]]
        mask = np.clip(mask, 0, max(self.id_to_class_label.values()))
      return mask.astype(np.int32)
    
    def generateInstanceMasks(self, img_metadata):
      cat_ids = self.coco.getCatIds()
      anns_ids = self.coco.getAnnIds(imgIds=img_metadata['id'], catIds=self.cat_ids, iscrowd=None)
      anns = self.coco.loadAnns(anns_ids)
      mask = np.zeros((img_metadata['height'], img_metadata['width']), dtype=np.int32)
      for i in range( len(anns) ):
        # assing instnace id+1 so background is id=0
        mask[self.coco.annToMask( anns[i] ) != 0] = i+1
      return mask

    def dataset_rel_path(self, path=''):
      path_parts = Path(path).parts
      if len(path_parts) < 4:
        raise ValueError('Invalid dataset path, it only has 2 or less subpaths')
      return str(Path(*path_parts[3:]))

    def __len__(self):
      return len(self.img_set_ids)


class InferenceDataset(SequenceDataset):

    def __init__(self,
                dataset_file,
                subset,
                class_labels,
                depth_rel_path,
                odometry_rel_file_path,
                frame_window_size,
                mask_robot_path=None,
                preds_rel_path=None,
                max_depth=-1,
                num_rm_frames=10,
                ):

      self.dataset_name = Path(dataset_file).stem
      self._root_dir = Path(dataset_file).parent.parent / self.dataset_name


      self.coco_mask = mask
      self.class_labels = class_labels

      self.subset = subset

      assert self.subset == 'train' or self.subset == 'val'

      self.max_depth = max_depth
      
      # Directories and root addresses
      self.annotation_files_dir = self._root_dir / (self.dataset_name + '.json')
      #  Image Sets lists (train, valid, eval)
      self.dataset_config_list_dir = self._root_dir / (self.dataset_name + '.yaml')
      with open(self.dataset_config_list_dir) as fp:
          self.dataset_config = yaml.load(fp, Loader=yaml.FullLoader)

      self.image_sets = self.dataset_config["image_sets"]
      self.stds = self.dataset_config["img_mu"]
      self.means = self.dataset_config["img_std"]

      # For inference in thie whole dataset we want to run on all images of all sets 
      self.img_set_ids = self.image_sets['train'] + self.image_sets['valid'] + self.image_sets['eval']
      
      # Load dataset file to get all folders where there are images
      f = io.StringIO()
      with redirect_stdout(f):
        self.coco = COCO(self.annotation_files_dir)

      metadata = self.coco.loadImgs(self.img_set_ids)

      seq_rel_paths = set([Path(self.dataset_rel_path(m['path'])).parent for m in metadata])
      
      # Specific for Esra's experiments on ALL images in the dataset, including sequences without ANY labels
      seq_rel_paths.add(Path('20200924/row1'))
      
      self.seq_paths = sorted([self._root_dir / p for p in seq_rel_paths])
      img_extension =  Path(metadata[0]['path']).suffix



      self.img_paths = ([sorted(list(seq_path.glob(f'./*{img_extension}'))) for seq_path in self.seq_paths])
      # clip sequencees to have the same length
      self.seq_length = min([len(l) for l in self.img_paths])
      self.img_paths = [l[:self.seq_length] for l in self.img_paths]      

      # Get Categories of loaded Dataset
      self.id_to_class_label = OrderedDict()
      self.cat_ids = set()
      # Get cat ids from dataset with names or supercategories sepecified in te config file
      for id, c in self.coco.cats.items():
        if c['supercategory'] in self.class_labels:
          self.id_to_class_label[id] = self.class_labels.index(c['supercategory'])
          self.cat_ids.add(id)
        elif c['name'] in self.class_labels:
          self.id_to_class_label[id] = self.class_labels.index(c['name'])
          self.cat_ids.add(id)

      # display COCO categories and supercategories
      self.classes = self.coco.loadCats(self.cat_ids)
      self.coco_labels = [cat['name'] for cat in self.classes]

      # For nerf we have a single set for validation, so we can use both val and eval to test

      self.depth_rel_path = depth_rel_path
      self.preds_rel_path = preds_rel_path

      self.num_rm_frames = num_rm_frames
      self.win_bound = frame_window_size if frame_window_size % 2 == 0 else frame_window_size - 1
      self.win_len = self.win_bound * 2 + 3 - (self.num_rm_frames * 2)

      # odd frames full as training frames
      self.train_frames_idxs = list(  range(-self.win_bound-1, self.win_bound+2,   2))
      # Odd frames as validation and pose-opt only frames
      self.val_frames_idxs = list(range(-self.win_bound-1 + num_rm_frames,   self.win_bound+2 - num_rm_frames))

      self.semantic_val_frame_idx = self.val_frames_idxs.index(0)

      self.odom_file_path = odometry_rel_file_path

      if isinstance(mask_robot_path, (str, Path)):
        self.robot_mask_path = str(mask_robot_path)

      # transformations for tensors
      self.tensorize_img = self.tensorize_depth = transforms.ToTensor()
    
    def filename_to_inference_seq_idx(self, idx):
      win_per_seq = math.ceil((self.seq_length - (self.num_rm_frames * 2)) / self.win_len)
      seq_idx = idx // win_per_seq
      img_idx = (self.win_bound + 2 + (idx * self.win_len)) % self.seq_length

      file_idxs = set([min(max(0, img_idx + shift), len(self.img_paths[seq_idx]) - 1)  for shift in range(-self.win_len//2 + 1, (self.win_len//2) + 1)])
      seq_filenames = [self.img_paths[seq_idx][fidx]  for fidx in file_idxs]

      return seq_idx, img_idx, seq_filenames

    def idx_to_inference_filename(self, idx):
      seq_idx, img_idx, _ = self.filename_to_inference_seq_idx(idx)
      return self.img_paths[seq_idx][img_idx]

    def getNRealImgLabelPairs(self, index, return_reproj_data=False):
      img_path = self.idx_to_inference_filename(index)

      return self.getNRealImgLabelPairsFromImgPath(img_path, return_reproj_data)
  
    def getImgLabelPairFromPath(self, path):      
      return self.getImgAndEmptyLabelFromPath(path)


class BUP20SequenceDataset(SequenceDataset):
  def __init__(self, dataset_file, subset='train',
               seq_num_frames=40,
               odom_src='odom',
               preds_rel_path=None,
               max_depth=-1,
               class_labels=['bg', 'pepper'],
               ):

    if odom_src == 'rgbd':
      odometry_rel_file_path = 'rgbd_odom.csv'
    elif odom_src == 'odom':
      odometry_rel_file_path = 'odometry.csv'
    elif odom_src == 'metashape':
      odometry_rel_file_path = 'metashape_cameras.npz'
    else:
      raise ValueError('BUP20 sequence dataset only supports [odom, rgbd, metashape] as poses source, ',
                       f'but {odom_src} was given')


    return super().__init__(dataset_file=dataset_file,
                            subset=subset,
                            class_labels=class_labels,
                            depth_rel_path='depth',
                            odometry_rel_file_path=odometry_rel_file_path,
                            frame_window_size=seq_num_frames,
                            preds_rel_path=preds_rel_path,
                            max_depth=max_depth,
                            )

class BUP20InferenceDataset(InferenceDataset):
  def __init__(self, dataset_file, subset='train',
               seq_num_frames=40, num_rm_frames=10,
               odom_src='odom',
               preds_rel_path=None,
               max_depth=-1,
               class_labels=['bg', 'pepper'],
               ):

    if odom_src == 'rgbd':
      odometry_rel_file_path = 'rgbd_odom.csv'
    elif odom_src == 'odom':
      odometry_rel_file_path = 'odometry.csv'
    elif odom_src == 'metashape':
      odometry_rel_file_path = 'metashape_cameras.npz'
    else:
      raise ValueError('BUP20 sequence dataset only supports [odom, rgbd, metashape] as poses source, ',
                       f'but {odom_src} was given')


    return super().__init__(dataset_file=dataset_file,
                            subset=subset,
                            class_labels=class_labels,
                            depth_rel_path='depth',
                            odometry_rel_file_path=odometry_rel_file_path,
                            frame_window_size=seq_num_frames,
                            preds_rel_path=preds_rel_path,
                            max_depth=max_depth,
                            num_rm_frames=num_rm_frames,
                            )

class SB20SequenceDataset(SequenceDataset):
  def __init__(self, dataset_file, subset='train', seq_num_frames=40, odom_src='odom', preds_rel_path=None):

    if odom_src == 'rgbd':
      odometry_rel_file_path = '../rgbd_odom.csv'
    elif odom_src == 'odom':
      odometry_rel_file_path = '../poseDict.csv'
    elif odom_src == 'metashape':
      odometry_rel_file_path = '../metashape.npz'
    else:
      raise ValueError('SB20 sequence dataset only supports [odom, rgbd, metashape] as poses source, ',
                       f'but {odom_src} was given')


    return super().__init__(dataset_file=dataset_file,
                            subset=subset,
                            class_labels=['bg', 'crop', 'weed'],
                            depth_rel_path='../depth',
                            odometry_rel_file_path=odometry_rel_file_path,
                            frame_window_size=seq_num_frames,
                            preds_rel_path=f'../{preds_rel_path}',
                            )