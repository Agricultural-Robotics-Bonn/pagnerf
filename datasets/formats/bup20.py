# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from copy import deepcopy
import cv2
from tqdm import tqdm
import numpy as np
import torch
from torch.multiprocessing import Pool
from wisp.core import Rays
from kaolin.render.camera import Camera
from wisp.ops.raygen import generate_pinhole_rays, generate_centered_pixel_coords
from wisp.ops.image import resize_mip
from pathlib import Path
import yaml
import json
from .format_base import DatasetFormatBase
import pickle

from datasets.utils import get_scale_from_ply_mesh, transform_cv_to_gl_poses

from .categories import stuff_class_names, things_class_names,  AD20K_to_replica_class_ids

from .agrobot_base import BUP20SequenceDataset, BUP20InferenceDataset

""" A module for loading Replica dataset files generated with habitat-sim."""

# Default image modes map between Replica folders and wisp
default_modes_map = {'rgb':'imgs',
                     'depth':'depths',
                     'semantic':'semantic',
                     'imap':'instance',
                     'instance_preds':'instance_preds',
                     'semantic_preds':'semantic_preds',
                     }

default_modes = list(default_modes_map.values())

default_class_labels = ['bg', 'pepper']

class BUP20(DatasetFormatBase):

    @staticmethod
    def get_semantic_info(root, class_labels=default_class_labels):

        if not class_labels:
            class_labels = default_class_labels
        # with open(Path(root).expanduser() / 'BUP_20.yaml') as f:
        #   sem_info = yaml.load(f, Loader=yaml.FullLoader)

        info = {}
        # All available calsses in the scene
        info['class_id_to_name'] = {i:l for i,l in enumerate(class_labels)}
        info['num_classes'] = len(info['class_id_to_name'])
        
        # Present classes id mapping to network output idx
        info['classes_present'] = list(info['class_id_to_name'].keys())
        info['num_present_classes'] = len(info['classes_present'])
        
        # get stuff and things ids and output indices
        info['stuff_ids'] = [info['classes_present'][0]]
        info['things_ids'] = info['classes_present'][1:]

        # Instance id mapping to network output idx
        info['num_instances'] = 200

        return info

    @staticmethod
    def load_scale_and_offset(root, model_rescaling='snap_to_bottom'):
        scale, offset = None, None

        mesh_file = list(Path(root).expanduser().glob('../*.ply'))
        if mesh_file:
            scale, offset = get_scale_from_ply_mesh(mesh_file[0], model_rescaling)     
        if scale is None:
            scale = 1.
        if offset is None:
            offset = [0., 0., -1.4]
        return scale, offset

    @staticmethod
    def load_data(root, split='train', bg_color='white', mip=None,
                load_modes=default_modes, modes_map=default_modes_map,
                scale=None, offset=None,
                add_noise_to_train_poses=False,
                pose_noise_strength=0.01,
                dataset_center_idx=0,
                pose_src='odom',
                max_depth=-1,
                mode='label_window',
                class_labels=[],
                *args, **kwargs):
        """ Replica pre-renders instance datasets lodader

        root folder is separated by modality:
         rgb
         depth
         semantic_calss
         semantic_instance 
         semantic_shuffled

        With each folder has all images with the following structure:
         /path/to/sequence/{modality}/{modality}_{frame_id}.png
        
        Camera trajectories and parameters are stored at:
         /path/to/sequence/traj_w_c.txt
         /path/to/sequence/cam_params.txt
      
        Dataset train, val & test split IDs are specified in file
         /path/to/sequence/info.yaml

        Scene mesh and semantic information available in:
         /path/to/sequence/../mesh.ply
         /path/to/sequence/../info_semantic.json
    
        
        Args:
            root (str): The root directory of the dataset.
            split (str): The dataset split to use from 'train', 'val', 'test'.
            bg_color (str): The background color to use for when alpha=0.
            num_workers (int): The number of workers to use for multithreaded loading. If -1, will not multithread.
            mip: If set, rescales the image by 2**mip.
            modes: modes to be loaded (e.g.: ['imgs', 'depths', 'cam_poses'])
            modes_map: modes name mapping dict from Replica folder naming to this data loader


        Returns:
            (dict of torch.FloatTensors): Different channels of information from NeRF.
        """
        if not load_modes:
            load_modes = default_modes
        
        if not class_labels:
            class_labels = default_class_labels


        preds_name = [m for m in load_modes if 'preds' in m][0]
        # During training do not load GT semantics if semantic predictions were requested 
        if split == 'train' and any('preds' in m for m in load_modes):
            load_modes = [m for m in load_modes if m in ['semantics', 'instance', 'instance_shuffled', 'sem_conf', 'inst_conf']]

        # During validation load GT semantics only 
        elif split in ['val', 'test']:
            load_modes = [m for m in load_modes if 'preds' not in m]

        root = Path(root).expanduser()
        resize_factor = 2**mip

        if mode == 'label_window':
            dataset_class = BUP20SequenceDataset
        elif mode == 'all_frames_window':
            dataset_class = BUP20InferenceDataset
        else:
            raise NotImplementedError(f'Dataset mode "{mode}" not implemented, only ["train", "inference"] available.')

        data = dataset_class(root / 'BUP_20.json',
                             subset=split,
                             seq_num_frames=40,
                             odom_src=pose_src,
                             preds_rel_path=preds_name,
                             max_depth=max_depth,
                             class_labels=class_labels,
                             )[dataset_center_idx]
        

        filenames = [d['file_names'] for d in data]
        poses_ts = [d['odom_ts'] for d in data]
        poses = torch.stack([d['odom'] for d in data])
        poses = transform_cv_to_gl_poses(poses)

        modes = {'imgs': torch.stack([d['rgb'] for d in data])}
       
        h,w = (int(modes['imgs'].shape[-2] // resize_factor),
               int(modes['imgs'].shape[-1] // resize_factor))

        modes['imgs'] = torch.nn.functional.interpolate(modes['imgs'], (h,w), mode='bilinear')
        # change to [b,h,w,c]
        modes['imgs'] = torch.moveaxis(modes['imgs'], 1, -1)
                
        if 'depths' in load_modes:
            modes['depths'] = torch.stack([d['depth'] for d in data])
            modes['depths'] = torch.nn.functional.interpolate(modes['depths'][:,None], (h,w), mode='bilinear')
            modes['depths'] = torch.moveaxis(modes['depths'], 1, -1)

        if 'semantics' in load_modes:
            modes['semantics'] = torch.stack([d['semantics'] for d in data])
            modes['semantics'] = torch.nn.functional.interpolate(modes['semantics'][:,None].type(torch.float), (h,w),
                                                                 mode='nearest').type(torch.LongTensor)
            modes['semantics'] = torch.moveaxis(modes['semantics'], 1, -1)
            

            modes['semantics_pred'] = torch.stack([d['semantics_pred'] for d in data])
            modes['semantics_pred'] = torch.nn.functional.interpolate(modes['semantics_pred'][:,None].type(torch.float), (h,w),
                                                                 mode='nearest').type(torch.LongTensor)
            modes['semantics_pred'] = torch.moveaxis(modes['semantics_pred'], 1, -1)

        if 'sem_conf' in load_modes:
            modes['sem_conf'] = torch.stack([d['sem_conf'] for d in data])
            modes['sem_conf'] = torch.nn.functional.interpolate(modes['sem_conf'], (h,w), mode='bilinear')
            modes['sem_conf'] = torch.moveaxis(modes['sem_conf'], 1, -1)

        if 'instance' in load_modes:
            modes['instance'] = torch.stack([d['imap'] for d in data])
            modes['instance'] = torch.nn.functional.interpolate(modes['instance'][:,None].type(torch.float), (h,w),
                                                                mode='nearest').type(torch.LongTensor)
            modes['instance'] = torch.moveaxis(modes['instance'], 1, -1)

            modes['instance_pred'] = torch.stack([d['imap_pred'] for d in data])
            modes['instance_pred'] = torch.nn.functional.interpolate(modes['instance_pred'][:,None].type(torch.float), (h,w),
                                                                mode='nearest').type(torch.LongTensor)
            modes['instance_pred'] = torch.moveaxis(modes['instance_pred'], 1, -1)

        if 'inst_conf' in load_modes:
            modes['inst_conf'] = torch.stack([d['inst_conf'] for d in data])
            modes['inst_conf'] = torch.nn.functional.interpolate(modes['inst_conf'], (h,w), mode='bilinear')
            modes['inst_conf'] = torch.moveaxis(modes['inst_conf'], 1, -1)    

        # rescale cam parameters acoording to samples rezising
        cam_params = data[0]['intrinsics'] / float(resize_factor)
        fx, fy, x0, y0 = (cam_params[0,0].item(),
                          cam_params[1,1].item(),
                          cam_params[0,2].item() - (w//2),
                          cam_params[1,2].item() - (h//2),
                         )

        #scale and offset cams aiming to fit the scene in the unit cube
        poses[..., :3, 3] *= scale
        poses[..., :3, 3] += torch.Tensor(offset) 

        # nerf-synthetic uses a default far value of 6.0
        default_far = 2 #6.0

        rays = []
        cameras = dict()
        base_camera = Camera.from_args(view_matrix=torch.eye(4),
                                width   = w,        height  = h,
                                focal_x = fx,       focal_y = fy,
                                x0      = x0,       y0      = y0,
                                far=default_far,    near=0.0,
                                dtype=torch.float)        

        ray_grid = generate_centered_pixel_coords(w, h, w, h, device='cuda')           
        base_rays = generate_pinhole_rays(base_camera.to('cuda'), ray_grid)

        opencv_coords = torch.tensor([[-1,  0,  0],
                                      [ 0, -1,  0],
                                      [ 0,  0,  1]])
        for i, pose in enumerate(tqdm(poses, desc='Computing rays')):
           
            # Create camera with image pose 
            camera = deepcopy(base_camera)
            camera.extrinsics.update(pose.type(camera.extrinsics.dtype))
           


            camera.change_coordinate_system(opencv_coords)
            
            if split == 'train' and add_noise_to_train_poses and i>0: # Keep first frame clean to use as anchor
                torch.manual_seed(0)
                camera.extrinsics.rotate( *(pose_noise_strength * (2.0 * torch.rand(3) - 1.0) * 3.14/2)) # random rot [-90,90]
                camera.extrinsics.translate(pose_noise_strength * (2.0 * torch.rand(3) - 1.0).type(torch.float)) # random trans [-1,1]

            cameras[poses_ts[i]] = camera

            # Transform rays from camera to world coordinates
            ray_orig, ray_dir = camera.to('cuda').extrinsics.inv_transform_rays(base_rays.origins, base_rays.dirs)
            ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
            rays.append(Rays(origins=ray_orig[0], dirs=ray_dir[0],
                             dist_min=camera.near, dist_max=camera.far).reshape(h,w,3).to('cpu'))
        
        base_rays = base_rays.reshape(h,w,3).to('cpu')
        rays = Rays.stack(rays).to(dtype=torch.float)

        alpha = modes['imgs'][... ,3:4]
        modes['imgs'] = modes['imgs'][... ,:3]

        if alpha.numel() == 0:
            masks = torch.ones_like(modes['imgs'][... ,0:1]).bool()
        else:
            masks = (alpha > 0.5).bool()

            if bg_color == 'black':
                modes['imgs'][... ,:3] -= ( 1 -alpha)
                modes['imgs'] = np.clip(modes['imgs'], 0.0, 1.0)
            else:
                modes['imgs'][... ,:3] *= alpha
                modes['imgs'][... ,:3] += ( 1 -alpha)
                modes['imgs'] = np.clip(modes['imgs'], 0.0, 1.0)
                
        modes.update({"masks": masks,
                      "rays": rays,
                      "cameras": cameras,
                      "cameras_ts":poses_ts,
                      "base_rays":base_rays,
                      "filenames":filenames})
        
        return modes
