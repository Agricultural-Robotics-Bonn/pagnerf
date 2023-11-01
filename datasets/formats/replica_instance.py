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
from kaolin.render.camera import Camera, blender_coords
from wisp.core import Rays
from wisp.ops.raygen import generate_pinhole_rays, generate_centered_pixel_coords
from wisp.ops.image import resize_mip
from pathlib import Path
import yaml
import json
from .format_base import DatasetFormatBase
import pickle

from datasets.utils import get_scale_from_ply_mesh, transform_cv_to_gl_poses

from .categories import stuff_class_names, things_class_names,  AD20K_to_replica_class_ids

""" A module for loading Replica dataset files generated with habitat-sim."""

# Default image modes map between Replica folders and wisp
default_modes_map = {'rgb':'imgs',
                     'depth':'depths',
                     'semantic_class':'semantics',
                     'semantic_instance':'instance',
                     'semantic_shuffled':'instance_shuffled',
                     'panoptic_preds':'panoptic_preds',
                     'panoptic_preds_coco_mask2former':'panoptic_preds_coco_mask2former',
                     'panoptic_preds_ade20k_mask2former':'panoptic_preds_ade20k_mask2former'
                     }

default_modes = list(default_modes_map.values())

class ReplicaInstance(DatasetFormatBase):

    # Local function for multiprocess. Just takes a frame from the JSON to load images and poses.
    @staticmethod
    def _load_imgs(frame_id, root, mip, load_modes, modes_map) :
        """Helper for multiprocessing for the standard dataset. Should not have to be invoked by users.

        Args:
            frame: Frame id to load.
            root: The root of the dataset.
            mip: If set, rescales the image by 2**mip.
            modes: modes to be loaded (e.g.: ['imgs', 'depths', 'cam_poses'])
            modes_map: modes name mapping dict from replica folder naming to this data loader

        Returns:
            (dict):frame_files Dictionary of the image modes (e.g.: color, depth, semantics), camera pose and intrinsics.
        """
        assert len(['panoptic' in mode for mode in load_modes]) > 1, \
            "Error: Specified to load multiple panoptic predictions, but only one can be loaded per experiment."
        
        out_dict = {}
        stuff_mask = torch.tensor([])
        for mode in modes_map:
            # load only specified modes
            if modes_map[mode] not in load_modes:
                continue
            
            if 'panoptic' in mode:
                with open(list((Path(root) / mode).glob(f'panoptic*_{frame_id}.pkl'))[0], 'rb') as f:
                    preds = pickle.load(f)
                
                # load semantics
                sem_preds = preds['sem_seg']['preds']
                sem_conf = preds['sem_seg']['confidence']
                if mip is not None:
                    sem_preds = resize_mip(sem_preds, mip, interpolation=cv2.INTER_NEAREST)
                    sem_conf = resize_mip(sem_conf, mip, interpolation=cv2.INTER_NEAREST)
                sem_preds = sem_preds[...,None]
                sem_preds = torch.LongTensor(sem_preds.astype(np.float32))
                # Map AD20K orphan classes to the no_object
                AD20K_valid_ids = torch.tensor(list(AD20K_to_replica_class_ids))
                sem_preds = torch.where(torch.isin(sem_preds, AD20K_valid_ids), sem_preds, 0)
                # Map all the valid classes present in Replica
                sem_preds = sem_preds.apply_(AD20K_to_replica_class_ids.get)
                out_dict['semantics'] = sem_preds

                sem_conf = sem_conf[...,None]
                sem_conf = torch.FloatTensor(sem_conf)
                sem_conf = torch.where(torch.isin(sem_preds, AD20K_valid_ids), sem_conf, 0.05)
                out_dict['sem_conf'] = sem_conf
                
                # load instances
                imap = preds['instances']['imap']
                if mip is not None:
                    imap = resize_mip(imap, mip, interpolation=cv2.INTER_NEAREST)
                imap = imap[...,None]
                imap = torch.LongTensor(imap.astype(np.float32))
                out_dict['instance'] = imap
                # out_dict['inst_conf'] = torch.clone(imap).type(torch.float32).apply_(lambda x: preds['instances']['scores'][int(x)])
                out_dict['inst_conf'] = torch.where(imap != 0, 1.0, 0.005)


                continue

            if 'depth' in mode:
                path = list((Path(root) / mode).glob(f'depth*_{frame_id}.png'))[0]
                read_flag = cv2.IMREAD_UNCHANGED
                interp_fn = cv2.INTER_LINEAR
                # convert depth from mm to m
                tensorize_fn = lambda x: torch.FloatTensor(x.astype(np.float32) / 1000.)
            elif 'semantic' in mode:
                path = list((Path(root) / mode).glob(f'semantic*_{frame_id}.png'))[0]
                read_flag = cv2.IMREAD_UNCHANGED
                interp_fn = cv2.INTER_NEAREST
                tensorize_fn = lambda x: torch.LongTensor(x.astype(np.float32))
            else:
                path = Path(root) / mode / f'{mode}_{frame_id}.png'
                read_flag = cv2.IMREAD_COLOR
                interp_fn = cv2.INTER_LINEAR
                # convert from cv2 default BGR to RGB
                tensorize_fn = lambda x: torch.FloatTensor(x[...,::-1].astype(np.float32) / 255.0)
            
            data = cv2.imread(str(path), read_flag)
            
            # Convert image mode data to tensor and interpolate if requested
            if mip is not None:
                data = resize_mip(data, mip, interpolation=interp_fn)
            if len(data.shape) < 3:
                data = data[...,None]

            t_data = tensorize_fn(data)
            
            out_dict[modes_map[mode]] = t_data
        
        return out_dict

    @staticmethod
    def _parallel_load_imgs(args):
        """Internal function for multiprocessing.
        """
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(1)
        result = ReplicaInstance._load_imgs(args['frame'], args['root'],
                                            mip=args['mip'], load_modes=args['load_modes'], modes_map=args['modes_map'])
        if result is None:
            return dict(basename=None, img=None, pose=None)
        else:
            return result

    @staticmethod
    def get_semantic_info(root):

        with open(Path(root).expanduser().parent / 'info_semantic.json') as f:
            sem_info = json.load(f)

        info = {}
        # All available calsses in the scene
        info['class_id_to_name'] = {0:'no_class'}
        info['class_id_to_name'].update({c['id']:c['name'] for c in sem_info['classes']})
        info['num_classes'] = len(info['class_id_to_name'])
        
        # Present classes id mapping to network output idx
        info['classes_present'] = list(set([o['class_id'] if o['class_id']>0 else 0 for o in sem_info['objects']]))
        info['num_present_classes'] = len(info['classes_present'])
        
        # get stuff and things ids and output indices
        info['stuff_ids'] = [id for id, name in info['class_id_to_name'].items() if name in stuff_class_names]
        info['things_ids'] = [id for id, name in info['class_id_to_name'].items() if name in things_class_names]

        # Instance id mapping to network output idx
        info['inst_id_to_class'] = {o['id']:o['class_id'] if (o['class_id']>0 and o['class_id'] in info['things_ids']) else 0 for o in sem_info['objects']}
        info['num_instances'] = len(info['inst_id_to_class'])

        return info

    @staticmethod
    def load_data(root, split='train', bg_color='white', num_workers=-1, mip=None,
                load_modes=default_modes, modes_map=default_modes_map,
                scale=None, offset=None,
                model_rescaling='snap_to_bottom',
                add_noise_to_train_poses=False,
                pose_noise_strength=0.01,
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

        # During training do not load GT semantics if semantic predictions were requested 
        if split == 'train' and any('preds' in m for m in load_modes):
            load_modes = [m for m in load_modes if m not in ['semantics', 'instance', 'instance_shuffled']]

        # During validation load GT semantics only 
        if split in ['val', 'test']:
            load_modes = [m for m in load_modes if 'preds' not in m]

        root = Path(root).expanduser()

        with open(root / 'info.yaml', "r") as f:
            split_ids = yaml.safe_load(f)['splits'][split]
        
        with open(root / 'cam_params.yaml', "r") as f:
            cam_params = yaml.safe_load(f)
        # rescale cam parameters acoording to samples rezising
        cam_params = {k:v / float(2**mip) for k,v in cam_params.items()}

        with open(root / 'traj_w_c.txt', "r") as f:
            poses = torch.FloatTensor(np.loadtxt(f, delimiter=" ").reshape(-1, 4, 4)[split_ids]) 
        poses = transform_cv_to_gl_poses(poses)

        # Always load cam poses and params to compute rays
        load_modes = list(set(load_modes + ['cam_pose', 'cam_params']))

        modes = {}
        if num_workers > 0:
            # threading loading images
            p = Pool(num_workers)
            try:
                iterator = p.imap(ReplicaInstance._parallel_load_imgs,
                    [dict(frame=str(frame), root=root, mip=mip, load_modes=load_modes, modes_map=modes_map) for frame in split_ids])
                for _ in tqdm(range(len(split_ids)), desc='Loading data'):
                    result = next(iterator)
                    for mode,data in result.items():
                        if data is None:
                          continue
                        if mode not in modes:
                            modes[mode] = [data]
                        else:
                            modes[mode].append(data)
            finally:
                p.close()
                p.join()
        else:
            for frame in tqdm(split_ids, desc='loading data'):
                result = ReplicaInstance._load_imgs(str(frame), root, mip=mip, load_modes=load_modes, modes_map=modes_map)
                for mode,data in result.items():
                    if data is None:
                        continue
                    if mode not in modes:
                            modes[mode] = [data]
                    else:
                        modes[mode].append(data)
        
        semantic_info = ReplicaInstance.get_semantic_info(root)
        stuff_ids = semantic_info['stuff_ids'] if 'stuff_ids' in semantic_info else None

        # Stack data samples
        for mode, data in modes.items():
            modes[mode] = torch.stack(data)

        if 'semantics' in modes and 'instance' in modes:
            stuff_mask = sum(modes['semantics']==i for i in stuff_ids).bool()
            modes['instance'][stuff_mask] = 0
            
        if 'semantics' in modes:
            # TODO: make semantic output have all possible outputs and not only the ones present in the secene
            modes['semantics'] = torch.where(torch.isin(modes['semantics'], torch.tensor(semantic_info['classes_present'])),
                                             modes['semantics'],
                                             0)
                
        assert modes, f'Unable to load any images from specified dataset path {root}'

        h, w = list(modes.values())[0].shape[1:3]

        mesh_file = list(root.glob('../*.ply'))
        if (scale is None or offset is None) and mesh_file:
            mesh_scale, mesh_offset = get_scale_from_ply_mesh(mesh_file[0], model_rescaling)
            if scale is None:
                scale = mesh_scale
            if offset is None:
                offset = mesh_offset
        # initialize scale and offset if they're still uninitialized
        if scale is None:
            scale = 1.
        if offset is None:
            offset = [0., 0., 0.]

        #scale and offset cams aiming to fit the scene in the unit cube
        poses[..., :3, 3] *= scale
        poses[..., :3, 3] += torch.Tensor(offset) 

        # nerf-synthetic uses a default far value of 6.0
        default_far = 6.0

        rays = []

        cameras = dict()

        base_camera = Camera.from_args(view_matrix=torch.eye(4),
                                focal_x=cam_params['fx'],
                                focal_y=cam_params['fy'],
                                width=w,
                                height=h,
                                far=default_far,
                                near=0.0,
                                x0=0.0,
                                y0=0.0,
                                dtype=torch.float64)        

        ray_grid = generate_centered_pixel_coords(w, h, w, h, device='cuda')           
        base_rays = generate_pinhole_rays(base_camera.to(ray_grid[0].device), ray_grid)
        for i in tqdm(range(len(split_ids)), desc='Computing rays'):
            view_matrix = torch.zeros_like(poses[i])
            view_matrix[:3, :3] = poses[i][:3, :3].T
            view_matrix[:3, -1] = torch.matmul(-view_matrix[:3, :3], poses[i][:3, -1])
            view_matrix[3, 3] = 1.0

            # Create camera with image pose 
            camera = deepcopy(base_camera)
            camera.extrinsics.update(view_matrix.type(camera.extrinsics.dtype))
            camera.change_coordinate_system(blender_coords())

            if split == 'train' and add_noise_to_train_poses and i>0: # Keep first frame clean to use as anchor
                torch.manual_seed(0)
                camera.extrinsics.rotate( *(pose_noise_strength * (2.0 * torch.rand(3) - 1.0) * 3.14/2)) # random rot [-90,90]
                camera.extrinsics.translate(pose_noise_strength * (2.0 * torch.rand(3) - 1.0).type(torch.float64)) # random trans [-1,1]

            cameras[str(split_ids[i])] = camera

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
                
        modes.update({"masks": masks, "rays": rays, "cameras": cameras, "base_rays":base_rays})
        
        return modes
