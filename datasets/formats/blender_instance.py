# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import math
import cv2
from tqdm import tqdm
import numpy as np
import skimage
import torch
from torch.multiprocessing import Pool
from kaolin.render.camera import Camera, blender_coords
from wisp.core import Rays
from wisp.ops.raygen import generate_pinhole_rays, generate_centered_pixel_coords
from wisp.ops.image import resize_mip
from pathlib import Path
import h5py
from plyfile import PlyData
from .format_base import DatasetFormatBase

from datasets.utils import get_scale_from_ply_mesh

""" A module for loading data files generated with blenderproc."""

# Default image modes map between blender instance and wisp
default_modes_map = {'colors':'imgs',
                     'depth':'depths',
                     'category_id_segmaps':'semantics',
                     'instance_segmaps':'instance',
                     'instance_shuff_segmaps':'instance_shuffled',
                     'cam_pose':'cam_poses',
                     'cam_params':'cam_params'
                     }

default_modes = list(default_modes_map.values())

class BlenderInstance(DatasetFormatBase):

    # Local function for multiprocess. Just takes a frame from the JSON to load images and poses.
    @staticmethod
    def _load_imgs(frame, root, mip, load_modes, modes_map) :
        """Helper for multiprocessing for the standard dataset. Should not have to be invoked by users.

        Args:
            root: The root of the dataset.
            frame: The frame object from the transform.json.
            mip: If set, rescales the image by 2**mip.
            modes: modes to be loaded (e.g.: ['imgs', 'depths', 'cam_poses'])
            modes_map: modes name mapping dict from blenderproc naming to this data loader

        Returns:
            (dict):frame_files Dictionary of the image modes (e.g.: color, depth, semantics), camera pose and intrinsics.
        """
        out_dict = {'basename':frame.split('.')[0]}
        
        with h5py.File(str(Path(root) / frame), 'r') as f:
            for mode, data in f.items():
                if 'segmaps' in mode:
                    img_conv_fn = lambda img: np.array(img, dtype=np.int)
                    interp_fn = cv2.INTER_NEAREST
                    tensorize_fn = torch.LongTensor
                else:
                    img_conv_fn = skimage.img_as_float32
                    interp_fn = cv2.INTER_AREA
                    tensorize_fn = torch.FloatTensor

                if mode in modes_map:
                    mode = modes_map[mode]

                if load_modes[0] != 'all' and mode not in load_modes:
                    continue
                # Simply convert camera parameters to tensors
                if 'cam' in mode:
                    out_dict[mode] = torch.FloatTensor(np.array(data))
                    continue
                
                # Convert image mode data to tensor and interpolate if requested
                data = img_conv_fn(data)                
                if mip is not None:
                    data = resize_mip(data, mip, interpolation=interp_fn)
                if len(data.shape) < 3:
                    data = data[...,None]
                out_dict[mode] = tensorize_fn(data)
        
        return out_dict

    @staticmethod
    def _parallel_load_imgs(args):
        """Internal function for multiprocessing.
        """
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(1)
        result = BlenderInstance._load_imgs(args['frame'], args['root'],
                                            mip=args['mip'], load_modes=args['load_modes'], modes_map=args['modes_map'])
        if result is None:
            return dict(basename=None, img=None, pose=None)
        else:
            return result

    @staticmethod
    def get_semantic_info(root):

        info = {}
        # All available calsses in the scene
        info['class_id_to_name'] = {0:'no_class'}
        info['class_id_to_name'].update({i+1:c for i,c in enumerate(['icosphere', 'cube', 'monkey', 'cylinder', 'plane'])})
        info['num_classes'] = len(info['class_id_to_name'])
        
        # Present classes id mapping to network output idx
        info['classes_present'] = list(info['class_id_to_name'].keys())
        info['num_present_classes'] = len(info['classes_present'])
        
        # get stuff and things ids and output indices
        info['stuff_ids'] = [id for id, name in info['class_id_to_name'].items() if name in ['plane']]
        info['things_ids'] = [id for id, name in info['class_id_to_name'].items() if name in ['cube', 'cylinder', 'icosphere', 'monkey']]

        # Instance id mapping to network output idx
        info['inst_id_to_class'] = {1:3,2:3,
                                    3:1,4:1,5:1,
                                    6:4,7:4,8:4,
                                    9:5,10:2}
        info['num_instances'] = len(info['inst_id_to_class'])

        return info

    @staticmethod
    def load_data(root, split='train', bg_color='white', num_workers=-1, mip=None,
                load_modes=default_modes, modes_map=default_modes_map,
                scale=None, offset=None,
                *args, **kwargs):
        """ blender proc instance datasets lodader

        root folder separated in train, val & eval splits

        with each multi modality sample file stored in:

        /path/to/dataset/{split}/{frame_num}.hdf5

        Args:
            root (str): The root directory of the dataset.
            split (str): The dataset split to use from 'train', 'val', 'test'.
            bg_color (str): The background color to use for when alpha=0.
            num_workers (int): The number of workers to use for multithreaded loading. If -1, will not multithread.
            mip: If set, rescales the image by 2**mip.
            modes: modes to be loaded (e.g.: ['imgs', 'depths', 'cam_poses'])
            modes_map: modes name mapping dict from blenderproc naming to this data loader


        Returns:
            (dict of torch.FloatTensors): Different channels of information from NeRF.
        """
        if not load_modes:
            load_modes = default_modes

        root = Path(root).expanduser()

        frame_files = list((Path(root) / split).glob('*.hdf5'))

        basenames = []
        modes = {}
        cams = {}
        
        # Always load cam poses and params to compute rays
        load_modes = list(set(load_modes + ['cam_pose', 'cam_params']))

        if num_workers > 0:
            # threading loading images

            p = Pool(num_workers)
            try:
                iterator = p.imap(BlenderInstance._parallel_load_imgs,
                    [dict(frame=str(frame), root=root, mip=mip, load_modes=load_modes, modes_map=modes_map) for frame in frame_files])
                for _ in tqdm(range(len(frame_files))):
                    result = next(iterator)
                    for mode,data in result.items():
                        if data is None:
                            continue
                        if mode == 'basename':
                            basenames.append(data)
                            continue
                        if 'cam' in mode:
                            out = cams
                        else:
                            out = modes
                        if mode not in out:
                            out[mode] = [data]
                        else:
                            out[mode].append(data)
            finally:
                p.close()
                p.join()
        else:
            for frame in tqdm(frame_files, desc='loading data'):
                result = BlenderInstance._load_imgs(str(frame), root, mip=mip, load_modes=load_modes, modes_map=modes_map)
                for mode,data in result.items():
                    if data is None:
                        continue
                    if mode == 'basename':
                        basenames.append(data)
                        continue
                    if 'cam' in mode:
                        out = cams
                    else:
                        out = modes
                    if mode not in out:
                        out[mode] = [data]
                    else:
                        out[mode].append(data)
        
        # Stack data samples
        for mode, data in modes.items():
            modes[mode] = torch.stack(data)

        for mode, data in cams.items():
            cams[mode] = torch.stack(data)

        semantic_info = BlenderInstance.get_semantic_info(root)
        stuff_ids = semantic_info['stuff_ids'] if 'stuff_ids' in semantic_info else None

        if 'semantics' in modes and 'instance' in modes:
            stuff_mask = sum(modes['semantics']==i for i in stuff_ids).bool()
            modes['instance'][stuff_mask] = 0
            
        if 'semantics' in modes:
            # TODO: make semantic output have all possible outputs and not only the ones present in the secene
            modes['semantics'] = torch.where(torch.isin(modes['semantics'], torch.tensor(semantic_info['classes_present'])),
                                             modes['semantics'],
                                             0)
        
        assert modes, f'Unable to load any images from specified dataset path {root}'
        assert cams, f'Unable to load any camera poses from specified dataset path {root}'

        h, w = list(modes.values())[0].shape[1:3]

        cam_K = cams['cam_params'][0] / float(2**mip)
        fx, fy = (cam_K[0,0], cam_K[1,1])
        x0, y0 = (0, 0)

        poses = cams['cam_poses']

        mesh_file = list(root.glob('./*.ply'))

        if (scale is None or offset is None) and mesh_file:
            mesh_scale, mesh_offset = get_scale_from_ply_mesh(mesh_file[0])
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
        for i in range(list(modes.values())[0].shape[0]):
            view_matrix = torch.zeros_like(poses[i])
            view_matrix[:3, :3] = poses[i][:3, :3].T
            view_matrix[:3, -1] = torch.matmul(-view_matrix[:3, :3], poses[i][:3, -1])
            view_matrix[3, 3] = 1.0
            camera = Camera.from_args(view_matrix=view_matrix,
                                    focal_x=fx,
                                    focal_y=fy,
                                    width=w,
                                    height=h,
                                    far=default_far,
                                    near=0.0,
                                    x0=x0,
                                    y0=y0,
                                    dtype=torch.float64)
            camera.change_coordinate_system(blender_coords())
            cameras[basenames[i]] = camera
            ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
                                                    camera.width, camera.height, device='cuda')
            rays.append \
                (generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid).reshape(camera.height, camera.width, 3).to
                    ('cpu'))

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
                
        modes.update({"masks": masks, "rays": rays, "cameras": cameras})

        return modes
