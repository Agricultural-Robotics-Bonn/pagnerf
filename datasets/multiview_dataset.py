# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from typing import Callable
import torch
from torch.utils.data import Dataset
from wisp.core import Rays


class MultiviewDataset(Dataset):
    """This is a static multiview image dataset class.

    This class should be used for training tasks where the task is to fit a static 3D volume from
    multiview images.

    TODO(ttakikawa): Support single-camera dynamic temporal scenes, and multi-camera dynamic temporal scenes.
    TODO(ttakikawa): Currently this class only supports sampling per image, not sampling across the entire
                     dataset. This is due to practical reasons. Not sure if it matters...
    """

    def __init__(self, 
        dataset_path             : str,
        multiview_dataset_format : str      = 'standard',
        mip                      : int      = None,
        bg_color                 : str      = None,
        dataset_num_workers      : int      = -1,
        load_modes               : list     = [],
        class_labels             : list     = [],
        transform                : Callable = None,
        scale                    : float    = None,
        offset                   : list     = None,
        model_rescaling          : str      = 'snap_to_bottom',
        add_noise_to_train_poses : bool     = False,
        pose_noise_strength      : float    = 0.01,
        dataset_center_idx       : int      = 0,
        split                    : str      = 'train',
        pose_src                 : str      = 'odom',
        max_depth                : float    = -1,
        dataset_mode             : str      = 'label_window',
        **kwargs
    ):
        """Initializes the dataset class.

        Note that the `init` function to actually load images is separate right now, because we don't want 
        to load the images unless we have to. This might change later.

        Args: 
            dataset_path (str): Path to the dataset.
            multiview_dataset_format (str): The dataset format. Currently supports standard (the same format
                used for instant-ngp) and the RTMV dataset.
            mip (int): The factor at which the images will be downsampled by to save memory and such.
                       Will downscale by 2**mip.
            bg_color (str): The background color to use for images with 0 alpha.
            dataset_num_workers (int): The number of workers to use if the dataset format uses multiprocessing.
        """
        self.root = dataset_path
        self.mip = mip
        self.bg_color = bg_color
        self.dataset_num_workers = dataset_num_workers
        self.transform = transform
        self.multiview_dataset_format = multiview_dataset_format
        self.load_modes = load_modes
        self.class_labels = class_labels
        self.scale = scale
        self.offset = offset
        self.model_rescaling = model_rescaling
        
        self.add_noise_to_train_poses = add_noise_to_train_poses
        self.pose_noise_strength = pose_noise_strength

        self.mesh_path = None 
        self.semantic_info = None

        self.dataset_center_idx = dataset_center_idx
        self.dataset_mode = dataset_mode
        self.pose_src = pose_src

        self.max_depth = max_depth

        self.split = split
        
        # load the requested dataset format parser
        if self.multiview_dataset_format in ['standard', 'NeRFStandard','nerf_standard']:
            from .formats.nerf_standard import NeRFStandard
            self.dataset_format = NeRFStandard
        elif self.multiview_dataset_format in ['replica', 'ReplicaInstance', 'replica_instance']:
            from .formats.replica_instance import ReplicaInstance
            self.dataset_format = ReplicaInstance
        elif self.multiview_dataset_format in ['bup20', 'bup_20', 'BUP20', 'BUP_20']:
            from .formats.bup20 import BUP20
            self.dataset_format = BUP20
        elif self.multiview_dataset_format in ['sb20', 'sb_20', 'SB20', 'SB_20']:
            from .formats.sb20 import SB20
            self.dataset_format = SB20
        else:
            raise ImportError(f'"{multiview_dataset_format}" multiview dataset format not supported...')

    def init(self):
        """Initializes the dataset.
        """

        # Get image tensors 
        
        self.coords_data = None
        self.coords = None

        if 'load_scale_and_offset' in vars(self.dataset_format):
            loaded_scale, loaded_offset = self.dataset_format.load_scale_and_offset(self.root, model_rescaling=self.model_rescaling)
            if self.scale is None:
                self.scale = loaded_scale
            if self.offset is None:
                self.offset = loaded_offset

        self.data = self.get_images(split=self.split)

        self.img_shape = self.data["imgs"].shape[1:3]
        self.num_imgs = self.data["imgs"].shape[0]

        for mode in [k for k in self.data if k not in ['cameras', 'cameras_ts', 'filenames']]:
            if mode == 'base_rays':
                self.data[mode] = self.data[mode].reshape(-1, 3)
            elif mode == 'rays':
                self.data[mode] = self.data[mode].reshape(self.num_imgs, -1, 3)
            else:
                num_channels = self.data[mode].shape[-1]
                self.data[mode] = self.data[mode].reshape(self.num_imgs, -1, num_channels)
        
        if 'get_semantic_info' in vars(self.dataset_format):
            self.semantic_info = self.dataset_format.get_semantic_info(self.root, self.class_labels)

        if 'get_semantic_info' in vars(self.dataset_format):
            self.semantic_info = self.dataset_format.get_semantic_info(self.root, self.class_labels)
        


    def get_images(self, split='train', mip=None):
        """Will return the dictionary of image tensors.

        Args:
            split (str): The split to use from train, val, test
            mip (int): If specified, will rescale the image by 2**mip.

        Returns:
            (dict of torch.FloatTensor): Dictionary of tensors that come with the dataset.
        """
        if mip is None:
            mip = self.mip
        
        data = self.dataset_format.load_data(self.root, split,
                                            bg_color=self.bg_color, num_workers=self.dataset_num_workers, mip=mip,
                                            coords=self.coords_data, load_modes=self.load_modes, scale=self.scale, offset=self.offset,
                                            add_noise_to_train_poses=self.add_noise_to_train_poses,
                                            pose_noise_strength=self.pose_noise_strength,
                                            dataset_center_idx=self.dataset_center_idx,
                                            pose_src=self.pose_src,
                                            max_depth=self.max_depth,
                                            mode=self.dataset_mode,
                                            class_labels=self.class_labels)
                                            
        if 'coords' in data:
            self.coords_data = data['coords']
            self.coords = data['coords']['values']

        return data

    def __len__(self):
        """Length of the dataset in number of rays.
        """
        return self.data["imgs"].shape[0]

    def __getitem__(self, idx : int):
        """Returns a ray.
        """
        out = {}
        for mode in [k for k in self.data if k not in ['cameras', 'base_rays']]:
            out[mode] = self.data[mode][idx]
        
        if 'base_rays' in self.data:
            out['base_rays'] = self.data['base_rays']

        if self.transform is not None:
            out = self.transform(out)
        
        out['cam_id'] = self.data['cameras_ts'][idx] if 'cameras_ts' in self.data else idx
        out['filename'] = self.data['filenames'][idx] if 'filenames' in self.data else ''
        return out
