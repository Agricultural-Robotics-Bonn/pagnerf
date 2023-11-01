# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch


class SampleRays:
    """ A dataset transform for sub-sampling a fixed amount of rays. """
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, inputs):
        rays = inputs['rays']

        if len(rays.shape) == 1:
            # single image ray sampling
            ray_idx = torch.randperm(
                inputs['imgs'].shape[0],
                device=rays.dirs.device)[:self.num_samples]
        
        elif len(rays.shape) == 2:
            # batch rays sampling mode
            b_size = rays.shape[0]
            rand = torch.rand(*rays.shape)
            b_idx = rand.argsort(dim=1)[:,:self.num_samples//b_size]
            ray_idx = [torch.arange(b_idx.shape[0])[:, None], b_idx]

        else:
            raise NotImplementedError("raysampling only implemented for single image and batch")

        out = {}
        for mode in [k for k in inputs if k not in ['cameras', 'cameras_ts', 'filenames']]:
            out[mode] = inputs[mode][ray_idx].contiguous()

        return out
