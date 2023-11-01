# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging as log
import time
import math

from wisp.utils import PsDebugger, PerfTimer
from wisp.ops.spc import sample_spc

import wisp.ops.spc as wisp_spc_ops
import wisp.ops.grid as grid_ops

from wisp.models.grids import BLASGrid
from wisp.models.decoders import BasicDecoder

import kaolin.ops.spc as spc_ops

from wisp.accelstructs import OctreeAS

class Occtree(BLASGrid):
    """This is a feature grid where the features are defined in a codebook that is hashed.
    """

    def __init__(self, 
        blas_level         : int   = 7,
        **kwargs
    ):
        """Initialize the hash grid class.

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
            interpolation_type (str): The type of interpolation function.
            multiscale_type (str): The type of multiscale aggregation. Usually 'sum' or 'cat'.
                                   Note that 'cat' will change the decoder input dimension.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The mean of the Gaussian distribution.
            codebook_bitwidth (int): The bitwidth of the codebook.
            blas_level (int): The level of the octree to be used as the BLAS.
        
        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        self.blas_level = blas_level

        self.kwargs = kwargs
    
        self.blas = OctreeAS()
        self.blas.init_dense(self.blas_level)
        self.dense_points = spc_ops.unbatched_get_level_points(self.blas.points, self.blas.pyramid, self.blas_level).clone()
        self.num_cells = self.dense_points.shape[0]
        self.occupancy = torch.zeros(self.num_cells)
        self.num_lods = 1
        self.active_lods = [0]

        self._register_blas_buffers()

    def _register_blas_buffers(self):
        # register grid accelerator for save/load operations
        self.register_buffer('blas_octree', self.blas.octree)
        self.register_buffer('blas_points', self.blas.points)
        self.register_buffer('blas_prefix', self.blas.prefix)
        self.register_buffer('blas_pyramid', self.blas.pyramid)
    
    def blas_init(self, octree):
        self.blas.init(octree)
        self._register_blas_buffers()

    def freeze(self):
        """Freezes the feature grid.
        """
        self.codebook.requires_grad_(False)

    def raymarch(self, rays, level=None, num_samples=64, raymarch_type='voxel'):
        """Mostly a wrapper over OctreeAS.raymarch. See corresponding function for more details.

        Important detail: the OctreeGrid raymarch samples over the coarsest LOD where features are available.
        """
        return self.blas.raymarch(rays, level=self.blas_level, num_samples=num_samples,
                                  raymarch_type=raymarch_type)
