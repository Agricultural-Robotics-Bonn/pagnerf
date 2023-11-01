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

import wisp.ops.spc as wisp_spc_ops

from wisp.models.grids import BLASGrid
from wisp.models.decoders import BasicDecoder

from wisp.accelstructs import OctreeAS
import kaolin.ops.spc as spc_ops
import kaolin.render.spc as spc_render

class VMSplitFeatureVolume(nn.Module):
    """Vector Matrix feature module implemented with an lod of grid_sample.
    As introduces in TensoRF:
    https://arxiv.org/abs/2203.09517
    """
    def __init__(self, density_n_comp, app_n_comp, res):
        """Initializes the feature volume.

        Args:
            fdim (int): The feature dimension.
            fsize (int): The height and width of the texture map.
            std (float): The standard deviation for the Gaussian initialization.
            bias (float): The mean for the Gaussian initialization.

        Returns:
            (void): Initializes the feature volume.
        """
        super().__init__()

        self.app_n_comp = app_n_comp
        self.density_n_comp = density_n_comp
        self.res = res

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]
        self.app_dim = 27

        self.init_vm_grid()

    def init_vm_grid(self):
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.res, 0.1)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.res, 0.1)
        self.basis_mat = torch.nn.Linear(3 * self.app_n_comp, self.app_dim, bias=False)

    def init_one_svd(self, n_component, gridSize, scale):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component, gridSize, gridSize))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component, gridSize, 1))))

        return torch.nn.ParameterList(plane_coef), torch.nn.ParameterList(line_coef)

    def compute_densityfeature(self, xyz_sampled):        
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature


    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat((plane_coef_point * line_coef_point).T)

    def forward(self, xyz_sampled):
        return self.compute_densityfeature(xyz_sampled), self.compute_appfeature(xyz_sampled)
    
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target, res_target), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target, 1), mode='bilinear', align_corners=True))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

class TensoRF(BLASGrid):
    """This is a feature grid representing the feature tensor as a vector matrix decomposition on a multiresolution pyramid.

    Since the VMgrid support region is bounded by an AABB, this uses an AABB
    as the BLAS. Hence the class is compatible with the usual packed tracers.
    """

    def __init__(self, 
        density_n_comp     : int = 16,
        color_n_comp       : int = 48,
        base_resolution    : int = 128,
        max_resolution     : int = 192,
        num_resolution     : int = 5,
        blas_level         : int = 7,
        **kwargs
    ):
        """Initialize the octree grid class.

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
            base_lod (int): The base LOD of the feature grid. This is the lowest LOD of the SPC octree
                            for which features are defined.
            num_lods (int): The number of LODs for which features are defined. Starts at base_lod.
            interpolation_type (str): The type of interpolation function.
            multiscale_type (str): The type of multiscale aggregation. Usually 'sum' or 'cat'.
                                   Note that 'cat' will change the decoder input dimension.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The mean of the Gaussian distribution.
        
        Returns:
            (void): Initializes the class.
        """
        super().__init__()


        self.density_n_comp = density_n_comp
        self.color_n_comp = color_n_comp
        self.blas_level = blas_level
        
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.num_resolutions = num_resolution
        self.current_resolution = base_resolution
        
        d_res = (max_resolution - base_resolution) // (num_resolution-1)
        self.resolutions = list(range(base_resolution,
                                      max_resolution + d_res,
                                      d_res))
        
        self.num_lods = 1
        self.base_lod = 0
        self.active_lods = [0]
        self.max_lod = 0

        log.info(f"Active LODs: {self.active_lods}")

        self.blas = OctreeAS()
        self.blas.init_dense(self.blas_level)
        self.dense_points = spc_ops.unbatched_get_level_points(self.blas.points, self.blas.pyramid, self.blas_level).clone()
        self.num_cells = self.dense_points.shape[0]
        self.occupancy = torch.zeros(self.num_cells)

        self._register_blas_buffers()

        self._init()

    def _register_blas_buffers(self):
        # register grid accelerator for save/load operations
        self.register_buffer('blas_octree', self.blas.octree)
        self.register_buffer('blas_points', self.blas.points)
        self.register_buffer('blas_prefix', self.blas.prefix)
        self.register_buffer('blas_pyramid', self.blas.pyramid)
    
    def blas_init(self, octree):
        self.blas.init(octree)
        self._register_blas_buffers()

    def _init(self):
        """Initializes everything that is not the BLAS.
        """

        self.features = nn.ModuleList([])
        self.num_feat = 0
        assert len(self.active_lods) == 1, \
            f'TensoRF has only 1 LOD that grows over training, but {len(self.active_lods)} were specified'
        self.features = VMSplitFeatureVolume(self.density_n_comp, self.color_n_comp, self.current_resolution)
        self.num_feat = self.features.density_n_comp + self.features.app_dim * self.features.app_n_comp

        log.info(f"# Feature Vectors: {self.num_feat}")
        
    def freeze(self):
        """Freezes the feature grid.
        """
        self.features.requires_grad_(False)

    def interpolate(self, coords, lod_idx, pidx=None):
        """Query multiscale features.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
            lod_idx  (int): int specifying the index to ``active_lods`` 
            pidx (torch.LongTensor): point_hiearchy indices of shape [batch]. Unused in this function.

        Returns:
            (torch.FloatTensor): interpolated features of shape [batch, num_samples, feature_dim]
        """
        
        batch, num_samples = coords.shape[:2]
        
        return tuple(f.reshape(batch, num_samples, -1) for f in self.features(coords))

    def step_upsample_vm_grid(self):
        cur_res_idx = self.resolutions.index(self.current_resolution)
        if cur_res_idx + 1 < len(self.resolutions):
            self.upsample_vm_grid(self.resolutions[cur_res_idx + 1])
        
    def upsample_vm_grid(self, target_res):
        self.features.upsample_volume_grid(target_res)
        self.current_resolution = target_res
    
    def raymarch(self, rays, level=None, num_samples=64, raymarch_type='voxel'):
        """Mostly a wrapper over OctreeAS.raymarch. See corresponding function for more details.

        Important detail: this is just used as an AABB tracer.
        """
        return self.blas.raymarch(rays, level=self.blas_level, num_samples=num_samples,
                                  raymarch_type=raymarch_type)
    
    def raytrace(self, rays, level=None, with_exit=False):
        """By default, this function will use the equivalent BLAS function unless overridden for custom behaviour.
        
        Important detail: this is just used as an AABB tracer.
        """
        return self.blas.raytrace(rays, level=self.blas_level, with_exit=with_exit)
