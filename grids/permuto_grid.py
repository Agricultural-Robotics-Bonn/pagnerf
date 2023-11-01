# Wrapper for Permutohedral Grid Encoder by Alexandru Radu Rosu
# https://github.com/RaduAlexandru/permutohedral_encoding

import torch
from torch import nn
import logging as log
import numpy as np

from wisp.models.grids import HashGrid
from permutohedral_encoding import PermutoEncoding


class PermutoGrid(HashGrid):
    """This is a feature grid where the features are defined in a codebook that is hashed.
    """
    def __init__(self,*args,
                 coarsest_scale = 1.0,
                 finest_scale   = 0.001,
                 capacity_log_2 = 18,
                 num_lods       = 24,    
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self._register_blas_buffers()
        
        self.coarsest_scale = coarsest_scale
        self.finest_scale = finest_scale
        self.capacity = pow(2,capacity_log_2)
        self.num_lods = num_lods

        self.multiscale_type = 'cat'
    
    def _register_blas_buffers(self):
        # register grid accelerator for save/load operations
        self.register_buffer('blas_octree', self.blas.octree)
        self.register_buffer('blas_points', self.blas.points)
        self.register_buffer('blas_prefix', self.blas.prefix)
        self.register_buffer('blas_pyramid', self.blas.pyramid)

    def blas_init(self, octree):
        self.blas.init(octree)
        self._register_blas_buffers()

    def set_capacity(self, capacity_log_2):
        self.capacity = pow(2,capacity_log_2)

    def init_from_scales(self):
        """Build a multiscale hash grid from a list of resolutions.
        """
        self.active_lods = [x for x in range(self.num_lods)]
        self.max_lod = self.num_lods - 1

        self.resolutions=np.geomspace(self.coarsest_scale, self.finest_scale, num=self.num_lods)
        log.info(f"Active Resolutions: {self.resolutions}")
        

        self.embedder = PermutoEncoding(
            3, # In pos dimension
            self.capacity,
            self.num_lods,
            self.feature_dim,
            self.resolutions)


    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def interpolate(self, coords, lod_idx=None, pidx=None):
        
        if coords.numel() == 0:
            return torch.empty([0,1,self.num_lods * self.feature_dim], device=coords.device)
        
        return self.embedder(coords.reshape(-1,3).type(torch.float))