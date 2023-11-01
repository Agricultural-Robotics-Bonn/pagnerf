# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import copy

import torch
from wisp.models.nefs import BaseNeuralField
from wisp.models.activations import get_activation_class
from wisp.models.layers import get_layer_class
from wisp.models.embedders import get_positional_embedder
from wisp.models.grids import *
from wisp.ops.geometric import sample_unif_sphere

from .panoptic_nef import PanopticNeF

class PanopticDDensityNeF(PanopticNeF):
    """ An exemplary contrastive for quick creation of new user neural fields.
        Clone this file and modify to create your own customized neural field.
    """
    def __init__(self,
    
        delta_num_layers        : int   = 1,
        delta_hidden_dim        : int   = 64,
        separate_sem_grid       : bool  = False,
        inst_soft_temperature   : float = 0.0,
        
        **kwargs):

        self.delta_num_layers = delta_num_layers
        self.delta_hidden_dim = delta_hidden_dim
        self.separate_sem_grid = separate_sem_grid
        self.inst_soft_temperature = inst_soft_temperature

        super().__init__(**kwargs)

    def init_decoder(self):
        super().init_decoder()
        
        # Delta density Decoder
        #########################
        # d_Dens [-inf, inf]
        # Intended to fix RGB density for better semantic renderings 
        if self.delta_num_layers == 0:
            self.delta_hidden_dim = self.input_dim_density
        self.decoder_delta_density = BasicDecoder(input_dim=self.input_dim_density,
                                        output_dim=1,
                                        activation=get_activation_class('none'),
                                        bias=True,
                                        layer=get_layer_class(self.layer_type),
                                        num_layers=self.delta_num_layers,
                                        hidden_dim=self.delta_hidden_dim,
                                        skip=[])

    def init_grid(self):
        super().init_grid()
        self.delta_grid = copy.deepcopy(self.grid)
        if self.grid_type == "PermutoGrid":
            self.delta_grid.set_capacity(self.kwargs['delta_capacity_log_2'])

    def get_nef_type(self):
        """Returns a text keyword describing the neural field type.

        Returns:
            (str): The key type
        """
        return 'delta_panoptic_nef'

    def prune(self):
        """Prunes the blas based on current state.
        """
        #TODO (csmitt): Prune Main and Delta grids separately according to their density
        #   (would need more engineering in the tracer to choose specific points
        #   from Delta grid)  
        if self.grid is None:
            return
            
        # if not self.grid_type not in ["PermutoGrid","HashGrid"]:
            # raise NotImplementedError

        # TODO: parametrize
        density_decay = 0.6
        min_density = ((0.01 * 512)/np.sqrt(3))
        
        # assume both grids have equal underlying AccGrids, thus samples Main grid only
        points = self.grid.dense_points.cuda()
        res = 2.0**self.grid.blas_level
        samples = torch.rand(points.shape[0], 3, device=points.device)
        samples = points.float() + samples
        samples = samples / res
        samples = samples * 2.0 - 1.0
        sample_views = torch.FloatTensor(sample_unif_sphere(samples.shape[0])).to(points.device)
        
        mask = torch.zeros_like(self.grid.occupancy).type(torch.bool).cuda()
        # Prune both main and delta grid
        for grid, channel in zip([self.grid, self.delta_grid], ['density', 'panoptic_density']):
  
            grid.occupancy = grid.occupancy.cuda()
            grid.occupancy = grid.occupancy * density_decay

            with torch.no_grad():
                density = self.forward(coords=samples[:,None], ray_d=sample_views, channels=channel)
            
            grid.occupancy = torch.stack([density[:, 0, 0], grid.occupancy], -1).max(dim=-1)[0]

            # keep points where (dens_main || dens_delta > min_density)
            mask = torch.logical_or(mask, grid.occupancy > min_density)

        # Prune points and reinitialize both AccGrid
        _points = points[mask]
        for grid in [self.grid, self.delta_grid]:
            octree = spc_ops.unbatched_points_to_octree(_points, grid.blas_level, sorted=True)
            grid.blas.init(octree)


    def register_forward_functions(self):
        """Register the forward functions.
        Forward functions define the named output channels this neural field supports.
        By registering forward functions, a tracer knows which neural field methods to use to obtain channels values.
        """
        # Here the rgba() function handles both the rgb and density channels at the same time
        self._register_forward_function(self.rgb_semantics, ["density", "rgb",
                                                             "delta_density", "panoptic_density",
                                                             "semantics", "inst_embedding"])

    def rgb_semantics(self, coords, ray_d, compute_channels, pidx=None, lod_idx=None):
        """Compute color, density and semantics for the provided coordinates.
        
         Dir[2] -------------------------------+         
                                               |         
                   +----+       +-----+     +-----+      
         Pos[3] ---|grid|---+---| MLP |--+--| MLP |--- RGB[3]
                   +----+   |   +-----+  |  +-----+       
                            |            +------+----- density[1]              
                            |                   |                          
                  +-----+ +---+               +---+                                     
         Pos[3] --|delta|-|sum|--+            |sum|--- panoptic_density[1]                                                           
                  | grid| +---+  |            +---+     
                  +-----+        |   +-----+    |
                                 +---| MLP |----+----- delta_density[1]
                                 |   +-----+   
                                 |   +-----+         
                                 +---| MLP |---------- semantics[num_classes]
                                 |   +-----+               
                                 |                         
                                 |   +-----+               
                                 +---| MLP |---------- inst_embedding[dim_embedding]
                                     +-----+                      

        Args:
            coords (torch.FloatTensor): packed tensor of shape [batch, num_samples, 3]
            ray_d (torch.FloatTensor): packed tensor of shape [batch, 3]
            pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                     Unused in the current implementation.
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
            channels (list(str)): list of channels to compute

        Returns:
            {"rgb": torch.FloatTensor, "density": torch.FloatTensor, 'semantics': torch.FloatTensor, 'inst_embeddings': torch.FloatTensor}:
                - RGB tensor of shape [batch, num_samples, 3] 
                - Density tensor of shape [batch, num_samples, 1]
                - semantic tensor of shape [batch, num_samples, num_classes]
                - ints_embedding tensor of shape [batch, num_samples, embedding_dim]
        """
        out_dict = {}
        if not compute_channels:
            return out_dict
        
        timer = PerfTimer(activate=False, show_memory=True)
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, num_samples, _ = coords.shape

        timer.check("rf_rgba_preprocess")

        ###########################################################################################################
        # Positional grid encoding
        ###########################################################################################################
        # Querry RGB grid
        feats = self.grid.interpolate(coords, lod_idx).squeeze()
        feats = feats * self.lod_weights.to(feats.device)
        if self.multiscale_type == 'sum':
            feats = feats.reshape(-1, lod_idx + 1, feats.shape[-1] // (lod_idx + 1)).sum(-2)
        timer.check("rf_rgba_interpolate")
        
        if self.position_input:
            raise NotImplementedError

        ###########################################################################################################
        # Density decoding
        ###########################################################################################################
        if any([c in compute_channels for c in ['density', 'rgb']]) or\
            'panoptic_density' in compute_channels and not self.separate_sem_grid: 
            # Decode high-dimensional vectors to RGBA.
            density_feats = self.decoder_density(feats)
            timer.check("rf_density_decode")
            
            # Density is [particles / meter], so need to be multiplied by distance
            density = torch.relu(density_feats[...,0:1]).reshape(batch, num_samples, 1)
            timer.check("rf_density_normalization")
            if 'density' in compute_channels:
                out_dict['density'] = density
        
        ###########################################################################################################
        # Color decoding
        ###########################################################################################################
        if 'rgb' in  compute_channels:
            # Optionally concat the positions to the embedding, and also concatenate embedded view directions.
            fdir = torch.cat([density_feats,
                self.view_embedder(-ray_d)[:,None].repeat(1, num_samples, 1).view(-1, self.view_embed_dim)], dim=-1)
            timer.check("rf_density_view_cat")

            # Colors are values [0, 1] floats
            colors = torch.sigmoid(self.decoder_color(fdir)).reshape(batch, num_samples, 3)
            timer.check("rf_color_decode")
            out_dict['rgb'] = colors

        ###########################################################################################################
        # Semantics decoding
        ###########################################################################################################

        # Grids additive fusion
        if any([c in compute_channels for c in ['delta_density', 'panoptic_density', 'semantics', 'inst_embedding']]): 
            
            feats_detached = feats.detach()
            coords_detached = coords.detach()
            
            # Querry semantic delta grid
            delta_feats = self.delta_grid.interpolate(coords_detached, lod_idx).squeeze()
            delta_feats = delta_feats * self.lod_weights.to(delta_feats.device)
            if self.multiscale_type == 'sum':
                delta_feats = delta_feats.reshape(-1, lod_idx + 1, delta_feats.shape[-1] // (lod_idx + 1)).sum(-2)
            timer.check("rf_delta_grid_interpolate")
            panop_feats = feats_detached + delta_feats if not self.separate_sem_grid else delta_feats

        # Semantic density decoding
        if any([c in compute_channels for c in ['delta_density', 'panoptic_density']]):

            delta_density = self.decoder_delta_density(panop_feats).reshape(batch, num_samples, 1)
            if 'delta_density' in compute_channels:
                out_dict['delta_density'] = delta_density
                timer.check("rf_delta_density_decode")            
        
        if 'panoptic_density' in compute_channels:
            
            density_detached = density_feats[...,0:1].reshape(batch, num_samples, 1).detach()
            panop_density = density_detached + delta_density if not self.separate_sem_grid else delta_density
            out_dict['panoptic_density'] = torch.relu(panop_density)

        if 'semantics' in compute_channels:
            # Semantic class decoding
            semantics = self.decoder_semantics(panop_feats)
            semantics = torch.sigmoid(semantics) if self.sem_sigmoid else semantics
            semantics = F.normalize(semantics,dim=-1) if self.sem_normalize else semantics
            semantics = F.softmax(semantics, dim=-1) if self.sem_softmax else semantics
            timer.check("rf_semantics_decode")
            out_dict['semantics'] = semantics

        if 'inst_embedding' in compute_channels:

            # Semantic instance embeddings decoding
            inst_embedding = self.decoder_inst(panop_feats)
            inst_embedding = torch.sigmoid(inst_embedding) if self.inst_sigmoid else inst_embedding
            inst_embedding = F.normalize(inst_embedding, dim=-1) if self.inst_normalize else inst_embedding

            inst_embedding = inst_embedding / self.inst_soft_temperature if self.inst_soft_temperature > 0.0 else inst_embedding
            inst_embedding = F.softmax(inst_embedding, dim=-1) if self.inst_softmax else inst_embedding
            timer.check("rf_instance_embedding_decode")
            out_dict['inst_embedding'] = inst_embedding

        return out_dict
