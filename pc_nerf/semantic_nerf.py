# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from wisp.models.nefs import BaseNeuralField
from wisp.models.activations import get_activation_class
from wisp.models.layers import get_layer_class
from wisp.models.embedders import get_positional_embedder
from wisp.ops.geometric import sample_unif_sphere
from wisp.models.grids import *

from grids.occtree import Occtree

class SemanticNeF(BaseNeuralField):
    """ An exemplary contrastive for quick creation of new user neural fields.
        Clone this file and modify to create your own customized neural field.
    """
    def __init__(self,
        
        # Semantic args
        num_classes             : int   = -1,

        **kwargs):
        
        self.num_classes = num_classes

        super().__init__(**kwargs)
    
    def init_embedder(self):
        """ Panoptic NeF uses viewing direction embedding only
        """
        self.view_embedder, self.view_embed_dim = get_positional_embedder(10, True)
        self.pos_embedder, self.pos_embed_dim = get_positional_embedder(10, True)

        log.info(f"View Embed Dim: {self.view_embed_dim}")
        log.info(f"Pos Embed Dim: {self.pos_embed_dim}")

    def init_decoder(self):
        """Create here any decoder networks to be used by the neural field.
        Decoders should map from features to output values (such as: rgb, density, sdf, etc), for example:
        """

        self.decoder_features = BasicDecoder(input_dim=self.pos_embed_dim,
                                        output_dim=self.hidden_dim,
                                        activation=get_activation_class(self.activation_type),
                                        bias=True,
                                        layer=get_layer_class(self.layer_type),
                                        num_layers=8,
                                        hidden_dim=self.hidden_dim,
                                        skip=[5])

        self.decoder_density = get_layer_class(self.layer_type)(self.hidden_dim, 1, bias=True)
        self.decoder_density.bias.data[0] = 1.0

        self.decoder_color = BasicDecoder(input_dim=self.hidden_dim + self.view_embed_dim,
                                        output_dim=3,
                                        activation=get_activation_class(self.activation_type),
                                        bias=True,
                                        layer=get_layer_class(self.layer_type),
                                        num_layers=1,
                                        hidden_dim=self.hidden_dim // 2,
                                        skip=[])

        self.decoder_semantics = BasicDecoder(input_dim=self.hidden_dim,
                                    output_dim=self.num_classes,
                                    activation=get_activation_class(self.activation_type),
                                    bias=True,
                                    layer=get_layer_class(self.layer_type),
                                    num_layers=1,
                                    hidden_dim=self.hidden_dim // 2,
                                    skip=[])
        
        
    def _get_grid_class(self):
        grid_class = Occtree
        return grid_class
    
    def init_grid(self):
        """ Creates the feature structure this neural field uses, i.e: Octree, Triplane, Hashed grid and so forth.
        The feature grid is queried with coordinate samples during ray tracing / marching.
        The feature grid may also include an occupancy acceleration structure internally to speed up
        tracers.
        Always set interpolation_type to 'cat' to apply lod weights after feature interpolation.
        """
        self.grid = Occtree()

    def get_nef_type(self):
        """Returns a text keyword describing the neural field type.

        Returns:
            (str): The key type
        """
        return 'panoptic_nef'

    def prune(self):
        """Prunes the blas based on current state.
        """
        # TODO(ttakikawa): Expose these parameters. 
        # This is still an experimental feature for the most part. It does work however.
        density_decay = 0.6
        min_density = ((0.01 * 512)/np.sqrt(3))
        self.grid.occupancy = self.grid.occupancy.cuda()
        self.grid.occupancy = self.grid.occupancy * density_decay
        points = self.grid.dense_points.cuda()
        #idx = torch.randperm(points.shape[0]) # [:N] to subsample
        res = 2.0**self.grid.blas_level
        samples = torch.rand(points.shape[0], 3, device=points.device)
        samples = points.float() + samples
        samples = samples / res
        samples = samples * 2.0 - 1.0
        sample_views = torch.FloatTensor(sample_unif_sphere(samples.shape[0])).to(points.device)
        with torch.no_grad():
            density = self.forward(coords=samples[:,None], ray_d=sample_views, channels="density")
        self.grid.occupancy = torch.stack([density[:, 0, 0], self.grid.occupancy], -1).max(dim=-1)[0]

        mask = self.grid.occupancy > min_density

        _points = points[mask]
        octree = spc_ops.unbatched_points_to_octree(_points, self.grid.blas_level, sorted=True)
        self.grid.blas.init(octree)

    def forward(self, channels=None, **kwargs):
        # use the channels argument to specify which channels need to be computed
        kwargs['compute_channels'] = channels
        return super().forward(channels, **kwargs)


    def register_forward_functions(self):
        """Register the forward functions.
        Forward functions define the named output channels this neural field supports.
        By registering forward functions, a tracer knows which neural field methods to use to obtain channels values.
        """
        # Here the rgba() function handles both the rgb and density channels at the same time
        self._register_forward_function(self.rgb_semantics, ["density", "rgb", "semantics"])

    def rgb_semantics(self, coords, ray_d, compute_channels, pidx=None, lod_idx=None):
        """Compute color, density and semantics for the provided coordinates.
        
         Dir[2] -------------------------------+         
                                               |         
                   +------+       +-----+     +-----+      
         Pos[3] ---|PosEnc|---+---| MLP |--+--| MLP |--- RGB[3]
                   +------+       +-----+  |  +-----+       
                                           |     
                                           |  +-----+        
                                           +--| MLP |--- density[1]
                                           |  +-----+        
                                           |
                                           |  +-----+   
                                           +--| MLP |--- semantics[num_classes]
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
        # Position grig encoding
        ###########################################################################################################
        # Embed coordinates into high-dimensional vectors.
        feats = self.decoder_features( self.pos_embedder(coords.view(-1,3)))
        timer.check("comute features")
        ###########################################################################################################
        # Density decoding
        ###########################################################################################################
        if any([c in compute_channels for c in ['density', 'rgb']]): 
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
            fdir = torch.cat([feats,
                self.view_embedder(-ray_d)[:,None].repeat(1, num_samples, 1).view(-1, self.view_embed_dim)], dim=-1)
            timer.check("rf_density_view_cat")

            # Colors are values [0, 1] floats
            colors = torch.sigmoid(self.decoder_color(fdir)).reshape(batch, num_samples, 3)
            timer.check("rf_color_decode")
            out_dict['rgb'] = colors

        ###########################################################################################################
        # Semantics decoding
        ###########################################################################################################

        if 'semantics' in compute_channels:
            # Compute semantic one-hot logits
            out_dict['semantics'] = self.decoder_semantics(feats)

        return out_dict
