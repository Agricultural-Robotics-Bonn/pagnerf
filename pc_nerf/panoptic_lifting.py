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
from wisp.models.grids import *
from grids.tensorf import TensoRF
from wisp.ops.geometric import sample_unif_sphere

class PanopticLiftingNeF(BaseNeuralField):
    """ An exemplary contrastive for quick creation of new user neural fields.
        Clone this file and modify to create your own customized neural field.
    """
    def __init__(self,
        
        # Semantic args
        num_classes             : int   = -1,
        num_instances           : int   = -1,

        sem_activation_type     : str   = None,
        sem_num_layers          : int   = None,
        sem_hidden_dim          : int   = None,
        sem_normalize           : bool  = False,
        sem_softmax             : bool  = False,
        sem_sigmoid             : bool  = False,
        sem_detach              : bool  = True,
        
        inst_num_layers         : int   = None,
        inst_hidden_dim         : int   = None,
        inst_normalize          : bool  = False,
        inst_softmax            : bool  = False,
        inst_sigmoid            : bool  = False,
        inst_detach             : bool  = True,

        panoptic_features_type  : str   = None,

        **kwargs):

        
        self.num_classes = num_classes
        self.num_instances = num_instances

        self.sem_activation_type = sem_activation_type
        self.sem_num_layers = sem_num_layers
        self.sem_hidden_dim = sem_hidden_dim
        self.sem_normalize = sem_normalize
        self.sem_softmax = sem_softmax
        self.sem_sigmoid = sem_sigmoid
        self.sem_detach = sem_detach
        
        self.inst_num_layers = inst_num_layers
        self.inst_hidden_dim = inst_hidden_dim
        self.inst_detach = inst_detach
        self.inst_softmax = inst_softmax
        self.inst_normalize = inst_normalize
        self.inst_sigmoid = inst_sigmoid

        super().__init__(**kwargs)

    def init_decoder(self):
        """Create here any decoder networks to be used by the neural field.
        Decoders should map from features to output values (such as: rgb, density, sdf, etc), for example:
        """
        self.input_dim_inst = 3
        self.input_dim_sem  = 3


        self.decoder_color = MLPRenderFeature(in_channels=27)

        self.sem_activation_type = self.sem_activation_type if self.sem_activation_type else self.activation_type
        self.sem_num_layers = self.sem_num_layers if self.sem_num_layers else self.num_layers 
        self.sem_hidden_dim = self.sem_hidden_dim if self.sem_hidden_dim else self.hidden_dim
        # Semantic Decoder
        # assert self.num_classes >= 2, log.error(f"'num_classes' needs to be >= 2, but {self.num_classes} was given.")
        if self.sem_num_layers == 0:
            self.sem_hidden_dim = self.input_dim

        self.decoder_semantics = BasicDecoder(input_dim=self.input_dim_sem,
                                    output_dim=self.num_classes,
                                    activation=get_activation_class(self.sem_activation_type),
                                    bias=True,
                                    layer=get_layer_class(self.layer_type),
                                    num_layers=self.sem_num_layers - 1,
                                    hidden_dim=self.sem_hidden_dim,
                                    skip=[])
        
        assert self.num_instances > 2, log.error(f"'num_instances' needs to be >= 2, but {self.num_classes} was given.")
        self.inst_num_layers = self.inst_num_layers if self.inst_num_layers else self.num_layers 
        self.inst_hidden_dim = self.inst_hidden_dim if self.inst_hidden_dim else self.hidden_dim
        if self.inst_num_layers == 0:
            self.inst_hidden_dim = self.input_dim
        
        self.decoder_inst = BasicDecoder(input_dim=self.input_dim_inst,
                                    output_dim=self.num_instances,
                                    activation=get_activation_class(self.sem_activation_type),
                                    bias=True,
                                    layer=get_layer_class(self.layer_type),
                                    num_layers=self.inst_num_layers - 1,
                                    hidden_dim=self.inst_hidden_dim,
                                    skip=[])
    def _get_grid_class(self):
        return TensoRF
    
    def init_grid(self):
        """ Creates the feature structure this neural field uses, i.e: Octree, Triplane, Hashed grid and so forth.
        The feature grid is queried with coordinate samples during ray tracing / marching.
        The feature grid may also include an occupancy acceleration structure internally to speed up
        tracers.
        Always set interpolation_type to 'cat' to apply lod weights after feature interpolation.
        """

        self.grid = TensoRF()

    def get_nef_type(self):
        """Returns a text keyword describing the neural field type.

        Returns:
            (str): The key type
        """
        return 'panoptic_nef'

    def prune(self):
        """Prunes the blas based on current state.
        """

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
        self._register_forward_function(self.rgb_semantics, ["density", "rgb", "semantics", "inst_embedding"])

    def rgb_semantics(self, coords, ray_d, compute_channels, pidx=None, lod_idx=None):
        """Compute color, density and semantics for the provided coordinates.
                   +----+              
         Dir[2] ---|Penc|----------+
                   +----+          |
                                   |         
                   +-------+    +-----+      
         Pos[3] ---|TensoRF|----| MLP |--- RGB[3]
                   +---+---+    +-----+       
                       |                  
                       +------------------ density[1]   
        
                       +-----+               
         Pos[3] ---+---| MLP |------------ semantics[num_classes]
                   |   +-----+               
                   |                         
                   |   +-----+               
                   +---| MLP |------------ inst_embedding[dim_embedding]
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
        # Position grid encoding
        ###########################################################################################################
        # Embed coordinates into high-dimensional vectors with the grid.
        if any([c in compute_channels for c in ['density', 'rgb']]): 
            density_feats, color_feats = self.grid.interpolate(coords,0)
            timer.check("rf_rgba_interpolate")
            
            # Density is [particles / meter], so need to be multiplied by distance
            density = torch.relu(density_feats[...,0:1]).reshape(batch, num_samples, 1)
            timer.check("rf_density_normalization")
            if 'density' in compute_channels:
                out_dict['density'] = density
        
        ###########################################################################################################
        # Color decoding
        ###########################################################################################################
        if 'rgb' in  compute_channels:
            colors = self.decoder_color(-ray_d, color_feats)
            timer.check("rf_color_decode")
            out_dict['rgb'] = colors

        ###########################################################################################################
        # Semantics decoding
        ###########################################################################################################
      
        if 'semantics' in compute_channels:
            semantics = self.decoder_semantics(coords)
            
            semantics = torch.sigmoid(semantics) if self.sem_sigmoid else semantics
            semantics = F.normalize(semantics,dim=-1) if self.sem_normalize else semantics
            semantics = F.softmax(semantics, dim=-1) if self.sem_softmax else semantics
            out_dict['semantics'] = semantics

        ###########################################################################################################
        # Semi-sup decoding
        ###########################################################################################################
        if 'inst_embedding' in compute_channels:

            inst_embedding = self.decoder_inst(coords)

            inst_embedding = torch.sigmoid(inst_embedding) if self.inst_sigmoid else inst_embedding
            inst_embedding = F.normalize(inst_embedding, dim=-1) if self.inst_normalize else inst_embedding
            inst_embedding = F.softmax(inst_embedding, dim=-1) if self.inst_softmax else inst_embedding
            out_dict['inst_embedding'] = inst_embedding

        return out_dict


class MLPRenderFeature(torch.nn.Module):

    def __init__(self, in_channels, out_channels=3, pe_view=2, pe_feat=2, dim_mlp_color=128, output_activation=torch.sigmoid):
        super().__init__()
        self.pe_view = pe_view
        self.pe_feat = pe_feat
        self.output_channels = out_channels
        self.view_independent = self.pe_view == 0 and self.pe_feat == 0
        self.in_feat_mlp = 2 * pe_view * 3 + 2 * pe_feat * in_channels + in_channels + (3 if not self.view_independent else 0)
        self.output_activation = output_activation
        layer1 = torch.nn.Linear(self.in_feat_mlp, dim_mlp_color)
        layer2 = torch.nn.Linear(dim_mlp_color, dim_mlp_color)
        layer3 = torch.nn.Linear(dim_mlp_color, out_channels)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, viewdirs, features):
        indata = [features.view(-1, features.shape[-1])]
        if not self.view_independent:
            indata.append(viewdirs)
        if self.pe_feat > 0:
            indata += [MLPRenderFeature.positional_encoding(features.view(-1, features.shape[-1]), self.pe_feat)]
        if self.pe_view > 0:
            indata += [MLPRenderFeature.positional_encoding(viewdirs, self.pe_view)]
        mlp_in = torch.cat(indata, dim=-1)
        out = self.mlp(mlp_in)
        out = self.output_activation(out)
        return out

    @staticmethod
    def positional_encoding(positions, freqs):
        freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)
        pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] + (freqs * positions.shape[-1],))
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts