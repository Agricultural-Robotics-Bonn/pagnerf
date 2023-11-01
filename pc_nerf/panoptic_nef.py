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
from grids.hash_grid_torch import HashGridTorch
from grids.hash_grid_tinycudann import HashGridTinyCudaNN
from grids.permuto_grid import PermutoGrid
from wisp.ops.geometric import sample_unif_sphere

class PanopticNeF(BaseNeuralField):
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

        self.panoptic_features_type = panoptic_features_type

        super().__init__(**kwargs)
    
    def init_embedder(self):
        """ Panoptic NeF uses viewing direction embedding only
        """
        self.view_embedder, self.view_embed_dim = get_positional_embedder(self.view_multires, 
                                                                        self.embedder_type == "positional")
        log.info(f"View Embed Dim: {self.view_embed_dim}")

    def _compute_input_dimension(self):
        ''' Compute decoders input dimension depending on grid parameters 
        Should be be overriden if input dimensions to specific decoders need to be different
        '''
        if self.position_input:
            raise NotImplementedError
        
        # Determine: What is the effective feature dimensions?
        # (are we using concatenation or summation to consolidate features from multiple LODs?)
        if self.multiscale_type == 'cat':
            self.effective_feature_dim = self.grid.feature_dim * self.num_lods
        elif self.multiscale_type == 'sum':
            self.effective_feature_dim = self.grid.feature_dim
        else:
            raise NotImplementedError(f"'{self.multiscale_type}' not supported by this neural field. "
                                      "supported options ['cat', 'sum']")

        # The input to the decoder is the effective feature dimension + positional embedding
        self.input_dim_density = self.effective_feature_dim
        if self.panoptic_features_type == 'position':
            self.input_dim_inst = 3
            self.input_dim_sem  = 3
        elif self.panoptic_features_type == 'pos_encoding':
            self.input_dim_inst = self.pos_embed_dim
            self.input_dim_sem  = self.pos_embed_dim
        else:
            self.input_dim_inst = self.effective_feature_dim
            self.input_dim_sem  = self.effective_feature_dim

    def init_decoder(self):
        """Create here any decoder networks to be used by the neural field.
        Decoders should map from features to output values (such as: rgb, density, sdf, etc), for example:
        """
        self._compute_input_dimension()

        self.decoder_density = BasicDecoder(input_dim=self.input_dim_density,
                                        output_dim=16,
                                        activation=get_activation_class(self.activation_type),
                                        bias=True,
                                        layer=get_layer_class(self.layer_type),
                                        num_layers=self.num_layers,
                                        hidden_dim=self.hidden_dim,
                                        skip=[])

        self.decoder_density.lout.bias.data[0] = 1.0

        self.decoder_color = BasicDecoder(input_dim=16 + self.view_embed_dim,
                                        output_dim=3,
                                        activation=get_activation_class(self.activation_type),
                                        bias=True,
                                        layer=get_layer_class(self.layer_type),
                                        num_layers=self.num_layers+1,
                                        hidden_dim=self.hidden_dim,
                                        skip=[])

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
                                    num_layers=self.sem_num_layers,
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
                                    num_layers=self.inst_num_layers,
                                    hidden_dim=self.inst_hidden_dim,
                                    skip=[])
    def _get_grid_class(self):
        if self.grid_type == "OctreeGrid":
            grid_class = OctreeGrid
        elif self.grid_type == "CodebookOctreeGrid":
            grid_class = CodebookOctreeGrid
        elif self.grid_type == "TriplanarGrid":
            grid_class = TriplanarGrid
        elif self.grid_type == "HashGrid":
            grid_class = HashGrid
        elif self.grid_type == "HashGridTorch":
            grid_class = HashGridTorch
        elif self.grid_type == "HashGridTinyCudaNN":
            grid_class = HashGridTinyCudaNN
        elif self.grid_type == "PermutoGrid":
            grid_class = PermutoGrid
        else:
            raise NotImplementedError(f"'{self.grid_type}' not supproted")
        return grid_class
    
    def init_grid(self):
        """ Creates the feature structure this neural field uses, i.e: Octree, Triplane, Hashed grid and so forth.
        The feature grid is queried with coordinate samples during ray tracing / marching.
        The feature grid may also include an occupancy acceleration structure internally to speed up
        tracers.
        Always set interpolation_type to 'cat' to apply lod weights after feature interpolation.
        """
        self.grid = self._get_grid_class()( self.feature_dim,
                                            base_lod=self.base_lod, num_lods=self.num_lods,
                                            interpolation_type=self.interpolation_type, multiscale_type='cat',
                                            **self.kwargs)
        
        self.lod_weights = torch.ones(self.num_lods * self.grid.feature_dim)


    def get_nef_type(self):
        """Returns a text keyword describing the neural field type.

        Returns:
            (str): The key type
        """
        return 'panoptic_nef'

    def prune(self):
        """Prunes the blas based on current state.
        """
        if self.grid is not None:
            
            if self.grid_type in ["HashGrid", "HashGridTorch", "HashGridTinyCudaNN", "TriplanarGrid"]:
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
            else:
                raise NotImplementedError

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
        
         Dir[2] -------------------------------+         
                                               |         
                   +----+       +-----+     +-----+      
         Pos[3] ---|grid|---+---| MLP |--+--| MLP |--- RGB[3]
                   +----+   |   +-----+  |  +-----+       
                            |            +------------ density[1]              
                            |   +-----+               
                            +---| MLP |--------------- semantics[num_classes]
                            |   +-----+               
                            |                         
                            |   +-----+               
                            +---| MLP |--------------- inst_embedding[dim_embedding]
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
        # Embed coordinates into high-dimensional vectors with the grid.
        feats = self.grid.interpolate(coords, lod_idx).squeeze()
        feats = feats * self.lod_weights.to(feats.device)
        if self.multiscale_type == 'sum':
            feats = feats.reshape(-1, lod_idx + 1, feats.shape[-1] // (lod_idx + 1)).sum(-2)
        timer.check("rf_rgba_interpolate")
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

        if 'semantics' in compute_channels:
            # Compute semantic one-hot logits
            sem_input = feats.detach() if self.sem_detach else feats
            semantics = self.decoder_semantics(sem_input)
            
            semantics = torch.sigmoid(semantics) if self.sem_sigmoid else semantics
            semantics = F.normalize(semantics,dim=-1) if self.sem_normalize else semantics
            semantics = F.softmax(semantics, dim=-1) if self.sem_softmax else semantics
            out_dict['semantics'] = semantics

        ###########################################################################################################
        # Semi-sup decoding
        ###########################################################################################################
        if 'inst_embedding' in compute_channels:
            if self.inst_direct_pos:
                inst_input = coords
            else:
                inst_input = feats.detach() if self.inst_detach else feats
            

            inst_embedding = self.decoder_inst(inst_input)

            inst_embedding = torch.sigmoid(inst_embedding) if self.inst_sigmoid else inst_embedding
            inst_embedding = F.normalize(inst_embedding, dim=-1) if self.inst_normalize else inst_embedding
            inst_embedding = F.softmax(self.decoder_inst(inst_input), dim=-1) if self.inst_softmax else inst_embedding
            out_dict['inst_embedding'] = inst_embedding

        return out_dict
