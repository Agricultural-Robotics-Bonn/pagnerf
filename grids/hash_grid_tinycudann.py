# Tiny Cuda NN hash grid wraper
import torch
import logging as log

from wisp.models.grids import HashGrid
import tinycudann as tcnn

class HashGridTinyCudaNN(HashGrid):
    """This is a feature grid where the features are defined in a codebook that is hashed.
    """
    def __init__(self,*args, **kwargs):   
        super().__init__(*args, **kwargs)

    def init_from_resolutions(self, resolutions):
        """Build a multiscale hash grid from a list of resolutions.
        """
        self.resolutions = resolutions
        self.num_lods = len(resolutions)
        self.active_lods = [x for x in range(self.num_lods)]
        self.max_lod = self.num_lods - 1

        log.info(f"Active Resolutions: {self.resolutions}")
        
        self.embedder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.num_lods,
                "n_features_per_level": self.feature_dim,
                "log2_hashmap_size": self.codebook_bitwidth,
                "base_resolution": resolutions[0],
                "per_level_scale": 2, #1.3819,
            },
        )

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def interpolate(self, coords, lod_idx, pidx=None):

        batch, num_samples, _ = coords.shape
        
        feats = self.embedder(coords.reshape(-1,3)).type(torch.float)

        if self.multiscale_type == 'cat':
            return feats
        elif self.multiscale_type == 'sum':
            return feats.reshape(batch, num_samples, len(self.resolutions), feats.shape[-1] // len(self.resolutions)).sum(-2)
        else:
            raise NotImplementedError