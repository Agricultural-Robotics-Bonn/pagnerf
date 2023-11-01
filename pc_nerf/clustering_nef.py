import torch

from wisp.models.nefs.base_nef import BaseNeuralField
from utils.clustering.clustering_base import ClusteringBase


class ClusteringNeF(BaseNeuralField):
    """Contrastive NeF with clustering interfaces on top of the semi-sup
    embedding optput
    """
    def __init__(self,

        cluster_class           : ClusteringBase = None,
        embedding_channel       : str = 'embedding',
        **kwargs):
        
        self.clustering_obj = cluster_class(**kwargs)

        assert embedding_channel in self.get_supported_channels(),\
            f'"{embedding_channel}" Channel not supported for custering, '\
            f'supported channels by NeF are: {self.get_supported_channels()}'
        self.embedding_channel = embedding_channel

    def get_nef_type(self):      
        return f'clustering_{super().get_nef_type()}'

    def register_forward_functions(self):
        ''' Wrap NeF's forward function to add clustering on top 
        '''
        super().register_forward_functions()
        self.nef_forward, supported_channels = list(self._forward_functions.items())[0]
        self._forward_functions = {}

        supported_channels.add('clusters')
        self._register_forward_function(self.cluster_nef, supported_channels)

    def train_clustering(self, X=None, labels=None):
        self.clustering_obj.train_clustering(X, labels)

    def predict_clusters(self, X=None):
        return self.clustering_obj.predict_clusters(X)
        
    def cluster_nef(self, coords, ray_d, compute_channels, pidx=None, lod_idx=None, **kwargs):
        '''Wrap forward pass and add clusters modality to NeF
        '''
        if isinstance(compute_channels, str):
            compute_channels = [compute_channels]
        if 'clusters' in compute_channels:
            if isinstance(compute_channels, set):
                compute_channels.add(self.embedding_channel)
            else:
                compute_channels.append(self.embedding_channel)
        # Run NeF foward pass 
        outputs = self.nef_forward(coords, ray_d, compute_channels, pidx, lod_idx, **kwargs)
        
        if 'clusters' in compute_channels:
            outputs['clusters'] = outputs[self.embedding_channel]

        return outputs


# Panotic Contrastive NeF wrappers
############################################################################

from .panoptic_nef import PanopticNeF
from .panoptic_dd_nef import PanopticDDensityNeF
from .panoptic_delta_nef import PanopticDeltaNeF

# Mean Shift clustering NeFs
from utils.clustering.mean_shift import MeanShift

class MeanShiftPanopticNeF(ClusteringNeF, PanopticNeF):
    def __init__(self, *args, **kwargs):
        PanopticNeF.__init__(self, *args, **kwargs)
        ClusteringNeF.__init__(self, *args,
                               cluster_class = MeanShift,
                               embedding_channel = 'inst_embedding',
                               **kwargs)
    def get_nef_type(self):      
        return 'mean_shift_panoptic_nef'

class MeanShiftPanopticDDensityNeF(ClusteringNeF, PanopticDDensityNeF):
    def __init__(self, *args, **kwargs):
        PanopticDDensityNeF.__init__(self,*args, **kwargs)
        ClusteringNeF.__init__(self, *args,
                               cluster_class = MeanShift,
                               embedding_channel = 'inst_embedding',
                               **kwargs)

class MeanShiftPanopticDeltaNeF(ClusteringNeF, PanopticDeltaNeF):
    def __init__(self, *args, **kwargs):
        PanopticDeltaNeF.__init__(self,*args, **kwargs)
        ClusteringNeF.__init__(self, *args,
                               cluster_class = MeanShift,
                               embedding_channel = 'inst_embedding',
                               **kwargs)

    def get_nef_type(self):      
        return 'mean_shift_panoptic_delta_nef'