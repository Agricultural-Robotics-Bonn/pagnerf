import torch
from torch import nn

from ..embedding import mean_class_embedding

class ClusteringBase(nn.Module):
    """Contrastive NeF with clustering interfaces on top of the semi-sup
    embedding optput
    """
    def __init__(self,
        
        num_clusters            : int = -1,
        distance_func           : str = 'cosine', #['cosine', euclidean]
        num_clustering_workers  : int = 1,
        
        **kwargs):

        super().__init__()
        
        self.distance_func = distance_func

        self.num_clusters = num_clusters


        self.num_workers = num_clustering_workers

    def train_clustering(self, X=None, labels=None):
        """Override and implement clustering specific training method
        """
        raise NotImplementedError("'train_clustering' is not implemented for this NeF.")

    def predict_clusters(self, X=None):
        """Override and implement clustering specific training method
        """
        raise NotImplementedError("'predict_clusters' is not implemented for this NeF.")