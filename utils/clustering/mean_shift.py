import torch
from sklearn.cluster import MeanShift as mean_shift_sklearn
from sklearn.cluster import estimate_bandwidth

from utils.embedding import mean_class_embedding
from .clustering_base import ClusteringBase


class MeanShift(ClusteringBase):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.ms = None

  def train_clustering(self, X, labels):
    centers = mean_class_embedding(X, labels)

    if centers.nelement() == 0:
      return

    centers = centers.detach().cpu().numpy()
    bandwidth = estimate_bandwidth(centers, quantile=0.08)

    self.ms = mean_shift_sklearn(bandwidth=bandwidth, bin_seeding=False, n_jobs=self.num_workers).fit(centers)
    
  def predict_clusters(self, X=None):
  
    if not isinstance(self.ms, mean_shift_sklearn):
      return torch.argmax(torch.nn.functional.normalize(X, dim=-1), dim=-1)

    device = X.device
    original_shape = X.shape[:-1]
    X = X.detach().flatten(end_dim=-2).cpu().numpy()
    preds = self.ms.predict(X)
    return torch.Tensor(preds).to(device).type(torch.int64).reshape(original_shape)
    