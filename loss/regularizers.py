import torch
import torch.nn.functional as F
from math import exp, log, floor

def segment_consistency_regularizer(embeddings, labels):
  ''' Computes class-wise mean embedding centers for each sanple in the batch.
  '''
  reg_loss = torch.tensor(0.0).to(embeddings.device)
  for x,l in zip(embeddings, labels):
    # get unique segments and split them in separate arrays
    _, sample_counts = l.unique(return_counts=True)
    x = x[torch.argsort(l)]

    sample_idxs = torch.zeros_like(sample_counts)
    sample_idxs[1:] = torch.cumsum(sample_counts, dim=0)[:-1]
    # get segments excluding the stuff/bg segment
    # tuple [n_segments]:[n_samples, feat_dim]
    x_samples = x.tensor_split(sample_idxs[1:].cpu())

    for segment in x_samples:
      # Count predicted IDs including stuff/bg (ID=0)
      segment_bins = segment.argmax(dim=-1).bincount()
      # Compute most likely label for the whole segment avoiding stuff/bg (ID=0)
      if segment_bins[1:].numel() == 0:
        continue
      # segment_best_label = segment_bins[1:].argmax() + 1
      segment_best_label = segment_bins[1:].argmax() + 1
      # if the bast majority of pixels are stuff/bg make it  
      if segment_bins[0] * 0.5 > segment_bins[segment_best_label]:
        segment_best_label = torch.tensor([0], device=segment.device)
      # Compute cross entropy loss assuming samples are softmaxed already
      reg_loss += F.nll_loss(torch.log(segment), segment_best_label.expand(segment.shape[0]))
    reg_loss /= len(x_samples)
  
  return reg_loss / embeddings.shape[0]

def sigma_sparsity_loss(sigmas):
  # Using Cauchy Sparsity loss on sigma values
  return torch.log(1.0 + 2*torch.pow(sigmas,2))

def tv_loss(values, fn):
  loss = 0.0
  size = values.shape[0] 
  dims = len(values.shape) - 1
  for d in range(dims):
    vals_d = torch.swapdims(values,d,0)
    loss += fn(vals_d[1:]-vals_d[:-1]) / size
  return loss

def tv_l1_loss(values):
  return tv_loss(values, lambda x: torch.abs(x).sum())

def tv_l2_loss(values):
  return tv_loss(values, lambda x: torch.pow(x,2).sum())

def grid_tv_loss(encoder, fn, sample_size = 0.2, num_dim_samples = 50, device='cuda'):
  # cordinate frame centered in the representation (i.e. axis spaning [-1,1])      
  min_vertex = torch.randn(3) * 2 * (1-sample_size) - 1
  edge_coords = min_vertex + torch.stack([torch.arange(num_dim_samples+1) for _ in range(3)], dim=-1)
  coords = torch.stack(torch.meshgrid(edge_coords.unbind(dim=-1)), dim=-1).to(device)
  orig_shape = coords.shape

  values = encoder(coords.reshape(-1,1,3)).reshape(*orig_shape[:-1],-1)
  return fn(values.reshape(*coords.shape[:-1], -1))

def grid_tv_l1_loss(encoder, *args, **kwargs):
  return grid_tv_loss(encoder, tv_l1_loss, *args, **kwargs)

def grid_tv_l2_loss(encoder, *args, **kwargs):
  return grid_tv_loss(encoder, tv_l2_loss, *args, **kwargs)