import torch

def mean_class_embedding(embeddings, labels):
  ''' Computes class-wise mean embedding centers for each sanple in the batch.
  '''
  centers = torch.empty(0, embeddings.shape[-1]).to(embeddings.device)
  for x,l in zip(embeddings, labels):
  
    _, inverse, sample_counts = l.unique(return_counts=True, return_inverse=True)

    x = x[torch.argsort(l)]
    
    l = inverse.sort()[0]

    sample_idxs = torch.zeros_like(sample_counts)
    sample_idxs[1:] = torch.cumsum(sample_counts, dim=0)[:-1]

    # tuple [n_inst]:[n_samples, feat_dim]
    x_samples = x.tensor_split(sample_idxs[1:].cpu())
    # Pad smaples variable number of samples to store in tensor
    x_samples_pad = torch.nn.utils.rnn.pad_sequence(x_samples)
    # batch instances Feat centers: [n_inst, feat_dim]
    b_centers = (x_samples_pad.sum(dim=0).T / sample_counts).T
    centers = torch.cat((centers, b_centers), dim=0)
  
  return centers