"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

from functools import reduce

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, pn_ratio=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.pos_weight =  min(1, pn_ratio * 2)
        self.neg_weight =  min(1, (1 - pn_ratio) * 2)

    def img_wise_forward(self, features, labels=None, mask=None, device='cuda'):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = self.pos_weight * logits - self.neg_weight * torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-16)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(batch_size, anchor_count)

    def forward(self, features, labels=None, mask=None, reduction='mean', anchor_mask=None, *args,**kwargs):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_rays, ...].
            labels: ground truth of shape [bsz, n_rays].
            mask: contrastive mask of shape [num_rays, num_rays], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if anchor_mask is not None:
            assert anchor_mask.shape == labels.shape,\
                f'anchor_mask and labes must be the same size, but got {anchor_mask.shape} and {labels.shape}'
            if anchor_mask.sum() == 0.0:
                return 0.0
            loss = []
        else:
            loss = torch.empty(0,labels.shape[-1]).to(labels.device)

        features = nn.functional.normalize(features, dim=-1)
        for i, (x, l) in enumerate(zip(features, labels)):
            x = x[:,None]
            if anchor_mask is not None:
                # Avoid computing loss if no valid labels are pressent 
                if anchor_mask[i].sum() == 0.0:
                    loss.append(torch.tensor(0.0).to(labels.device))
                    continue
                # Mask samples and compute loss 
                x, l = (x[anchor_mask[i]], l[anchor_mask[i]])
                if l.unique().numel() < 2:
                    loss.append(torch.tensor(0.0).to(labels.device))
                    continue
                loss.append(self.img_wise_forward(x, l, mask=None, device=device).view(1,-1))
            else:
                # campute loss for the complete image
                loss = torch.cat((loss, self.img_wise_forward(x, l, mask=None, device=device).view(1,-1)), dim=0)
        
        if reduction == 'sum':
            if isinstance(loss,list):
                return reduce(lambda x,y: x+y.sum(), loss, torch.tensor([0.0]).to(labels.device))
            return torch.sum(loss)
        elif reduction == 'mean':
            if isinstance(loss,list):
                if anchor_mask is not None:
                    norm = anchor_mask.sum()
                else:
                    norm = labels.numel()
                return reduce(lambda x,y: x+y.sum()/norm, loss, torch.tensor([0.0]).to(labels.device))
            return loss.mean()
        elif reduction in ['none', None]:
            return loss
        else:
            raise NotImplementedError(f"Unsupported reduction method {reduction}. Posible options are ['sum', 'mean', 'none']")
            