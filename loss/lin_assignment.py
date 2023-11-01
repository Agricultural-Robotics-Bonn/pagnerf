# from panoptic lifting implementation
#
#https://github.com/nihalsid/panoptic-lifting/blob/7af7a3e8477ead8e57f699a240d993e3bc21ee42/trainer/train_panopli_tensorf.py#L195-L206

import numpy as np
import torch
from torch import nn
import scipy
import torch.nn.functional as F

class LinAssignmentLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @torch.no_grad()
    def create_virtual_gt_with_linear_assignment(self, labels_gt, predicted_scores):
            labels = sorted(torch.unique(labels_gt).cpu().tolist())[:predicted_scores.shape[-1]]
            predicted_probabilities = torch.softmax(predicted_scores, dim=-1)
            cost_matrix = np.zeros([len(labels), predicted_probabilities.shape[-1]])
            for lidx, label in enumerate(labels):
                cost_matrix[lidx, :] = -(predicted_probabilities[labels_gt == label, :].sum(dim=0) / ((labels_gt == label).sum() + 1e-4)).cpu().numpy()
            assignment = scipy.optimize.linear_sum_assignment(np.nan_to_num(cost_matrix))
            new_labels = torch.zeros_like(labels_gt)
            for aidx, lidx in enumerate(assignment[0]):
                new_labels[labels_gt == labels[lidx]] = assignment[1][aidx]
            return new_labels

    def forward(self, inst_embeddings, labels_gt, *args, **kwargs):
        loss = torch.Tensor([0.0]).to(labels_gt.device)
        for s,gt in zip(inst_embeddings, labels_gt):
            virt_labels = self.create_virtual_gt_with_linear_assignment(gt, s)
            predicted_labels = s.argmax(dim=-1)
            if torch.any(virt_labels != predicted_labels):  # should never reinforce correct labels
                loss += F.nll_loss(torch.log(s + 1e-27), virt_labels, reduction='mean')
                # loss += F.cross_entropy(torch.log(s + 1e-27), virt_labels, reduction='mean')
            
        return loss / inst_embeddings.shape[0]