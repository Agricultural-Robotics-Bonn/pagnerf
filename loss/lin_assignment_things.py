# from panoptic lifting implementation
#
#https://github.com/nihalsid/panoptic-lifting/blob/7af7a3e8477ead8e57f699a240d993e3bc21ee42/trainer/train_panopli_tensorf.py#L195-L206

import numpy as np
import torch
from torch import nn
import scipy
import torch.nn.functional as F

from utils.outlier_rejection import centers_from_3d_points_with_ids, add_position_id_range_cost

class LinAssignmentThingsLoss(nn.Module):
    def __init__(self, outlier_rejection=False, min_distance=0.2, max_distance=0.5, *args, **kwargs):
        super().__init__()
        self.outlier_rejection = outlier_rejection
        self.min_distance = min_distance
        self.max_distance = max_distance

        self.inst_centers_db = torch.zeros([0,4]).to('cuda')

    @torch.no_grad()
    def create_virtual_gt_with_linear_assignment(self, inst_probabilities, labels_gt, points_3d=None):
            # Leave first element for stuff
            things_mask = labels_gt > 0
            things_gt = labels_gt[things_mask]
            things_prob = inst_probabilities[things_mask][...,1:]
            # Compute surrogate labels
            labels = sorted(torch.unique(things_gt).cpu().tolist())[:things_prob.shape[-1]]
            cost_matrix = np.zeros([len(labels), things_prob.shape[-1]])
            for lidx, label in enumerate(labels):
                cost_matrix[lidx, :] = -(things_prob[things_gt == label, :].sum(dim=0) / ((things_gt == label).sum() + 1e-4)).cpu().numpy()
            
            # Update cost matrix to avoid early assignment of repeated IDs 
            assert self.outlier_rejection and points_3d is not None or not self.outlier_rejection, \
                'Outlier rejection requires 3d points'
            if self.outlier_rejection:
                # Compute centers of the current things Id gts
                points_3d_gt = torch.cat([points_3d[things_mask], things_gt[:,None]], dim=-1)
                current_inst_centers = centers_from_3d_points_with_ids(points_3d_gt)
                # Update cost matrix to have high cost when trying to assign repeated IDs
                cost_matrix = add_position_id_range_cost(cost_matrix, current_inst_centers)

            assignment = scipy.optimize.linear_sum_assignment(np.nan_to_num(cost_matrix))
            
            things_labels = torch.zeros_like(things_gt)
            for aidx, lidx in enumerate(assignment[0]):
                things_labels[things_gt == labels[lidx]] = assignment[1][aidx]
            
            new_labels = torch.zeros_like(labels_gt)
            # Add 1 since the full predicted scores has one more component than things score
            new_labels[things_mask] = things_labels + 1

            return new_labels

    def forward(self, inst_probabilities, labels_gt, stuff_mask, points_3d=None, *args, **kwargs):
        loss = torch.zeros_like(inst_probabilities[...,0])
        for i,(p,gt,m) in enumerate(zip(inst_probabilities, labels_gt, stuff_mask)):
            # train only where there's stuff and instances detected
            valid_mask = torch.logical_or(m, gt > 0)
            gt_valid = gt[valid_mask]
            p_valid = p[valid_mask]
            
            p_3d_valid = points_3d[i][valid_mask] if points_3d is not None else None
            
            # map labels
            virt_labels = self.create_virtual_gt_with_linear_assignment(p_valid, gt_valid, p_3d_valid)
            
            predicted_labels = p_valid.argmax(dim=-1)

            # opt1: train only wrong labels
            # wrong_mask = virt_labels != predicted_labels            
            # p_wrong = p_valid[wrong_mask]
            # virt_labels_wrong = virt_labels[wrong_mask]

            # loss += F.nll_loss(torch.log(p_wrong + 1e-27), virt_labels_wrong, reduction='mean')
            
            # opt2: train all when there's at least 1 wrong pixel
            if torch.any(virt_labels != predicted_labels):
                loss[i][valid_mask] = F.nll_loss(torch.log(p_valid + 1e-27), virt_labels, reduction='none')
                     
        return loss