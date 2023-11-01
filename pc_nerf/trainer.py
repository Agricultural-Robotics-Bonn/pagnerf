# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
from pathlib import Path
import pickle
import time
import datetime
import logging as log
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import DataLoader
from wisp.trainers import BaseTrainer
from wisp.utils import PerfTimer
from wisp.ops.image import write_png
from wisp.datasets import default_collate
from wisp.core import Rays
# from wisp.ops.image.metrics import psnr, lpips, ssim
from functools import partial
from pc_nerf.ba_pipeline import BAPipeline
from pc_nerf.ba_pipeline_lie import BAPipelineLie

from utils.lod_anneling import LODAnneling
from loss.regularizers import segment_consistency_regularizer

from torchmetrics import JaccardIndex as IoU
from torchmetrics import PeakSignalNoiseRatio as psnr
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils.metrics.panoptic_quality import PanopticQuality as PQ

from utils.outlier_rejection import rays_to_3d_points, mask_center_of_mass_outlier_rejection

from datasets.transforms.ray_sampler import SampleRays

from imgviz.label import label_colormap, label2rgb
from imgviz.instances import instances2rgb
from imgviz.depth import depth2rgb
import imageio

from loss.regularizers import grid_tv_l1_loss, grid_tv_l2_loss, sigma_sparsity_loss
from kornia.morphology import opening

from kaolin.render.camera import Camera
from kaolin.visualize import Timelapse

class PanopticTrainer(BaseTrainer):
    """ A contrastive trainer for optimizing a single neural radiance field object.
    This contrastive merely serves as an example: users should override and modify according to their project requirements.
    """
    def __init__(self, *args, **kwargs):

        self.inst_loss_type = kwargs['extra_args']['inst_loss']
        if self.inst_loss_type == 'sup_contrastive':
            from loss.sup_contrastive import SupConLoss
            self.inst_loss = SupConLoss
        elif self.inst_loss_type == 'linear_assignment':
            from loss.lin_assignment import LinAssignmentLoss
            self.inst_loss = LinAssignmentLoss
        elif self.inst_loss_type == 'linear_assignment_things':
            from loss.lin_assignment_things import LinAssignmentThingsLoss
            self.inst_loss = LinAssignmentThingsLoss
        elif self.inst_loss_type:
            raise ValueError(f'instance loss type "{self.inst_loss}" not supported.')
        
        if self.inst_loss:
            args_inst = {k:v for k,v in kwargs['extra_args'].items() if k in self.inst_loss.__init__.__code__.co_varnames}
            args_inst['temperature'] = kwargs['extra_args']['inst_temperature']
            args_inst['pn_ratio'] = kwargs['extra_args']['inst_pn_ratio']
            args_inst['outlier_rejection'] = kwargs['extra_args']['inst_outlier_rejection']
            self.inst_loss = self.inst_loss(**args_inst)

        # color parameters
        self.rgb_weight = kwargs['extra_args']['rgb_weight']

        # semantic parameters
        self.sem_weight = kwargs['extra_args']['sem_weight']
        self.sem_epoch_start = kwargs['extra_args']['sem_epoch_start']
        self.sem_conf_enable = kwargs['extra_args']['sem_conf_enable']
        self.sem_inst_weight = kwargs['extra_args']['contrast_sem_weight']
        self.sem_temperature = kwargs['extra_args']['sem_temperature']
        self.sem_segment_reg_weight = kwargs['extra_args']['sem_segment_reg_weight']
        self.inst_segment_reg_weight = kwargs['extra_args']['inst_segment_reg_weight']
        self.inst_segment_reg_epoch_start = kwargs['extra_args']['inst_segment_reg_weight']

        # instance params
        self.inst_weight = kwargs['extra_args']['inst_weight']
        self.inst_dist_func = kwargs['extra_args']['inst_dist_func']
        self.inst_conf_enable = kwargs['extra_args']['inst_conf_enable']
        self.inst_epoch_start = kwargs['extra_args']['inst_epoch_start']
        self.inst_conf_bootstrap_epoch_start = kwargs['extra_args']['inst_conf_bootstrap_epoch_start']
        self.inst_outlier_rejection = kwargs['extra_args']['inst_outlier_rejection']

        # otimizer parameters
        self.optimize_extrinsics = kwargs['extra_args']['optimize_extrinsics']
        self.extrinsics_epoch_start = kwargs['extra_args']['extrinsics_epoch_start']
        self.extrinsics_epoch_end = kwargs['extra_args']['extrinsics_epoch_end']

        self.extrinsics_lr = kwargs['extra_args']['extrinsics_lr']

        self.use_lr_scheduler = kwargs['extra_args']['use_lr_scheduler']
        self.lr_scheduler_type = kwargs['extra_args']['lr_scheduler_type']
        # One cycle lr scheduler params
        self.lr_warmup_epochs = kwargs['extra_args']['lr_warmup_epochs']
        self.lr_div_factor = kwargs['extra_args']['lr_div_factor']
        # Step lr scheduler params
        self.lr_step_size = kwargs['extra_args']['lr_step_size']
        self.lr_step_gamma = kwargs['extra_args']['lr_step_gamma']

        self.use_lod_anneling = kwargs['extra_args']['lod_anneling']
        self.lod_annel_epochs = kwargs['extra_args']['lod_annel_epochs']
        self.lod_annel_epoch_start = kwargs['extra_args']['lod_annel_epoch_start']

        self.delta_grid_lr_weight = kwargs['extra_args']['delta_grid_lr_weight']

        # regularizers
        self.grid_tvl1_reg = kwargs['extra_args']['grid_tvl1_reg']
        self.grid_tvl2_reg = kwargs['extra_args']['grid_tvl2_reg']
        self.delta_grid_tvl1_reg = kwargs['extra_args']['delta_grid_tvl1_reg']
        self.delta_grid_tvl2_reg = kwargs['extra_args']['delta_grid_tvl2_reg']

        self.tv_window_size = kwargs['extra_args']['tv_window_size']
        self.tv_edge_num_samples = kwargs['extra_args']['tv_edge_num_samples']

        self.ray_sparcity_reg = kwargs['extra_args']['ray_sparcity_reg']

        self.inst_num_dilations = kwargs['extra_args']['inst_num_dilations']


        # Validation params
        self.val_mip = kwargs['extra_args']['val_mip']
        self.num_clustering_samples = kwargs['extra_args']['num_clustering_samples']
        self.num_val_frames_to_save = kwargs['extra_args']['num_val_frames_to_save']
        self.render_val_labels = kwargs['extra_args']['render_val_labels']
        
        self.dataset_num_workers = kwargs['extra_args']['dataset_num_workers']

        # val dataset for val pose alignment to NeF
        self.val_dataset = kwargs['extra_args']['val_dataset']
        self.optimize_val_extrinsics = kwargs['extra_args']['optimize_val_extrinsics']
        self.val_extrinsics_start = kwargs['extra_args']['val_extrinsics_start']
        self.val_extrinsics_end = kwargs['extra_args']['val_extrinsics_end']
        self.val_extrinsics_every = kwargs['extra_args']['val_extrinsics_every']

        self.prune_at_epoch = kwargs['extra_args']['prune_at_epoch']
        self.prune_at_start = kwargs['extra_args']['prune_at_start']

        self.low_res_val = kwargs['extra_args']['low_res_val']

        self.save_grid = kwargs['extra_args']['save_grid']

        self.save_preds = kwargs['extra_args']['save_preds']

        self.sem_softmax = kwargs['extra_args']['sem_softmax']

        super().__init__(*args,**kwargs)
        
        self.num_epochs
        self.val_extrinsics_end = self.val_extrinsics_end if self.val_extrinsics_end >= 0 else self.num_epochs
        self.extrinsics_epoch_end = self.extrinsics_epoch_end if self.extrinsics_epoch_end >=0 else self.num_epochs



        if self.use_lr_scheduler and self.lr_scheduler_type == 'one_cycle':
            one_cycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(torch.optim.Adam([torch.Tensor()]),
                                                                    epochs=self.num_epochs+1,
                                                                    max_lr=1,
                                                                    steps_per_epoch=len(self.train_data_loader),
                                                                    pct_start=float(self.lr_warmup_epochs/self.num_epochs),
                                                                    div_factor=self.lr_div_factor,
                                                                    final_div_factor=self.lr_div_factor)
            
            def step_lambda(step, one_cycle):
                one_cycle.last_epoch = step
                return one_cycle.get_lr()[0]

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, partial(step_lambda, one_cycle=one_cycle_scheduler))
        elif self.use_lr_scheduler and self.lr_scheduler_type == 'step':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                step_size=self.lr_step_size * len(self.train_data_loader),
                                                                gamma=self.lr_step_gamma
                                                                )
            
        elif self.use_lr_scheduler and self.lr_scheduler_type == 'panoptic_step':
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                  lr_lambda=[lambda epoch:
                                                                  self.lr_step_gamma if (epoch != 0) and (epoch % (self.lr_step_size * len(self.train_data_loader)) == 0) and \
                                                                  any(p in g['name'] for p in ['sem','inst','delta'])
                                                                  else 1.0
                                                                  for g in self.optimizer.param_groups])
        

        if self.use_lod_anneling:
            self.lod_anneler = LODAnneling(self.pipeline.nef,
                                            epochs=self.lod_annel_epochs,
                                            steps_per_epoch=len(self.train_data_loader),
                                            )
        
        if self.save_grid:
            self.grid_timelapse = Timelapse(os.path.join(self.log_dir, 'grid'))

        self.training_time = 0.0

            

    def init_dataloader(self):
        self.train_dl = DataLoader(self.dataset,
                                   batch_size=self.batch_size,
                                   collate_fn=default_collate,
                                   shuffle=True, pin_memory=True, num_workers=self.dataset_num_workers)
        
        if self.val_dataset is not None:
            self.val_dl = DataLoader(self.val_dataset,
                                     batch_size=self.batch_size,
                                     collate_fn=default_collate,
                                     shuffle=True, pin_memory=True, num_workers=self.dataset_num_workers)
    
        self.train_data_loader = self.train_dl

    def init_optimizer(self):
        params_dict = { name : param for name, param in self.pipeline.nef.named_parameters() }
        
        params = []
        decoder_params = []
        inst_params = []
        sem_params = []
        grid_params = []
        delta_grid_params = []
        rest_params = []

        for name in params_dict:
            
            if 'decoder' in name:
                # If "decoder" is in the name, there's a good chance it is in fact a decoder,
                # so use weight_decay
                decoder_params.append(params_dict[name])
            elif 'inst' in name:
                inst_params.append(params_dict[name])
            elif 'sem' in name:
                sem_params.append(params_dict[name])
            elif 'delta_grid' in name:
                delta_grid_params.append(params_dict[name])
            elif 'grid' in name:
                # If "grid" is in the name, there's a good chance it is in fact a grid,
                # so use grid_lr_weight
                grid_params.append(params_dict[name])

            else:
                rest_params.append(params_dict[name])

        params.append({"params" : decoder_params,
                       "lr": self.lr,
                       "name": "decoder"}),
        
        params.append({"params" : sem_params,
                       "lr": self.lr,
                       "name": "sem"}),
        
        params.append({"params" : inst_params,
                       "lr": self.lr,
                       "name": "inst"}),
        
        params.append({"params" : delta_grid_params,
                       "lr": self.lr * self.delta_grid_lr_weight,
                       "weight_decay": self.weight_decay,
                       "name": "delta_grid"})
        

        params.append({"params" : grid_params,
                       "lr": self.lr * self.grid_lr_weight,
                       "weight_decay": self.weight_decay,
                       "name": "grid"})
        
        
        params.append({"params" : rest_params,
                       "lr": self.lr,
                       "name": "rest"})

        self.optimizer = self.optim_cls(params, **self.optim_params)

        if self.optimize_extrinsics:
            assert isinstance(self.pipeline, (BAPipeline, BAPipelineLie)), (f'Camera extrinsics optimization was requested, but pipeline is of class "{type(self.pipeline)}"',
                                                            'a BAPipeline is required. Check your configs to resolve this')
            
            lr = self.extrinsics_lr if self.extrinsics_lr >= 0 else self.lr            
            
            # Add train camera patameters to optimizer
            self.optimizer.add_param_group({"params" : self.pipeline.cameras.extrinsics.parameters(),
                                            "lr": lr,
                                            "name": "extrinsics"
                                            })

    def begin_epoch(self):
        # Add estrinisics parameters to optimizer if required
        if self.optimize_extrinsics:
            assert isinstance(self.pipeline, (BAPipeline, BAPipelineLie)), (f'Camera extrinsics optimization was requested, but pipeline is of class "{type(self.pipeline)}"',
                                                           'a BAPipeline is required. Check your configs to resolve this')      
            # Add train camera patameters to optimizer
            self.pipeline.cameras.extrinsics.parameters().requires_grad = self.extrinsics_epoch_start <= self.epoch <= self.extrinsics_epoch_end
        
        # optimize val extrinsics for this epoch if required
        if self.optimize_val_extrinsics and \
           self.val_extrinsics_start <= self.epoch <= self.val_extrinsics_end and \
           self.epoch % self.val_extrinsics_every == 0:
            log.info('Optimizing val poses only on this epoch...')
            self.train_data_loader = self.val_dl
        
            self.training_val_poses = True
            for p in self.pipeline.nef.parameters():
                p.requires_grad = False
        
        else:
            self.train_data_loader = self.train_dl
        
            self.training_val_poses = False  
            for p in self.pipeline.nef.parameters():
                p.requires_grad = True


        super().begin_epoch()


    def pre_epoch(self, epoch):
        super().pre_epoch(epoch)
        self.epoch_start_time = time.time()
    
    def end_epoch(self):
        reint_optimizer = False
        if self.extra_args["prune_every"] > -1 and \
            self.epoch > 0 and self.epoch % self.extra_args["prune_every"] == 0 or \
            self.epoch == self.prune_at_epoch or \
            (self.prune_at_start and self.epoch == 0):
            
            log.info('Prunning grid blas...')

            self.pipeline.nef.prune()
            reint_optimizer = True
        
        if 'num_resolutions' in vars(self.pipeline.nef.grid) and \
            self.epoch > 0 and \
            (self.epoch % (self.num_epochs // self.pipeline.nef.grid.num_resolutions)) == 0:
            old_res = self.pipeline.nef.grid.current_resolution
            self.pipeline.nef.grid.step_upsample_vm_grid()
            if old_res != self.pipeline.nef.grid.current_resolution:
                log.info(f'Upsampled TensoRF resolution from {old_res}^3 to {self.pipeline.nef.grid.current_resolution}^3')
                reint_optimizer = True

        if reint_optimizer:    
            self.init_optimizer()
        
        super().end_epoch()

        if self.extra_args["voxel_raymarch_epoch_start"] == self.epoch:
            log.info(f'Changing from {self.pipeline.nef.raymarch_type} to voxel raymarch...')
            self.pipeline.nef.raymarch_type = 'voxel'
            self.pipeline.tracer.raymarch_type = 'voxel'
            self.pipeline.tracer.num_steps = self.extra_args["samples_per_voxel"]

    
    def init_log_dict(self):
        """Custom log dict. """
        #clear validation results from dict to avoid flat values in tensorboard plots
        self.log_dict = {k:v for k,v in self.log_dict.items() if 'val' not in k}
        if self.training_val_poses:
            self.log_dict['rgb_val_pose_loss'] = 0.0
        else:
            self.log_dict['total_loss'] = 0
            self.log_dict['total_iter_count'] = 0
            self.log_dict['rgb_loss'] = 0.0
            self.log_dict['sem_loss'] = 0.0
            self.log_dict['contrast_sem_loss'] = 0.0
            self.log_dict['inst_loss'] = 0.0


    #################################################
    ## Training Step
    #################################################

    def step(self, epoch, n_iter, data):

        """Implement the optimization for color and semantic losses. """
        self.scene_state.optimization.iteration = n_iter

        timer = PerfTimer(activate=False, show_memory=False)

        # Map to device
        batch_size = data['imgs'].shape[0]

        img_gts = data['imgs'].to(self.device).squeeze(0)

        if 'semantics' in data and epoch >= self.sem_epoch_start:
            if 'semantics_pred' in data:
                sem_gts = data['semantics_pred'].to(self.device).squeeze()
            else:
                sem_gts = data[[k for k in data if 'semantics' in k][0]].to(self.device).squeeze()
        
        if 'instance' in data and epoch >= self.inst_epoch_start:
            if 'instance_pred' in data:
                inst_gts = data['instance_pred'].to(self.device).squeeze()
            else:
                inst_gts = data[[k for k in data if 'instance' in k][0]].to(self.device).squeeze()
        
        if 'sem_conf' in data and epoch >= self.sem_epoch_start:
            sem_conf = data['sem_conf'].to(self.device).squeeze()
        
        if 'inst_conf' in data and epoch >= self.inst_epoch_start:
            inst_conf = data['inst_conf'].to(self.device).squeeze()
        
        # for extrinsics optimization
        cam_ids = data['cam_id'] if 'cam_id' in data else None
        
        rays_key = 'rays' if not isinstance(self.pipeline, (BAPipeline, BAPipelineLie)) else 'base_rays'
        rays = data[rays_key].to(self.device)
          
        timer.check("map to device")

        self.optimizer.zero_grad(set_to_none=True)
        timer.check("zero grad")
        loss = 0
        with torch.cuda.amp.autocast():
            channels = ['rgb']
            channels += ['semantics'] if epoch >= self.sem_epoch_start and not self.training_val_poses else []
            channels += ['inst_embedding'] if epoch >= self.inst_epoch_start and not self.training_val_poses else []
            channels += ['depth'] if self.inst_outlier_rejection else []
            
            rb = self.pipeline(rays=rays, lod_idx=None, channels=channels, cam_ids=cam_ids, stage='train')
            timer.check("inference")

            if 'ray_sparcity_loss' in dir(rb):
                loss += rb.ray_sparcity_loss

            # RGB Loss
            if self.rgb_weight > 0.0:
                rgb_loss = torch.abs(rb.rgb[..., :3] - img_gts[..., :3].reshape(-1,3))
                rgb_loss = rgb_loss.mean()
            
                loss += self.rgb_weight * rgb_loss
                if not self.training_val_poses:
                    self.log_dict['rgb_loss'] += rgb_loss.item()
                else:
                    self.log_dict['rgb_val_pose_loss'] += rgb_loss.item()
            
            if not self.training_val_poses:
                # Semantic Loss
                if 'semantics' in data and epoch >= self.sem_epoch_start and \
                    self.sem_weight > 0. and \
                    not self.training_val_poses:
                    
                    if self.sem_softmax:
                        sem_loss = F.nll_loss(torch.log(rb.semantics + 1e-27) / self.sem_temperature, sem_gts.reshape(-1), reduction='none')
                    else:
                        sem_loss = F.cross_entropy(rb.semantics / self.sem_temperature, sem_gts.reshape(-1), reduction='none')
                    
                    if 'sem_conf' in  data and self.sem_conf_enable:
                        sem_loss *= sem_conf.reshape(-1)
                    sem_loss = torch.mean(sem_loss)

                    if self.sem_segment_reg_weight > 0.0:
                        sem_loss += self.sem_segment_reg_weight * segment_consistency_regularizer((rb.semantics + 1e-27).reshape(batch_size, -1, rb.semantics.shape[-1]),
                                                                        sem_gts.reshape(batch_size, -1))
                        
                    sem_loss = sem_loss.mean()
                    loss += self.sem_weight * sem_loss
                    self.log_dict['sem_loss'] += sem_loss.item()



                    if self.sem_inst_weight > 0.:
                        contrast_sem_loss = self.inst_loss((rb.semantics + 1e-27).reshape(batch_size, -1, rb.semantics.shape[-1]), sem_gts.reshape(batch_size, -1))
                        loss += self.sem_inst_weight * contrast_sem_loss
                        self.log_dict['contrast_sem_loss'] += contrast_sem_loss.item()


                # Instance loss
                if self.inst_loss and epoch >= self.inst_epoch_start and \
                    epoch >= self.sem_epoch_start and self.inst_weight > 0. and\
                    not self.training_val_poses:
                    
                    inst_embed = vars(rb)['inst_embedding']
                    inst_embed = inst_embed.reshape(batch_size, -1, inst_embed.shape[-1])
                    inst_gts = inst_gts.reshape(batch_size, -1)

                    
                    if self.inst_loss_type in ['linear_assignment', 'linear_assignment_things']:
                        stuff_ids = torch.tensor(self.dataset.semantic_info['stuff_ids']).to(sem_gts.device)
                        stuff_mask = torch.isin(sem_gts, stuff_ids)
                        
                    # # Opt1: no loss where there is no istance but semantics says otherwise
                    # ##################################################################################################
                    if 'contrastive' in self.inst_loss_type:
                        things_ids = torch.tensor(self.dataset.semantic_info['things_ids']).to(sem_gts.device)
                        undetected_mask = torch.logical_and(torch.isin(sem_gts, things_ids), inst_gts == 0)
                        inst_mask = torch.logical_not(undetected_mask)
                        inst_loss = self.inst_loss(inst_embed, inst_gts, reduction='mean', anchor_mask=inst_mask)
                    
                    elif self.inst_loss_type == 'linear_assignment':
                        inst_loss = self.inst_loss(inst_embed, inst_gts, stuff_mask)
    
                    elif self.inst_loss_type == 'linear_assignment_things':
                        points_3d = None
                        if self.inst_outlier_rejection:
                            with torch.no_grad():
                                if isinstance(self.pipeline, BAPipeline):
                                    cameras = self.pipeline.get_cameras_from_ids(cam_ids)
                                else:
                                    cameras = Camera.cat([self.dataset.data['cameras'][cid] for cid in cam_ids])
                                    cameras = cameras.to(rays.dirs.device)

                                points_3d = rays_to_3d_points(rays, rb.depth, cameras).reshape(batch_size, -1, 3)
                        inst_loss = self.inst_loss(inst_embed, inst_gts, stuff_mask=stuff_mask, points_3d=points_3d)
                    
                    else:
                        # this should never happend since loss type check is done at trainer init
                        raise NotImplemented(f'Instance loss of type {self.inst_loss_type} not implemented.')

                    if self.inst_segment_reg_weight > 0.0 and self.inst_segment_reg_epoch_start > 0 and epoch > self.inst_segment_reg_epoch_start:
                        inst_loss += self.inst_segment_reg_weight * segment_consistency_regularizer((inst_embed + 1e-27).reshape(batch_size, -1, inst_embed.shape[-1]),
                                                                        inst_gts.reshape(batch_size, -1))



                    # Opt2: Small loss for where there's no detected instances (very bad)
                    ##################################################################################################
                    # inst_loss = self.inst_loss(inst_embed, inst_gts, reduction='none')
                    # inst_loss *= (inst_gts == 0) * 0.01

                    # Op3: do nowhing about it
                    ##################################################################################################
                    # inst_loss = self.inst_loss(inst_embed, inst_gts, reduction='none')

                    # Op4: Run contrastive only on things and low regularization of background from sematics
                    ##################################################################################################
                    # # contrastiv only on things
                    # things_embed, things_gts = (inst_embed[inst_gts!=0][None], inst_gts[inst_gts!=0][None])
                    # inst_loss = self.inst_loss(things_embed, things_gts, reduction='mean')
                    # # background regularization
                    # stuff_ids = torch.tensor(self.dataset.semantic_info['stuff_ids']).to(sem_gts.device)
                    # stuff_embed = F.normalize(inst_embed[torch.isin(sem_gts, stuff_ids)])                
                    # inst_loss += F.cross_entropy(stuff_embed, torch.ones(stuff_embed.shape[0], dtype=torch.int64).to(stuff_embed.device))*10
                    if 'inst_conf' in data and self.inst_conf_enable:
                        inst_loss *= inst_conf

                    inst_loss = inst_loss.mean()

                    loss += self.inst_weight * inst_loss
                    self.log_dict['inst_loss'] += inst_loss.item()
                if self.grid_tvl1_reg > 0.0:
                    loss += grid_tv_l1_loss(self.pipeline.nef.grid.interpolate,
                                            sample_size=self.tv_window_size,
                                            num_dim_samples=self.tv_edge_num_samples).mean() * self.grid_tvl1_reg
                if self.grid_tvl2_reg > 0.0:
                    loss += grid_tv_l2_loss(self.pipeline.nef.grid.interpolate,
                                            sample_size=self.tv_window_size,
                                            num_dim_samples=self.tv_edge_num_samples).mean() * self.grid_tvl2_reg
                    

                inst_nef_func = lambda x: self.pipeline.nef(coords=x, ray_d=None, channels='inst_embedding')
                if self.delta_grid_tvl1_reg > 0.0 and 'delta_grid' in dir(self.pipeline.nef):
                    loss += grid_tv_l1_loss(inst_nef_func,
                                            sample_size=self.tv_window_size,
                                            num_dim_samples=self.tv_edge_num_samples).mean() * self.delta_grid_tvl1_reg
                if self.delta_grid_tvl2_reg > 0.0 and 'delta_grid' in dir(self.pipeline.nef):
                    loss += grid_tv_l1_loss(inst_nef_func,
                                            sample_size=self.tv_window_size,
                                            num_dim_samples=self.tv_edge_num_samples).mean() * self.delta_grid_tvl2_reg

                timer.check("loss")

                self.log_dict['total_loss'] += loss.item()
        
        self.log_dict['total_iter_count'] += 1
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        timer.check("backward and step")

        if self.use_lr_scheduler:
            self.lr_scheduler.step()
        
        timer.check("lr scheduler step")

        if self.use_lod_anneling and epoch >= self.lod_annel_epoch_start:
            self.lod_anneler.step()
            self.scene_state.optimization.losses['anneling'] = self.lod_anneler.anneling_fn(torch.linspace(0, self.pipeline.nef.num_lods, 300), epoch * len(self.train_data_loader) + n_iter).tolist()
            self.scene_state.optimization.losses['annel_weights'] = self.pipeline.nef.lod_weights.tolist()

        timer.check("lod anneler step")

    def log_tb(self, epoch):
        self.epoch_time = time.time() - self.epoch_start_time

        self.training_time += self.epoch_time

        log_text = 'EPOCH {}/{}'.format(epoch, self.num_epochs)
        log_text += f' {self.epoch_time:.2f}s'

        if not self.training_val_poses:
            self.log_dict['total_loss'] /= self.log_dict['total_iter_count']
            log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'])
            self.log_dict['rgb_loss'] /= self.log_dict['total_iter_count']
            log_text += ' | rgb loss: {:>.3E}'.format(self.log_dict['rgb_loss'])
            self.log_dict['sem_loss'] /= self.log_dict['total_iter_count']
            log_text += ' | sem loss: {:>.3E}'.format(self.log_dict['sem_loss'])
            self.log_dict['inst_loss'] /= self.log_dict['total_iter_count']
            log_text += ' | semi-sup loss: {:>.3E}'.format(self.log_dict['inst_loss'])
            for key in self.log_dict:
                if 'loss' in key and 'val' not in key:
                    self.writer.add_scalar(f'Loss/{key}', self.log_dict[key], epoch)
        else:
            self.log_dict['rgb_val_pose_loss'] /= self.log_dict['total_iter_count']
            log_text += ' | rgb val pose loss: {:>.3E}'.format(self.log_dict[f'rgb_val_pose_loss'])
            self.writer.add_scalar(f'Loss/{"rgb_val_pose_loss"}', self.log_dict['rgb_val_pose_loss'], epoch)

        log_text += f' | Total train time: {str(datetime.timedelta(seconds=self.training_time))} ({self.training_time:.2f}s)'


        log.info(log_text)

        self.pipeline.eval()


    #################################################
    ## Validation
    #################################################

    def batch_render(self, pipeline, rays, channels=['rgb'], cam_ids=None):
  
        if isinstance(pipeline, BAPipeline):
            rays = pipeline.transform_rays(rays, cam_ids)

        render_batch = self.renderer.render_batch
        render = lambda ray_pack: pipeline(rays=ray_pack, lod_idx=None, channels=channels)
        for i, ray_pack in enumerate(rays.split(render_batch)):
            if i == 0:
                rb = render(ray_pack)
            else:
                rb += render(ray_pack)
        return rb

    def evaluate_metrics(self, epoch, data, name=''):

        rgb_imgs = []

        sem_info = self.dataset.semantic_info  if 'semantic_info' in vars(self.dataset) else None 
        if sem_info is not None:
            sem_imgs = []
            sem_rgb_imgs = []
            sem_rgb_pred_imgs = []

            inst_imgs = []
            inst_pred_imgs = []
            inst_confs = []
            inst_confs_pred = []
            inst_rgb_imgs = []
            inst_rgb_pred_imgs = []
            depth_imgs = []
            
            sem_cmap = label_colormap(sem_info['num_classes'])
            iou_metric = IoU(num_classes=sem_info['num_classes'], task='multiclass', average='macro').to('cuda')
            iou_metric_pred = IoU(num_classes=sem_info['num_classes'], task='multiclass', average='macro').to('cuda')
            pq_metric = PQ(things=set(sem_info['things_ids']), stuff=set(sem_info['stuff_ids']),allow_unknown_preds_category=True).to('cuda')
            pq_metric_pred = PQ(things=set(sem_info['things_ids']), stuff=set(sem_info['stuff_ids']),allow_unknown_preds_category=True).to('cuda')
            map_metric = MeanAveragePrecision(iou_type="segm")
            map_metric_pred = MeanAveragePrecision(iou_type="segm")
            
        psnr_metric = psnr().to('cuda')

        num_imgs = len(data['imgs'])
        cam_ids = data['cameras_ts'] if 'cameras_ts' in data else torch.arange(num_imgs)

        render_time = 0.0
        with torch.no_grad():
            for idx in tqdm(range(num_imgs)):
                gts =  data['imgs'][idx]
                
                if self.optimize_val_extrinsics:
                    rays = data['base_rays'].reshape(-1,3).to('cuda') 
                else:
                    rays = data['rays'][idx].reshape(-1,3).to('cuda') 

                channels = ['rgb']
                if sem_info is not None:
                    channels += ['semantics'] if epoch >= self.sem_epoch_start else []
                    channels += ['inst_embedding'] if epoch >= self.inst_epoch_start else []
                
                channels += ['depth']
                
                render_start_time = time.time()
                rb = self.batch_render(self.pipeline,
                                       rays,
                                       channels=channels,
                                       cam_ids=[cam_ids[idx]])
                render_time += time.time() - render_start_time

                rb = rb.reshape(*gts.shape[:2], -1)

                psnr_metric.update(rb.rgb[...,:3], gts[...,:3].cuda())

                rgb_imgs.append(rb.cpu().image().byte().rgb.numpy())

                depth_imgs.append(depth2rgb(rb.depth.squeeze().cpu().numpy()))

                sem_gts_img = None
                if 'semantics' in data and epoch >= self.sem_epoch_start:
                    sem_gts = data['semantics'][idx].squeeze()
                    semantics = torch.argmax(rb.semantics, dim=-1)
                    # compute metrics only in frames with labels
                    if not torch.all(sem_gts == -1):
                        iou_metric.update(semantics, sem_gts.cuda())
                        sem_gts_img = label2rgb(sem_gts, colormap=sem_cmap)

                    sem_imgs.append(label2rgb(semantics.cpu(), colormap=sem_cmap))
                    sem_rgb_imgs.append(label2rgb(semantics.cpu(), image=rgb_imgs[-1], colormap=sem_cmap))
                    if 'semantics_pred' in data:
                        sem_pred = data['semantics_pred'][idx].squeeze()
                        sem_rgb_pred_imgs.append(label2rgb(sem_pred.cpu(), image=rgb_imgs[-1], colormap=sem_cmap))
                        if not torch.all(sem_gts == -1):
                            iou_metric_pred.update(sem_pred.cuda(), sem_gts.cuda())

                inst_gts_img = None
                if 'instance' in data and epoch >= self.inst_epoch_start:
                    inst_gts = data[[k for k in data if 'instance' in k if 'preds' not in k][0]][idx].squeeze()

                    # build imap and all needed for pq
                    #   [b, (semantic_mask, imap), h, w]
                    if 'contrastive' in  self.inst_loss_type:
                        instances = self.pipeline.nef.predict_clusters(rb.inst_embedding)
                    else:
                        instances = torch.argmax(rb.inst_embedding, dim=-1)
                        inst_conf = torch.max(rb.inst_embedding, dim=-1)[0]
                        inst_confs.append(depth2rgb(inst_conf.cpu().numpy(), 0, 1))
                    
                    if 'inst_conf' in data:
                        inst_conf_pred = data['inst_conf'][idx].squeeze()
                        inst_confs_pred.append(depth2rgb(inst_conf_pred.cpu().numpy(), 0, 1))



                    # clean instances with erosion dialtion
                    inst_rgb = rgb_imgs[-1].copy()
                    inst_type = instances.dtype
                    mask_ids = instances.unique()
                    if mask_ids.shape[0] > 1:
                        inst_masks = (instances == mask_ids[1:][:, None, None]).type(inst_type)
                        if self.inst_num_dilations > 0:
                            kernel = torch.ones(3,3).to(inst_masks.device)
                            for _ in range(self.inst_num_dilations):
                                inst_masks = opening(inst_masks[None], kernel)[0].type(inst_type)

                        if self.inst_outlier_rejection:
                            inst_masks = mask_center_of_mass_outlier_rejection(inst_masks)

                        
                        # remove small masks
                        small_masks = inst_masks.sum(dim=[1,2]) < 100 #(0.0005 * inst_masks.shape[-1] * inst_masks.shape[-2])
                        inst_masks[small_masks] = torch.zeros_like(inst_masks[0])
                        

                        # cat a background channel for correct ID assingment
                        instances = torch.cat(((inst_masks.sum(dim=0) == 0).type(inst_type)[None], inst_masks))
                        instances = mask_ids[torch.argmax(instances, dim=0)]
                        non_empty_masks = inst_masks.sum(dim=[1,2]).bool()
                        if torch.any(non_empty_masks):
                            bboxes = masks_to_boxes(inst_masks[non_empty_masks])
                            bbox_cmap = label_colormap(instances.max().item()+1)[instances.unique().cpu()][1:]
                            rgb_bbox = draw_bounding_boxes(rb.image().byte().rgb.movedim(2,0), boxes=bboxes, colors=list(map(tuple,bbox_cmap)), width=6)
                            inst_rgb = rgb_bbox.movedim(0,2).cpu().numpy()
                    
                    else:
                        inst_masks = torch.zeros_like(instances[None])

                    # compute metrics only in frames with labels
                    if not torch.all(sem_gts == -1) and not torch.all(inst_gts == -1):
                        panoptic = torch.cat((semantics[None], instances[None]), dim=0)[None]
                        
                        panoptic_labels = torch.cat((sem_gts.cuda()[None], inst_gts.cuda()[None]))[None]
                        pq_metric.update(panoptic, panoptic_labels)

                        inst_gts = inst_gts.cuda()
                        gt_ids = inst_gts.unique()
                        gt_masks = (inst_gts == gt_ids[1:][:, None, None]).type(inst_gts.dtype)

                        map_metric.update(preds = [{'masks':inst_masks.type(torch.uint8),
                                                    'scores':torch.ones(inst_masks.shape[0]).cuda(),
                                                    'labels':torch.ones(inst_masks.shape[0]).cuda()}],
                                          target= [{'masks':gt_masks.type(torch.uint8),
                                                    'labels':torch.ones(gt_masks.shape[0]).cuda()}])
                        
                        inst_gts_img = label2rgb(inst_gts.cpu(), colormap=label_colormap(inst_gts.max()+1))

                    inst_imgs.append(label2rgb(instances.cpu(), colormap=label_colormap(instances.max().item()+1)))
                    
                    inst_pxs = inst_imgs[-1] != 0
                    inst_rgb[inst_pxs] = (1 - 0.7) * inst_rgb[inst_pxs] + 0.7 * inst_imgs[-1][inst_pxs]
                    inst_rgb_imgs.append(inst_rgb)

                    if 'instance_pred' in data:
                        inst_pred = data['instance_pred'][idx].squeeze()
                        inst_pred_imgs.append(label2rgb(inst_pred.cpu(), colormap=label_colormap(inst_pred.max().item()+1)))
                        panoptic_preds = torch.cat((sem_pred[None], inst_pred[None]), dim=0)[None]

                        #############################################################
                        inst_pred = inst_pred.cuda()
                        inst_pred_rgb = rgb_imgs[-1].copy()
                        inst_pred_type = inst_pred.dtype
                        mask_pred_ids = inst_pred.unique()
                        if mask_pred_ids.shape[0] > 1:
                            inst_pred_masks = (inst_pred == mask_pred_ids[1:][:, None, None]).type(inst_pred_type)

                            non_empty_masks = inst_pred_masks.sum(dim=[1,2]).bool()
                            bboxes = masks_to_boxes(inst_pred_masks[non_empty_masks])
                            bbox_cmap = label_colormap(inst_pred.max().item()+1)[inst_pred.unique().cpu()][1:]
                            rgb_bbox = draw_bounding_boxes(rb.image().byte().rgb.movedim(2,0), boxes=bboxes, colors=list(map(tuple,bbox_cmap)), width=6)
                            inst_pred_rgb = rgb_bbox.movedim(0,2).cpu().numpy()
                        
                        inst_pxs = inst_pred_imgs[-1] != 0
                        inst_pred_rgb[inst_pxs] = (1 - 0.7) * inst_pred_rgb[inst_pxs] + 0.7 * inst_pred_imgs[-1][inst_pxs]
                        inst_rgb_pred_imgs.append(inst_pred_rgb)
                        #############################################################

                        if not torch.all(sem_gts == -1) and not torch.all(inst_gts == -1):
                            pq_metric_pred.update(panoptic_preds.cuda(), panoptic_labels)

                            inst_pred = inst_pred.cuda()
                            pred_ids = inst_pred.unique()
                            inst_masks = (inst_pred == pred_ids[1:][:, None, None]).type(inst_pred.dtype)

                            map_metric_pred.update(preds = [{'masks':inst_masks.type(torch.uint8),
                                                             'scores':torch.ones(inst_masks.shape[0]).cuda(),
                                                             'labels':torch.ones(inst_masks.shape[0]).cuda()}],
                                                   target= [{'masks':gt_masks.type(torch.uint8),
                                                             'labels':torch.ones(gt_masks.shape[0]).cuda()}])
                if self.save_preds and sem_info is not None and epoch >= self.sem_epoch_start and epoch >= self.inst_epoch_start:
                    # pred tensor: [[sem,inst,inst_confs],h,w]
                    panoptic_path = (Path(self.log_dir).parent.parent / 'panoptic')
                    panoptic_path.mkdir(parents=True, exist_ok=True)
                    with open(panoptic_path / (data['filenames'][idx].split('.')[0] + '.pkl'), 'wb') as f:
                        pickle.dump(torch.cat((semantics[None], instances[None]), dim=0).type(torch.uint8).cpu(), f)
                    conf_path = (Path(self.log_dir).parent.parent / 'inst_conf')
                    conf_path.mkdir(parents=True, exist_ok=True)
                    with open(conf_path / (data['filenames'][idx].split('.')[0] + '.pkl'), 'wb') as f:
                        pickle.dump(inst_conf.cpu(), f)

                if  idx%self.num_val_frames_to_save == 0 or \
                    self.num_val_frames_to_save >= num_imgs or \
                    (inst_gts_img is not None or sem_gts_img is not None) and self.render_val_labels:
                
                    out_name = f"{idx}"
                    # Save rgb rendered frame and gt
                    write_png(os.path.join(self.valid_log_dir, out_name + ".png"), rb.cpu().image().byte().rgb.numpy())
                    write_png(os.path.join(self.valid_log_dir, out_name + "_gt.png"), (gts[...,:3] * 255).type(torch.uint8).numpy())
                    # Save semantic rendered frame, gt and preds if loaded
                    if sem_info is not None and epoch >= self.sem_epoch_start:
                        write_png(os.path.join(self.valid_log_dir, out_name + "_sem.png"), sem_imgs[-1])
                        if 'semantics_pred' in data:
                            sem_pred_img = label2rgb(data['semantics_pred'][idx].squeeze(), colormap=sem_cmap) 
                            write_png(os.path.join(self.valid_log_dir, out_name + "_sem_pred.png"), sem_pred_img)
                        if sem_gts_img is not None:
                            write_png(os.path.join(self.valid_log_dir, out_name + "_sem_gt.png"), sem_gts_img)
                    # Save instance rendered frame, gt and preds if loaded
                    if sem_info is not None and epoch >= self.inst_epoch_start:
                        write_png(os.path.join(self.valid_log_dir, out_name + "_inst.png"), inst_imgs[-1])
                        if 'instance_pred' in data:
                            inst_pred = data['instance_pred'][idx].squeeze()
                            inst_pred_img = label2rgb(inst_pred, colormap=label_colormap(inst_pred.max().item()+1)) 
                            write_png(os.path.join(self.valid_log_dir, out_name + "_inst_pred.png"), inst_pred_img)
                        if inst_gts_img is not None:
                            write_png(os.path.join(self.valid_log_dir, out_name + "_inst_gt.png"), inst_gts_img)

            print('saving validation videos')
            imageio.mimwrite(os.path.join(self.valid_log_dir, 'rgb.mp4'), rgb_imgs, fps=15, quality=8)
            imageio.mimwrite(os.path.join(self.valid_log_dir, 'depth.mp4'), depth_imgs, fps=15, quality=8)
            if sem_info is not None and epoch >= self.sem_epoch_start:
                imageio.mimwrite(os.path.join(self.valid_log_dir, 'sem.mp4'), sem_imgs, fps=15, quality=8)
                imageio.mimwrite(os.path.join(self.valid_log_dir, 'sem_rgb.mp4'), sem_rgb_imgs, fps=15, quality=8)
                if sem_rgb_pred_imgs:
                    imageio.mimwrite(os.path.join(self.valid_log_dir, 'sem_pred_rgb.mp4'), sem_rgb_pred_imgs, fps=15, quality=8)
            if sem_info is not None and epoch >= self.inst_epoch_start:
                imageio.mimwrite(os.path.join(self.valid_log_dir, 'inst.mp4'), inst_imgs, fps=15, quality=8)
                imageio.mimwrite(os.path.join(self.valid_log_dir, 'inst_conf.mp4'), inst_confs, fps=15, quality=8)
                imageio.mimwrite(os.path.join(self.valid_log_dir, 'inst_rgb.mp4'), inst_rgb_imgs, fps=15, quality=8)
                imageio.mimwrite(os.path.join(self.valid_log_dir, 'inst_conf_pred.mp4'), inst_confs_pred, fps=15, quality=8)
                if inst_rgb_pred_imgs:
                    imageio.mimwrite(os.path.join(self.valid_log_dir, 'inst_pred.mp4'), inst_pred_imgs, fps=15, quality=8)
                    imageio.mimwrite(os.path.join(self.valid_log_dir, 'inst_pred_rgb.mp4'), inst_rgb_pred_imgs, fps=15, quality=8)

        metrics_ret = {}     
        
        psnr_total = psnr_metric.compute()
        log_text = 'EPOCH {}/{}'.format(epoch, self.num_epochs)
        log_text += f', Render time/img {render_time / num_imgs:.2f}s'
        log_text += ' | {}: {:.2f}'.format(f"{name} PSNR", psnr_total)
        metrics_ret['val/psnr'] = psnr_total.item()
        
        if sem_info is not None:
            # IoU gain
            iou_gain = iou_metric.compute() - iou_metric_pred.compute()
            log_text += ' | {}: {:.6f}'.format(f"{name} iou_gain", iou_gain)
            metrics_ret['val/iou_gain'] = iou_gain.item()
            # PQ things gain
            pq_things_gain = pq_metric.compute()['things']['pq'] - pq_metric_pred.compute()['things']['pq']
            log_text += ' | {}: {:.6f}'.format(f"{name} pq_things_gain", pq_things_gain)
            metrics_ret['val/pq_things_gain'] = pq_things_gain.item()
            
            # Map metric
            for map_m, suffix in zip([map_metric, map_metric_pred], ['','_pred']):
                map_result = map_m.compute()
                for metric in [m for m in map_result if m in ['map', 'map_50', 'map_75']]:
                    log_text += ' | {}: {:.6f}'.format(f"{name} {metric}{suffix}", map_result[metric])
                    metrics_ret[f'val/{metric}_{suffix}'] = map_result[metric].item()
            

            for iou, suffix in zip([iou_metric, iou_metric_pred], ['','_pred']):   
                iou_total = iou.compute()
                log_text += ' | {}: {:.6f}'.format(f"{name} iou{suffix}", iou_total)
                metrics_ret[f'val/iou{suffix}'] = iou_total.item()

            for pq, suffix in zip([pq_metric, pq_metric_pred], ['','_pred']):
                pq_result = pq.compute()
                for group in pq_result:
                    for metric in [m for m in pq_result[group] if m != 'n']:
                        log_text += ' | {}: {:.6f}'.format(f"{name} {metric}_{group}{suffix}", pq_result[group][metric])
                        metrics_ret[f'val/{metric}_{group}{suffix}'] = pq_result[group][metric].item()

        log.info(log_text)
        # log metrics in tensorboard
        for k,v in metrics_ret.items():
            self.writer.add_scalar(k, v, self.epoch)

        return metrics_ret

    def validate(self, epoch=0):
        self.pipeline.eval()
        
        log.info("Beginning validation...")
        
        # Sample traing set and train clustering
        if 'train_clustering' in  dir(self.pipeline.nef) and epoch >= self.inst_epoch_start:
            with torch.no_grad():
                log.info(f"Training clustering with {self.num_clustering_samples} samples...")
                # Sample num_clustering_samples random rays from all training instance masks
                inst_gt = [k for k in self.dataset.data if 'instance' in k][0]
                rays_key = 'base_rays' if isinstance(self.pipeline, BAPipeline) is not None else 'rays'
                data = {inst_gt:self.dataset.data[inst_gt].squeeze(),
                        'rays':self.dataset.data[rays_key]}
                if rays_key == 'base_rays':
                    data['rays'] = Rays.stack([data['rays']]*data[inst_gt].shape[0])
                data = SampleRays(self.num_clustering_samples)(data)
                rays = data['rays'].reshape(-1,3).to('cuda')

                cam_ids = data['cameras_ts'] if 'cameras_ts' in data else torch.arange(data[inst_gt].shape[0])
                
                # Render instance embeddings
                rb = self.batch_render(self.pipeline,
                                       rays,
                                       channels=["inst_embedding"],
                                       cam_ids=cam_ids)
                inst_embed = rb.inst_embedding.reshape(*data[inst_gt].shape, rb.inst_embedding.shape[-1])
                self.pipeline.nef.train_clustering(F.normalize(inst_embed, dim=-1), data[inst_gt].cuda())

        mip = self.val_mip
        if epoch >= self.num_epochs and not self.low_res_val:
            mip = 0
        
        data = self.dataset.get_images(split="val", mip=mip)
        img_shape = data["imgs"][0].shape
        log.info(f"Loaded validation dataset with {len(data['imgs'])} images at resolution {img_shape[0]}x{img_shape[1]}")

        self.valid_log_dir = os.path.join(self.log_dir, "val",  f"epoch_{epoch}")
        log.info(f"Saving validation result to {self.valid_log_dir}")
        if not os.path.exists(self.valid_log_dir):
            os.makedirs(self.valid_log_dir)

        metrics_dict = self.evaluate_metrics(epoch, data)

        metrics_dict.update({'epoch': epoch,
                            'stamp' : self.log_dir.split('/')[-1],
                            'exp_name' : self.log_dir.split('/')[-2]})
        
        self.log_dict.update(metrics_dict)
        
        # write/append data frame to CSV file
        df = pd.DataFrame([metrics_dict])
        csv_path = os.path.join(self.log_dir, "metrics.csv")
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode='a', index=False, header=False)