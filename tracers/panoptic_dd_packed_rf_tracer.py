# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
import kaolin.render.spc as spc_render
from wisp.core import RenderBuffer
from wisp.utils import PsDebugger, PerfTimer
from wisp.tracers import PackedRFTracer

from loss.regularizers import sigma_sparsity_loss


class PanopticDDensityPackedRFTracer(PackedRFTracer):
    """Tracer class for sparse (packed) radiance fields.

    This tracer class expects the use of a feature grid that has a BLAS (i.e. inherits the BLASGrid
    class).
    """
    def __init__(self, ray_sparcity_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.render_channels = {'depth', 'alpha', 'hit'}
        self.base_channels = {'rgb', 'density'}
        self.panoptic_channels = {'delta_density', 'panoptic_density',
                                  'semantics', 'inst_embedding'}
        
        self.ray_sparcity_reg = ray_sparcity_reg
    
    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.
        
        Returns:
            (set): Set of channel strings.
        """
        return {'depth', 'hit', 'rgb', 'alpha',
                'delta_density', 'panoptic_density',
                'semantics', 'inst_embedding'}
    
    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.
        
        Returns:
            (set): Set of channel strings.
        """
        return {'rgb', 'density', 'panoptic_density'}

    def trace(self, nef, channels, extra_channels, rays,
              lod_idx=None, raymarch_type='voxel', num_steps=64, step_size=1.0, bg_color='white',
              stage='val'):
        """Trace the rays against the neural field.

        Args:
            nef (nn.Module): A neural field that uses a grid class.
            channels (set): The set of requested channels. The trace method can return channels that 
                            were not requested since those channels often had to be computed anyways.
            extra_channels (set): If there are any extra channels requested, this tracer will by default
                                  perform volumetric integration on those channels.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3]
            lod_idx (int): LOD index to render at. 
            raymarch_type (str): The type of raymarching algorithm to use. Currently we support:
                                 voxel: Finds num_steps # of samples per intersected voxel
                                 ray: Finds num_steps # of samples per ray, and filters them by intersected samples
            num_steps (int): The number of steps to use for the sampling.
            step_size (float): The step size between samples. Currently unused, but will be used for a new
                               sampling method in the future.
            bg_color (str): The background color to use. TODO(ttakikawa): Might be able to simplify / remove

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the render.
        """
        #TODO(ttakikawa): Use a more robust method
        assert nef.grid is not None, "this tracer requires a grid"

        timer = PerfTimer(activate=False, show_memory=False)
        N = rays.origins.shape[0]
        
        if lod_idx is None:
            lod_idx = nef.grid.num_lods - 1

        # This tracer will attempt to use the highest level of detail for the ray sampling.
        ridx, pidx, samples, depths, deltas, boundary = nef.grid.raymarch(rays, 
                level=nef.grid.active_lods[lod_idx], num_samples=num_steps, raymarch_type=raymarch_type)
        
        timer.check("Raymarch")

        # Get the indices of the ray tensor which correspond to hits
        ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]

        # Compute the channels for each ray and their samples
        hit_ray_d = rays.dirs.index_select(0, ridx)
        
        outputs = {}

        # always compute RGB density
        sample_channels = set(channels - self.render_channels)
        sample_channels.update(['density'])
        if any(c in self.panoptic_channels for c in channels):
            sample_channels.update(['panoptic_density'])
        out_feats = nef(coords=samples, ray_d=hit_ray_d, pidx=pidx, lod_idx=lod_idx, channels=sample_channels)
        timer.check("Channels forward")

        if self.ray_sparcity_reg > 0.0 and stage == 'train':
            all_rays_loss = sigma_sparsity_loss(out_feats['density'].squeeze())
            ray_wise_loss = torch.scatter_add(torch.zeros_like(rays.origins[:,0]), 0, ridx, all_rays_loss)            
            outputs['ray_sparcity_loss'] = ray_wise_loss.mean() * self.ray_sparcity_reg

        del ridx#, rays
        # Compute optical thickness
        tau = out_feats['density'].reshape(-1, 1) * deltas
        _ , transmittance = spc_render.exponential_integration(torch.tensor([]).to(nef.device), tau, boundary, exclusive=True)
        
        # Compute alpha
        alpha = spc_render.sum_reduce(transmittance, boundary)
        timer.check("Sum Reduce")
        out_alpha = torch.zeros(N, 1, device=nef.device)
        out_alpha[ridx_hit.long()] = alpha
        outputs['alpha'] = out_alpha
        # Compute hit
        hit = torch.zeros(N, device=nef.device).bool()
        hit[ridx_hit.long()] = alpha[...,0] > 0.0
        outputs['hit'] = hit

        if 'panoptic_density' in out_feats:
            panop_boundary = boundary.detach()
            panop_tau = out_feats['panoptic_density'].reshape(-1, 1) * deltas.detach()
            # if 'panoptic_density' not in channels:
            #     del out_feats['panoptic_density']

            _ , panop_transmittance = spc_render.exponential_integration(torch.tensor([]).to(nef.device), panop_tau, panop_boundary, exclusive=True) 

            # Compute alpha
            panop_alpha = spc_render.sum_reduce(panop_transmittance, panop_boundary)
            timer.check("Sum Reduce")
        
        #del deltas

        if 'rgb' in channels:
            ray_colors = spc_render.sum_reduce(out_feats['rgb'].reshape(-1, 3).contiguous() * transmittance, boundary)
            # Populate the background
            if bg_color == 'white':
                rgb = torch.ones(N, 3, device=nef.device)
                color = (1.0-alpha) + alpha * ray_colors
            else:
                rgb = torch.zeros(N, 3, device=nef.device)
                color = alpha * ray_colors
            rgb[ridx_hit.long()] = color
            outputs['rgb'] = rgb

        if "depth" in channels:
            ray_depth = spc_render.sum_reduce(depths.reshape(-1, 1) * transmittance, boundary)
            depth = torch.zeros(N, 1, device=ray_depth.device)
            depth[ridx_hit.long(), :] = ray_depth
            outputs['depth'] = depth

        for panop_channel in [c for c in channels if c in self.panoptic_channels]:
            panop_ridx_hit = ridx_hit.detach()
            outputs[panop_channel] = self._integrate_features(out_feats[panop_channel],
                                                              panop_alpha, panop_transmittance,
                                                              panop_boundary, panop_ridx_hit, N)
        
        extra_outputs = {}
        for channel in extra_channels:
            feats = nef(coords=samples,
                        ray_d=hit_ray_d,
                        pidx=pidx,
                        lod_idx=lod_idx,
                        channels=channel)

            extra_outputs[channel] = self._integrate_features(feats, alpha, transmittance, boundary, ridx_hit, N)

        # TODO: Might need to return panoptic_hit and panoptic_alpha as well for proper rendering
        return RenderBuffer(**outputs, **extra_outputs)

    def _integrate_features(self, feats, alpha, transmittance, boundary, ridx_hit, N):
        num_channels = feats.shape[-1]

        ray_feats = spc_render.sum_reduce(transmittance * feats.view(-1, num_channels).contiguous(), boundary.contiguous())
        
        composited_feats = alpha * ray_feats
        out_feats = torch.zeros(N, num_channels, device=feats.device)
        out_feats[ridx_hit.long()] = composited_feats
        return out_feats
