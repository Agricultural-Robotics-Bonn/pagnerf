import sys
import numpy as np
import tqdm
import logging as log

import torch

from wisp.ops.raygen import generate_pinhole_rays, generate_centered_pixel_coords
from wisp.core import Rays
from kaolin.render.camera import Camera, CameraExtrinsics, PinholeIntrinsics
from wisp.core import Rays


from utils.outlier_rejection import rays_to_3d_points


@torch.no_grad()
def offline_nef_batch_render(nef, coords, ray_d, channels=[], render_batch=20000):

    render = lambda coords, ray_d: nef(coords=coords[:,None], ray_d=ray_d, channels=channels)
    batched_coords = coords.split(render_batch)
    batched_dirs = ray_d.split(render_batch) if ray_d is not None else [None]*render_batch
    
    for i, (coord_pack,dir_pack) in enumerate(zip(batched_coords, batched_dirs)):
        # Render and make a list if rendering a single channel
        render_batch = render(coord_pack,dir_pack)
        render_batch = [render_batch] if not isinstance(render_batch, list) else render_batch
        # batch render
        if i == 0:
            rb = {c:r for c,r in zip(channels,render_batch)}
        else:
            rb = {c:torch.cat((rb[c],r), dim=0) for c,r in zip(channels,render_batch)}
    return rb

# TODO (csmitt): renderer copied from validation step (esto no se hace...)
@torch.no_grad()
def offline_pipeline_batch_render(pipeline, rays, channels=[], render_batch=20000):
    render = lambda ray_pack: pipeline(rays=ray_pack, lod_idx=None, channels=channels)
    for i, ray_pack in enumerate(rays.split(render_batch)):
        if i == 0:
            rb = render(ray_pack)
        else:
            rb += render(ray_pack)
    return rb

@torch.no_grad()
def get_dense_occupied_points(nef, blas_level, min_density=None, limits=[]):
    #
    # limits [tesor]: [2,3] -> [[min,max],[x,y,z]]

    if min_density is None:
        min_density = ((0.01 * 512)/np.sqrt(3))
    res = 2.0**blas_level
    
    # Dense grid points [res^3,3] in normalized cube [-1,1]
    points = torch.stack(torch.meshgrid(*[torch.arange(res).cuda()]*3),-1).reshape(-1,3) / res * 2.0 - 1.0
    
    if limits.numel() > 0:
        x_mask = torch.logical_and(points[:,0] > limits[0,0], points[:,0] < limits[1,0])
        y_mask = torch.logical_and(points[:,1] > limits[0,1], points[:,1] < limits[1,1])
        z_mask = torch.logical_and(points[:,2] > limits[0,2], points[:,2] < limits[1,2])
        points_mask = torch.logical_and( x_mask, torch.logical_and(y_mask, z_mask))
        points = points[points_mask]

    # Add wiggle to points in normalized coordinates
    # TODO (csmitt): parametrize how much wiggle
    noise = torch.rand(*points.shape, device=points.device) / res *2.0 - 1.0
    samples = points + noise

    # (csmitt): In the original prune code for NeFs that use ray_d despite density ins independant of it
    # might be needed for some NeFs 
    # sample_views = torch.FloatTensor(sample_unif_sphere(samples.shape[0])).to(points.device)

    # Query grid density
    density = offline_nef_batch_render(nef, samples, None, channels=["density"])['density']
    # Keep occupied points
    mask = density[:, 0, 0] > min_density
    
    return points[mask]

@torch.no_grad()
def render_points_at_depth(pipeline, mip, channels=[],limits=[]):
    resize_factor = 2**mip
    
    cameras = pipeline.cameras
    h,w = (int(cameras.intrinsics.height // resize_factor),
           int(cameras.intrinsics.width  // resize_factor))

    intrinsic_params = cameras.intrinsics.parameters()[0] / resize_factor
    base_intrinsics = PinholeIntrinsics(w, h, intrinsic_params,
                                        near=cameras.intrinsics.near,
                                        far=cameras.intrinsics.far)
    base_cameras = Camera(CameraExtrinsics.from_view_matrix(torch.eye(4).cuda()),
                         base_intrinsics)

    
    ray_grid = generate_centered_pixel_coords(w, h, w, h, device='cuda')         
    base_rays = Rays.stack([generate_pinhole_rays(base_cameras.to('cuda'), ray_grid)] * len(cameras))


    cam_ids = list(pipeline.cam_id_to_idx.keys())
    rays = pipeline.transform_rays(base_rays, cam_ids)
    
    render_channels = ['depth', 'density','rgb'] + channels
    rb = offline_pipeline_batch_render(pipeline, rays=rays, channels=render_channels)

    points_3d = rays_to_3d_points(base_rays.reshape(-1,3), rb.depth, cameras)
    
    inst_embedding = torch.argmax(rb.inst_embedding, dim=-1)
    points_mask = rb.density[:,0] > 40
    points_mask = torch.logical_and(points_mask, rb.alpha[:,0] > 0.9)
    points_mask = torch.logical_and(points_mask, rb.hit)
    points_mask = torch.logical_and(points_mask, rb.depth[:,0] < 0.8)
    points_mask = torch.logical_and(points_mask, rb.depth[:,0] > 0.6)
    
    # inst_mask = torch.logical_and(inst_embedding != 0)
    
    rendered_points = {'points': points_3d[points_mask].detach().cpu(),
                       'inst_embedding': inst_embedding[points_mask].detach().cpu(),
                       'color':rb.rgb[points_mask].detach().cpu()}

    # rendered_points.update({c:rb[c] for c in channels})

    return rendered_points



# post processing methods for the following channels: ["density", "rgb","semantics", "inst_embedding"]

def post_process_density(density):
    return density

def post_process_rgb(rgb):
    return rgb

def post_process_semantics(semantics):
    return torch.argmax(semantics, dim=-1)

def post_process_inst_embedding(inst_embedding):
    return torch.argmax(inst_embedding, dim=-1)
    
    
@torch.no_grad()
def generate_pc_map(nef, blas_level, name='nerf_pc', min_density=None, limits=[], channels=[], ray_d=None):
    
    log.info('Computing occupied points...')
    points = get_dense_occupied_points(nef, blas_level, min_density, limits)

    log.info('Rendering points...')
    rb = offline_nef_batch_render(nef, points, None, channels=channels)

    # data = [{'name': name,
    #          'points': points.detach().cpu(),
    #         }]
    
    # for c in channels:
    #     post_process_fn = getattr(sys.modules[__name__], f'post_process_{c}')
    #     data[0][c] = post_process_fn(rb[c].detach().cpu())

    inst_points = torch.argmax(rb['inst_embedding'], dim=-1) != 0

    data = [{'name': name,
             'points': points[inst_points].detach().cpu(),
             'instances': torch.argmax(rb['inst_embedding'][inst_points], dim=-1).detach().cpu(),
            }]
    

    
    return data

@torch.no_grad()
def generate_pc_map_from_views(pipeline, name='nerf_pc', limits=[], channels=[], mip=0):
    
    log.info('Computing occupied points...')
    rendered_points = render_points_at_depth(pipeline, mip, channels,limits)
    rendered_points['name'] = name

    # for c in channels:
    #     post_process_fn = getattr(sys.modules[__name__], f'post_process_{c}')
    #     rendered_points[c] = post_process_fn(rendered_points[c].detach().cpu())

    
    return [rendered_points]