import torch

from torch import nn

from wisp.models.nefs import BaseNeuralField
from wisp.tracers.base_tracer import BaseTracer
from wisp.models.pipeline import Pipeline
from wisp.core.rays import Rays
from kaolin.render.camera import Camera
from utils.camera import lie, hom_pose
# from utils.pose_vis import CameraPoseVisualizer


class PoseLie(nn.Module):
    def __init__(self, cameras: Camera):
        super().__init__()
        
        # so3 rotation
        rot = lie.SO3_to_so3(cameras.extrinsics.R)
        self.register_parameter('r', nn.Parameter(rot))
        self.register_parameter('t', nn.Parameter(cameras.extrinsics.t))

def cat_kaolin_cameras(cameras: 'Camera | list[Camera] | dict[str,Camera]'):
    if isinstance(cameras, dict):
        kaolin_cameras = Camera.cat(cameras.values())
    elif isinstance(cameras, (tuple,list)):
        kaolin_cameras = Camera.cat(cameras)
    elif isinstance(cameras, Camera):
        assert cameras.extrinsics.parameters().shape[0] > 1, (
        'Tried to create a camera database module with a Camera obejet with a signle camera extrinsincs',
        'but needs to have more than one')
        kaolin_cameras = cameras
    else:
        raise NotImplementedError('cameras constructor argument must be one of the folowing: Camera | list[Camera] | dict[str,Camera]',
                                    f'but{type(cameras)} was given.')

    return kaolin_cameras

class BAPipelineLie(Pipeline):
    """Bundle adjustment pipeline class

    Pipelines adds a pose database nn.Module in addition to a NeF and tracer

    """
    
    def __init__(self, nef: BaseNeuralField, cameras: 'Camera | list[Camera] | dict[str,Camera]',
                 tracer: BaseTracer = None, anchor_first_frame: bool =False):
        """Initialize the Pipeline.

        Args:
            nef (nn.Module): Neural fields module.
            cam_db (CamDatabase): Cam database for implicit bundle adjustment optimization.
            tracer (nn.Module or None): Forward map module.
        """
        super().__init__(nef, tracer)

        self.kaolin_cameras = cat_kaolin_cameras(cameras)
        
        self.camera_near = self.kaolin_cameras.near
        self.camera_far = self.kaolin_cameras.far
        self.cameras = PoseLie(self.kaolin_cameras)

        self.anchor_first_frame = anchor_first_frame
        
        # self.visualizer = CameraPoseVisualizer([-0.4, 0.4], [-0.4, 0.4], [-0.4, 0.4], self.cameras.t.shape[0])
        # cams = self.kaolin_cameras.extrinsics.view_matrix()
        # self.visualizer.extrinsic2pyramid_batch(cams.detach().cpu().numpy(), focal_len_scaled=0.1)

        
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.cameras = self.cameras.to(*args, **kwargs)

        if self.anchor_first_frame:
            r,t = self.cameras.parameters()
            mask_r = torch.ones_like(r)
            mask_r[0] = 0.0
            mask_t = torch.ones_like(t)
            mask_t[0] = 0.0
            r.register_hook(lambda grad: grad * mask_r)
            t.register_hook(lambda grad: grad * mask_t)

        return self

    def forward(self, *args, cam_ids=None, **kwargs):
        """ Transform base rays if cam_ids are passed requested camera poses
        and trace if a tracer is available
        """
        
        # transform base rays acocording to frames extrinsics
        if isinstance(cam_ids, torch.Tensor):
            num_cameras = cam_ids.shape[0]  # C - number of cameras
            o, d = (kwargs['rays'].origins, kwargs['rays'].dirs)
            o = o.reshape(num_cameras, -1, 3)[..., None]    # Expand as (C, B, 3, 1)
            d = d.reshape(num_cameras, -1, 3)[..., None]    # Expand as (C, B, 3, 1)
            batch_size = d.shape[1] # B - number of vectors
            # Rotate Directions
            R = lie.so3_to_SO3(self.cameras.r)
            t = self.cameras.t
            R = R[cam_ids, None].expand(num_cameras, batch_size, 3, 3)   # Expand as (C, B, 3, 3)
            R_T = R.transpose(2, 3)     # Transforms orientation from camera to world
            rays_dir = (R_T @ d).squeeze(-1)   # Inverse rotation is transposition: R^(-1) = R^T
            # transform origins
            t = t[cam_ids, None].expand(num_cameras, batch_size, 3, 1)   # Expand as (C, B, 3, 1)
            rays_orig = (R_T @ (o - t)).squeeze(-1)
            
            rays_dir = rays_dir / torch.linalg.norm(rays_dir, dim=-1, keepdim=True)
            kwargs['rays'] = Rays(origins=rays_orig.type(torch.float32), dirs=rays_dir.type(torch.float32),
                                  dist_min=self.camera_near, dist_max=self.camera_far).reshape(-1,3)
        
        if self.tracer is not None:
            return self.tracer(self.nef, *args, **kwargs)
        else:
            return self.nef(*args, **kwargs)

    def update_cameras(self):
        self.kaolin_cameras.extrinsics.update(hom_pose( lie.so3_to_SO3(self.cameras.r),
                                                        self.cameras.t.squeeze(-1)))