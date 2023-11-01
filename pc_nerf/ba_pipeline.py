import torch

from wisp.models.nefs import BaseNeuralField
from wisp.tracers.base_tracer import BaseTracer
from wisp.models.pipeline import Pipeline
from wisp.core.rays import Rays
from kaolin.render.camera import Camera


class BAPipeline(Pipeline):
    """Bundle adjustment pipeline class

    Pipelines adds a pose database nn.Module in addition to a NeF and tracer

    """
    
    def __init__(self, nef: BaseNeuralField, cameras: 'Camera | list[Camera] | dict[str,Camera]',
                 tracer: BaseTracer = None,
                 anchor_frame_idxs: 'list[int]' = [],
                 pose_opt_only_frame_idxs: 'list[int]' = []):
        """Initialize the Pipeline.

        Args:
            nef (nn.Module): Neural fields module.
            cam_db (CamDatabase): Cam database for implicit bundle adjustment optimization.
            tracer (nn.Module or None): Forward map module.
        """
        super().__init__(nef, tracer)

        if isinstance(cameras, dict):
            self.cam_id_to_idx = {cam_id:idx for idx,cam_id in enumerate(cameras.keys())}
            self.cameras = Camera.cat(cameras.values())
        elif isinstance(cameras, (tuple,list)):
            self.cameras = Camera.cat(cameras)
        elif isinstance(cameras, Camera):
            assert cameras.extrinsics.parameters().shape[0] > 1, (
            'Tried to create a camera database module with a Camera obejet with a signle camera extrinsincs',
            'but needs to have more than one')
            self.cameras = cameras
        else:
            raise NotImplementedError('cameras constructor argument must be one of the folowing: Camera | list[Camera] | dict[str,Camera]',
                                      f'but{type(cameras)} was given.')
        
        self.cameras.extrinsics.switch_backend('matrix_6dof_rotation')
        
        self.anchor_frame_idxs = anchor_frame_idxs
        self.pose_opt_only_frame_idxs = pose_opt_only_frame_idxs

        self.cameras.extrinsics._backend.params = torch.nn.Parameter(self.cameras.extrinsics.parameters())

        self.register_parameter(name='camera_extrinsics', param=self.cameras.extrinsics.parameters())
        
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.cameras.intrinsics = self.cameras.intrinsics.to(*args, **kwargs)
        if isinstance(self.anchor_frame_idxs, (tuple,list)) and len(self.anchor_frame_idxs) > 0:
            params = self.cameras.extrinsics.parameters()
            grad_mask = torch.ones_like(params)
            grad_mask[self.anchor_frame_idxs] = 0.0
            params.register_hook(lambda grad: grad * grad_mask)

        return self

    def forward(self, *args, cam_ids=None, **kwargs):
        """ Transform base rays if cam_ids are passed requested camera poses
        and trace if a tracer is available
        """
        # transform base rays acocording to frames extrinsics
        if isinstance(cam_ids, (tuple,list,torch.Tensor)):
            kwargs['rays'] = self.transform_rays(kwargs['rays'], cam_ids)

            
        if self.tracer is not None:
            return self.tracer(self.nef, *args, **kwargs)
        else:
            return self.nef(*args, **kwargs)
    
    def get_cameras_from_ids(self, cam_ids):
        assert isinstance(cam_ids, (tuple,list,torch.Tensor))
        if isinstance(cam_ids, (tuple,list)):
            cam_ids = torch.tensor([self.cam_id_to_idx[id] for id in cam_ids], dtype=torch.long)
        assert cam_ids.nelement() > 0
        return self.cameras[cam_ids]

    def transform_rays(self, base_rays, cam_ids):
        cameras = self.get_cameras_from_ids(cam_ids)
        
        batch_rays = base_rays.reshape(len(cameras), -1, 3)
        rays_orig, rays_dir = cameras.extrinsics.inv_transform_rays(batch_rays.origins, batch_rays.dirs)
        rays_dir = rays_dir / torch.linalg.norm(rays_dir, dim=-1, keepdim=True)
        return Rays(origins=rays_orig.type(torch.float32), dirs=rays_dir.type(torch.float32),
                                dist_min=cameras.near, dist_max=cameras.far).reshape(-1,3)
