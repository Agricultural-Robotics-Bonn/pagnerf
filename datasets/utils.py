import numpy as np
from plyfile import PlyData
import torch



def get_scale_from_ply_mesh(ply_path='', scaling_option='snap_to_bottom'):
  ''' Estimate scale and offset from a ply mesh catesian bounds
  '''
  with open(ply_path, 'rb') as f:
    vertices = PlyData.read(f).elements[0].data
    bounds = np.array([[np.array(vertices[k].min()), np.array(vertices[k]).max()] for k in ['x','y','z']])
    lengths = np.abs(bounds[:,1] - bounds[:,0])
    centers = (bounds[:,1] + bounds[:,0]) / 2.
  
    if scaling_option == 'largest':
      # Compute scale according to the largest XYZ bound and shrink it 2%
      scale = 0.98 * 2. / lengths[np.argmax(lengths)]
      # Center the model at the unit cube
      offset = -centers * scale
    
    elif scaling_option == 'snap_to_bottom':
      # Compute scale according to the largest XY bound
      scale = 2. / lengths[np.argmax(lengths[:2])]  
      # Center the model at the unit cube
      offset = -centers * scale
      # Make the bottom of the model touch the bottom plane of the unit cube 
      offset[2] = -bounds[2,0] * scale - 1
    
    else:
      raise NotImplementedError(f'Unimplememnted model scaling option: {scaling_option}')
    
  return scale, offset

def transform_cv_to_gl_poses(poses):
  device = poses.device
  
  # rotate poses 180deg around x axis
  cv_to_gl = torch.eye(4).to(device)
  cv_to_gl[[[1,2],[1,2]]] = -1

  return poses @ cv_to_gl[None]