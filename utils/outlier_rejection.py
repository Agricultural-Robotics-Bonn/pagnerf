import torch

from kaolin.render.camera import Camera
from wisp.core import Rays

# Computes available IDs for a given X position for each instance
# and assigns a high cost to those IDs that are not available
def add_position_id_range_cost(cost_matrix, inst_centers,
                               frame_min_length = 0.3,
                               max_num_inst_at_x = 30,
                               id_margin_at_frame_length = 30):
    # cost_matrix: [N,M] (N: detection masks, M: predicted masks)
    # current_inst_centers: [M,4] torch tensor ([N,[x,y,z,ID]])
    # frame_min_length: minimum length of the frame aprox. at the closest target depth in meters
    # max_num_inst_at_x: maximum number of instances at a given x position
    # id_margin_at_frame_length: margin in number of IDs at the frame length
    #  
    # return: cost_matrix [N,M]
    
    # get number of IDs
    num_ids = cost_matrix.shape[1]
    # Available IDs slope
    m = (max_num_inst_at_x + id_margin_at_frame_length )/ frame_min_length
    # x limit to wraparround the IDs
    x_limit = (num_ids - id_margin_at_frame_length) / m
    # lambda function to compute the lower bound of available IDs for each instance x position
    available_ids_0 = lambda x: torch.clamp(m * (x % x_limit), 0, num_ids - 1).type(torch.long)
    # lambda function to compute both bounds of available IDs for each instance x position
    available_ids = lambda x: (available_ids_0(x), torch.clamp(available_ids_0(x) + id_margin_at_frame_length, 0, num_ids - 1))
    # Lambda to remap x position from [1,-1] to [0,1]
    x_remap = lambda x: (-x + 1) / 2

    # Compute the available IDs for each instance x position [2,N]
    available_ids_x = available_ids(x_remap(inst_centers[:,0]))
    # Compute the mask of available IDs for each instance x position [N,M]

    # # TODO (csmitt): test y id rejection
    # # Lambda to remap x position from [1,-1] to [0,1]
    # y_remap = lambda y: (y * 0.8 + 1) / 2
    # # disperse available IDs along y axis
    # max_num_inst_at_y = 8
    # available_ids_y = lambda y, ids_x: (torch.max(ids_x[0], ids_x[0] + (y*max_num_inst_at_x) - (max_num_inst_at_y/2)).type(torch.int),
    #                                     torch.min(ids_x[1], ids_x[0] + (y*max_num_inst_at_x) + (max_num_inst_at_y/2)).type(torch.int))
    # available_ids_x = available_ids_y(y_remap(inst_centers[:,1]), available_ids_x)

    available_ids_mask = torch.logical_and(available_ids_x[0][:,None] <= torch.arange(num_ids)[None,:].to(inst_centers.device),
                                            torch.arange(num_ids)[None,:].to(inst_centers.device) <= available_ids_x[1][:,None])
    # Set the cost of the unavailable IDs to a very high value
    cost_matrix[~available_ids_mask.cpu()] = 10000

    return cost_matrix


# Takes a tensor of 3d points [N,[x,y,z,ID]] and computes the average positions of the points with the same ID.
# This is done in an efficient vectorized manner.
def centers_from_3d_points_with_ids(points):
    # points: [N,[x,y,z,ID]]
    # return: [I,[x,y,z,ID]]
    #         K: number of unique IDs
    #         [K,[x,y,z,ID]]: centers of the points with the same ID

    # get unique center ids [I]
    ids = points[:,3].unique()
    # same ID mask [I,N]
    same_id_mask = ids[:,None] == points[None,:,3]
    # Compute mean centers of all points with the same ID [I,3]
    centers = (same_id_mask.type(torch.float) @ points[:,:3]) / same_id_mask.sum(dim=1)[:,None]
    # add ID to centers [I,4]
    centers = torch.cat((centers, ids[:,None]), dim=1)
    # return centers [I,4]
    return centers

# Unproject 2d points to 3d points using the depth map and the camera extrinisics
def rays_to_3d_points(rays: Rays, depth: torch.Tensor, cameras: Camera):
    # rays: [N] composed of [origins: [N,3], dirs: [N,3]]
    # depth: [N] with positive z values
    # cameras: Camera [M]
    #
    # return: [N,3] 3d points in world coordinates
    #
    # camera and ray convention:
    #    Y
    #    ^
    #    |
    #    |---------> X
    #   /
    # Z
    #

    # Project image depth onto rays 
    rays_hit_cam = depth.squeeze() # * (rays.dirs @ torch.tensor([0,0,-1.]).to(depth.device))
    # Unproject rays to 3d points in camera coordinates
    points_cam = (rays.dirs * rays_hit_cam[:,None]).reshape(len(cameras),-1,3)
    # Transform points to world coordinates
    cam_origins, cam_pos = cameras.extrinsics.inv_transform_rays(rays.origins.reshape(len(cameras),-1,3), points_cam)
    # return 3d points in world coordinates
    return (cam_origins + cam_pos).reshape(-1,3)
    
def instance_rays_to_3d_centers(rays: Rays, depth: torch.Tensor, cameras: Camera, inst_ids: torch.Tensor):
    # rays: [N] composed of [origins: [N,3], dirs: [N,3]]
    # depth: [N] with positive z values
    # cameras: Camera [M]
    # inst_ids: [N] with instance ids
    #
    # return: [K,4] 3d centers in world coordinates
    #         K: number of unique IDs
    #         [K,[x,y,z,ID]]: centers of the points with the same ID
    #
    
    # Unproject rays to 3d points in world coordinates
    points = rays_to_3d_points(rays, depth, cameras)
    # Add instance ids to points
    id_points = torch.cat((points, inst_ids[:,None]), dim=1)
    # Compute centers of the points with the same ID
    return centers_from_3d_points_with_ids(id_points)

def center_of_mass(mask):
    # Create a grid of coordinates
    h, w = mask.shape[-2:]
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
    grid_y = grid_y.float().to(mask.device)
    grid_x = grid_x.float().to(mask.device)

    # Compute the mass (sum of all mask values)
    mass = mask.sum(dim=(-2, -1))

    # Compute the center of mass coordinates
    center_y = torch.sum(grid_y * mask, dim=(-2, -1)) / mass
    center_x = torch.sum(grid_x * mask, dim=(-2, -1)) / mass

    return center_y, center_x, grid_y, grid_x

def mask_center_of_mass_outlier_rejection(mask, std_threshold=2.0):
    center_y, center_x, grid_y, grid_x = center_of_mass(mask)
    distance_y = grid_y - center_y[:, None, None]
    distance_x = grid_x - center_x[:, None, None]
    distance = torch.sqrt(distance_y ** 2 + distance_x ** 2)

    # Calculate the mean and standard deviation of the squared distances
    nan_distance = distance.clone()
    nan_distance[~mask.type(torch.bool)] = torch.nan

    mean_distance = torch.nanmean(nan_distance, dim=(-1,-2), keepdim=True)
    std_distance = torch.sqrt(torch.nanmean(torch.pow(nan_distance - mean_distance, 2), dim=(-1,-2), keepdim=True))

    # Create a mask of points that are within the desired range
    mask_within_range = distance <= (mean_distance + std_threshold * std_distance)

    # Apply the mask to the original mask tensor
    filtered_mask = mask.clone()
    filtered_mask[~mask_within_range] = 0

    return filtered_mask