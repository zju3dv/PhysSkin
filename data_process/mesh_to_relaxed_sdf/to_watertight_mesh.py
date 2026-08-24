import math
import numpy as np
from scipy.spatial.transform import Rotation
import cubvh
import torch
import trimesh
import cumcubes
import os
import sys
from omegaconf import OmegaConf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_dense_grid_points(
    bbox_min = np.array((-1.05, -1.05, -1.05)),
    bbox_max= np.array((1.05, 1.05, 1.05)),
    resolution = 512,
    indexing = "ij"
):
    """Generate a dense grid of 3D points within the specified bounding box."""
    length = bbox_max - bbox_min
    num_cells = resolution
    x = np.linspace(bbox_min[0], bbox_max[0], resolution + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    xyz = xyz.reshape(-1, 3)  # (resolution+1)^3 x 3
    grid_size = [resolution + 1, resolution + 1, resolution + 1]

    return xyz, grid_size

def get_rotation_matrix(angle, axis='y'):
    """Create a 4x4 rotation matrix for the given angle and axis."""
    matrix = np.identity(4)
    if hasattr(Rotation, "as_matrix"):  # scipy>=1.4.0
        matrix[:3, :3] = Rotation.from_euler(axis, angle).as_matrix()
    else:  # scipy<1.4.0
        matrix[:3, :3] = Rotation.from_euler(axis, angle).as_dcm()
    return matrix

def get_equidistant_camera_angles(count):
    """Generate equidistant camera angles using Fibonacci sphere distribution."""
    increment = math.pi * (3 - math.sqrt(5))
    for i in range(count):
        theta = math.asin(-1 + 2 * i / (count - 1))
        phi = ((i + 1) * increment) % (2 * math.pi)
        yield phi, theta

def get_camera_transform_looking_at_origin(rotation_y, rotation_x, camera_distance=2):
    """Create a camera transformation matrix looking at the origin."""
    camera_transform = np.identity(4)
    camera_transform[2, 3] = camera_distance
    camera_transform = np.matmul(get_rotation_matrix(rotation_x, axis='x'), camera_transform)
    camera_transform = np.matmul(get_rotation_matrix(rotation_y, axis='y'), camera_transform)
    return camera_transform

def get_camera_transforms(scan_count, bounding_radius):
    """Generate camera transformation matrices for multiple viewpoints."""
    camera_transforms = []
    for phi, theta in get_equidistant_camera_angles(scan_count):
        camera_transform = get_camera_transform_looking_at_origin(phi, theta, camera_distance=2 * bounding_radius)
        camera_transform = torch.from_numpy(camera_transform).cuda()
        camera_transforms.append(camera_transform)
    return camera_transforms

def is_inside_strict(grid_xyz, grid_udf, mesh_tracer, camera_transforms):
    """Determine inside points using strict visibility criteria.
    Points are considered inside only if they are occluded from all camera viewpoints."""
    # Initialize all points as inside, then exclude based on visibility
    inside_mask = torch.ones(grid_xyz.shape[0], dtype=torch.bool, device=grid_xyz.device)

    for camera_transform in camera_transforms:
        # Calculate ray origin (camera position) and direction
        ray_o = camera_transform[:3, 3].unsqueeze(0).repeat(grid_xyz.shape[0], 1)
        ray_d = grid_xyz - ray_o
        line_depth = torch.norm(ray_d, dim=-1, keepdim=True)
        ray_d = ray_d / (line_depth + 1e-6)  # Normalize direction

        # Get closest intersection depth of rays with mesh
        _, _, render_depth = mesh_tracer.ray_trace(ray_o, ray_d)
        # Determine if point is occluded (render depth < point distance)
        is_occluded = (render_depth < line_depth.squeeze(-1))

        # Update inside mask: keep only points occluded from all viewpoints
        inside_mask &= is_occluded

    grid_udf[inside_mask] *= 0
    return grid_udf

def is_inside_relax(grid_xyz, grid_udf, mesh_tracer, camera_transforms, threshold=10):
    """Determine inside points using relaxed visibility criteria.
    Points are considered inside if they are visible from fewer than threshold cameras."""
    # Initialize observation counter for each point
    observation_count = torch.zeros(grid_xyz.shape[0], dtype=torch.int, device=grid_xyz.device)
    # Track points that still need processing (those with observation count <= threshold)
    active_mask = torch.ones(grid_xyz.shape[0], dtype=torch.bool, device=grid_xyz.device)
    
    for camera_transform in camera_transforms:
        if not active_mask.any():
            break  # All points have been processed
            
        # Select currently active points
        current_indices = active_mask.nonzero().squeeze(1)
        current_xyz = grid_xyz[current_indices]
        
        # Calculate ray origin (camera position) and direction
        ray_o = camera_transform[:3, 3].unsqueeze(0).expand(current_xyz.size(0), 3)
        ray_d = current_xyz - ray_o
        line_depth = torch.norm(ray_d, dim=-1, keepdim=True)
        ray_d = ray_d / (line_depth + 1e-6)  # Normalize direction
        
        # Get closest intersection depth of rays with mesh
        _, _, render_depth = mesh_tracer.ray_trace(ray_o, ray_d)
        
        # Determine if point is occluded (render depth < point distance)
        is_occluded = (render_depth < line_depth.squeeze(-1))
        
        # Increment observation count for points that are not occluded (visible)
        visible_points = ~is_occluded
        global_visible_indices = current_indices[visible_points]
        observation_count[global_visible_indices] += 1
        
        # Update active mask: deactivate points that exceed the threshold
        exceeded_threshold = observation_count > threshold
        active_mask[exceeded_threshold] = False
    
    # Points are considered inside if they are visible from fewer than threshold cameras
    inside_mask = (observation_count <= threshold)
    
    # Set UDF to 0 for inside points
    grid_udf[inside_mask] *= 0
    return grid_udf

def main(cfg):
    """Main function to convert a mesh to a watertight version."""
    os.makedirs(cfg.output.save_dir, exist_ok=True)
    mesh_path = cfg.input.mesh_path
    mesh = trimesh.load(mesh_path, force='mesh')

    # Normalize mesh to [-1,1]
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) / 2
    scale = 2.0 / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    print("vertices min is: ", vertices.min(0))
    print("vertices min is: ", vertices.max(0))

    # Calculate grid points UDF (unsigned distance field)
    resolution = cfg.processing.resolution
    grid_xyz, grid_size = generate_dense_grid_points(resolution=resolution)
    grid_xyz = torch.FloatTensor(grid_xyz).cuda()
    f = cubvh.cuBVH(torch.as_tensor(vertices, dtype=torch.float32, device='cuda'), 
                   torch.as_tensor(mesh.faces, dtype=torch.float32, device='cuda'))
    grid_udf, _, _ = f.unsigned_distance(grid_xyz, return_uvw=False)
    
    # Process points in batches to manage memory usage
    batch_size = cfg.processing.points_per_batch
    n_batches = int(math.ceil(grid_xyz.shape[0] / batch_size))

    # Generate camera viewpoints and select inside/outside determination method
    camera_transforms = get_camera_transforms(scan_count=cfg.processing.num_camera, bounding_radius=1.25)
    inside_func = is_inside_strict if cfg.processing.use_strict else is_inside_relax

    # Determine inside points in batches
    sdfs = []
    for i_start in range(0, grid_xyz.shape[0], batch_size):
        i_end = min(i_start + batch_size, grid_xyz.shape[0])
        points = grid_xyz[i_start:i_end]
        points_udf = grid_udf[i_start:i_end]
        points_sdf = inside_func(points, points_udf, f, camera_transforms, threshold=cfg.processing.threshold)
        sdfs.append(points_sdf)

    sdfs = torch.cat(sdfs, dim=0)

    # Reshape SDF grid and extract mesh using marching cubes
    sdfs = sdfs.reshape(resolution+1, resolution+1, resolution+1)
    if not cfg.processing.use_dmc:
        v, f = cumcubes.marching_cubes(sdfs, 2/resolution)
        v = v/(resolution+2) * 2 - 1

        # Create mesh and extract largest component
        wt_mesh = trimesh.Trimesh(vertices=v.cpu().numpy(), faces=f.cpu().numpy())
    else:
        # Optional dependency used only when processing.use_dmc=true.
        from diso import DiffDMC

        eps = 2/resolution
        diffdmc = DiffDMC(dtype=torch.float32).cuda()
        vertices, faces = diffdmc(sdfs, isovalue=eps, normalize=False)
        bbox_min = np.array((-cfg.processing.scale, -cfg.processing.scale, -cfg.processing.scale))
        bbox_max= np.array((cfg.processing.scale, cfg.processing.scale, cfg.processing.scale))
        bbox_size = bbox_max - bbox_min
        vertices = (vertices + 1) / grid_size[0] * bbox_size[0] + bbox_min[0]
        wt_mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
    
    print("vertices min is: ", vertices.min(0))
    print("vertices min is: ", vertices.max(0))
    
    components = wt_mesh.split(only_watertight=False)
    bbox = []
    for c in components:
        bbmin = c.vertices.min(0)
        bbmax = c.vertices.max(0)
        bbox.append((bbmax - bbmin).max())
    max_component = np.argmax(bbox)
    wt_mesh = components[max_component]
    
    vertices = wt_mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) / 2
    scale = 1.0 / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    wt_mesh.vertices = vertices
    
    bbmin = wt_mesh.vertices.min(0)
    bbmax = wt_mesh.vertices.max(0)

    print("bbox min:", bbmin)
    print("bbox max:", bbmax)
    print("extent:", bbmax - bbmin)
    print("max abs:", np.abs([bbmin, bbmax]).max())

    # Save the watertight mesh
    remesh_path = os.path.join(cfg.output.save_dir, f"{cfg.input.mesh_id}.obj")
    wt_mesh.export(remesh_path)
    logger.info(f"Watertight mesh saved to {remesh_path}")
    return True

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "configs", "remesh.yaml")
    cfg = OmegaConf.load(config_path)
    cli_cfg = OmegaConf.from_dotlist(sys.argv[1:])
    print("cli_cfg is: ", cli_cfg)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    print("cfg is: ", cfg)

    main(cfg)
