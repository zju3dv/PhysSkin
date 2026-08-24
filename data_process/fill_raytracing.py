"""Fill a watertight mesh with grid points using CUDA ray tracing."""

import argparse
import os

import numpy as np
import open3d as o3d
import raytracing
import torch
import trimesh


def compute_grid_inside_mesh(mesh, grid_coords, num_dirs=6):
    """Classify points as interior using intersections from six directions."""
    mesh_tracer = raytracing.RayTracer(mesh.vertices, mesh.faces)
    rays_o = torch.from_numpy(grid_coords).cuda()

    if num_dirs == 26:
        grid_dir = torch.tensor([-1, 0, 1])
        x, y, z = torch.meshgrid(grid_dir, grid_dir, grid_dir, indexing="ij")
        grid = torch.stack([x, y, z], dim=-1).reshape(-1, 3).float().cuda()
        rays_d = grid[(grid != 0).any(dim=-1)]
    elif num_dirs == 6:
        rays_d = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
        ).float().cuda()
    else:
        raise ValueError("num_dirs must be 6 or 26")
    rays_d = rays_d / torch.norm(rays_d, dim=-1)[..., None]

    intersections = torch.zeros(
        (len(grid_coords), len(rays_d)), dtype=torch.uint8, device="cuda"
    )
    grid_indices = torch.arange(len(intersections), device="cuda")
    rays_chunk = 81920

    complete_length = (len(grid_coords) // rays_chunk) * rays_chunk
    for start in range(0, complete_length, rays_chunk):
        end = start + rays_chunk
        chunk_origins = rays_o[start:end]
        chunk_size = len(chunk_origins)
        origins = chunk_origins[:, None].repeat(1, len(rays_d), 1).reshape(-1, 3)
        directions = rays_d[None].repeat(chunk_size, 1, 1).reshape(-1, 3)
        chunk_indices = grid_indices[start:end]

        _, face_normals, face_ids, _ = mesh_tracer.trace(origins, directions)
        interior_hits = (face_normals * directions).sum(-1) > 0
        interior_hits = interior_hits.reshape(chunk_size, -1)

        rows, columns = torch.where(interior_hits)
        intersections[chunk_indices[rows], columns] = 1
        rows, columns = torch.where(~interior_hits)
        intersections[chunk_indices[rows], columns] = 2
        rows, columns = torch.where(face_ids.reshape(chunk_size, -1) == -1)
        intersections[chunk_indices[rows], columns] = 0

    required_hits = int(num_dirs * 0.7)
    return ((intersections == 1).sum(-1) >= required_hits).cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh_file", required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--surface_path", required=True)
    parser.add_argument("--save_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    samples_dir = os.path.join(args.save_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    o3d_mesh = o3d.io.read_triangle_mesh(args.mesh_file)
    vertices = torch.as_tensor(
        np.asarray(o3d_mesh.vertices), dtype=torch.float32, device="cuda"
    )
    bounds_min = vertices.min(dim=0).values
    bounds_max = vertices.max(dim=0).values
    axes = [
        torch.linspace(bounds_min[i], bounds_max[i], args.resolution, device="cuda")
        for i in range(3)
    ]
    grid_x, grid_y, grid_z = torch.meshgrid(*axes, indexing="ij")
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    mesh = trimesh.load_mesh(args.mesh_file)
    inside_mask = compute_grid_inside_mesh(mesh, grid_points.cpu().numpy())
    inside_points = grid_points.cpu().numpy()[inside_mask]
    inside_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inside_points))
    inside_cloud.paint_uniform_color([0.5, 0.5, 1.0])

    surface_cloud = o3d.io.read_point_cloud(args.surface_path)
    combined_cloud = surface_cloud + inside_cloud
    o3d.io.write_point_cloud(
        os.path.join(args.save_dir, "internal_filled.ply"), combined_cloud
    )
    o3d.io.write_point_cloud(
        os.path.join(args.save_dir, "only_internal_filled.ply"), inside_cloud
    )

    np.savez_compressed(
        os.path.join(samples_dir, "internal_filled.npz"),
        points=np.asarray(combined_cloud.points, dtype=np.float32),
    )
    np.savez_compressed(
        os.path.join(args.save_dir, "mesh_indices_triangles.npz"),
        mesh_indices=np.arange(len(o3d_mesh.vertices)),
        mesh_triangles=np.asarray(o3d_mesh.triangles),
    )


if __name__ == "__main__":
    main()
