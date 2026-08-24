import os
import open3d as o3d
import numpy as np
import argparse
import gc
import trimesh
import fpsample
from pysdf import SDF
import bpy
import math
import bmesh
from omegaconf import OmegaConf
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_vertices_as_ply_open3d(vertices, filepath):
    """Save point cloud vertices as PLY file using Open3D."""
    points = vertices
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector((points+1)/2)
    o3d.io.write_point_cloud(filepath, point_cloud, write_ascii=True)

def process_sharp_surface(mesh_path, angle_threshold, point_number=200000):
    # Ref Dora3D: https://github.com/Seed3D/Dora/tree/main/sharp_edge_sampling
    sharpness_threshold = math.radians(angle_threshold)

    bpy.ops.wm.obj_import(filepath=mesh_path)
    obj = bpy.context.selected_objects[0]

    # Enter Edit mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Ensure edge selection mode
    bpy.ops.mesh.select_mode(type="EDGE")

    # Select Sharp Edges
    bpy.ops.mesh.edges_select_sharp(sharpness=sharpness_threshold)

    # Return to object mode to read the selected edges.
    bpy.ops.object.mode_set(mode='OBJECT')

    bm = bmesh.new()
    bm.from_mesh(obj.data)

    sharp_edges = [edge for edge in bm.edges if edge.select]

    # Collect vertex pairs for sharp edges
    sharp_edges_vertices = []
    link_normal1 =[]
    link_normal2 = []
    sharp_edges_angle = []
    # Non-duplicate vertex set
    vertices_set = set()
    for edge in sharp_edges:
        vertices_set.update(edge.verts[:])
        
        sharp_edges_vertices.append([edge.verts[0].index, edge.verts[1].index])# Collect vertex pair indices for sharp edges

        normal1 = edge.link_faces[0].normal
        normal2 = edge.link_faces[1].normal

        link_normal1.append(normal1)
        link_normal2.append(normal2)

        if normal1.length==0.0 or normal2.length==0.0:
            sharp_edges_angle.append(0.0)
        # Compute the angle between the two normals
        else:
            sharp_edges_angle.append(math.degrees(normal1.angle(normal2)))

    vertices=[]
    vertices_index=[]
    vertices_normal=[]

    for vertice in vertices_set:
        vertices.append(vertice.co)
        vertices_index.append(vertice.index)
        vertices_normal.append(vertice.normal)


    vertices = np.array(vertices)
    vertices_index = np.array(vertices_index)
    vertices_normal = np.array(vertices_normal)

    sharp_edges_count = np.array(len(sharp_edges))
    sharp_edges_angle_array = np.array(sharp_edges_angle)

    if sharp_edges_count>0:
        sharp_edge_link_normal = np.array(np.concatenate([link_normal1,link_normal2], axis=1))
        nan_mask = np.isnan(sharp_edge_link_normal)
        # Replace NaN values with 0 using boolean indexing
        sharp_edge_link_normal = np.where(nan_mask, 0, sharp_edge_link_normal)
        
        nan_mask = np.isnan(vertices_normal)
        # Replace NaN values with 0 using boolean indexing
        vertices_normal = np.where(nan_mask, 0, vertices_normal)

    # Convert to numpy arrays
    sharp_edges_vertices_array = np.array(sharp_edges_vertices)

    if sharp_edges_count>0:
        mesh = trimesh.load(mesh_path,process =False)
        num_target_sharp_vertices = point_number // 2
        sharp_edge_length = sharp_edges_count
        sharp_edges_vertices_pair = sharp_edges_vertices_array
        sharp_vertices_pair = mesh.vertices[sharp_edges_vertices_pair]
        epsilon = 1e-4
        edge_normal =  0.5*sharp_edge_link_normal[:,:3] + 0.5 * sharp_edge_link_normal[:,3:]
        norms = np.linalg.norm(edge_normal, axis=1, keepdims=True)
        norms = np.where(norms > epsilon, norms, epsilon)
        edge_normal = edge_normal / norms
        known_vertices = vertices
        known_vertices_normal = vertices_normal
        known_vertices = np.concatenate([known_vertices,known_vertices_normal], axis=1)

        num_known_vertices = known_vertices.shape[0]
        if num_known_vertices < num_target_sharp_vertices:
            num_new_vertices = num_target_sharp_vertices - num_known_vertices 
            if num_new_vertices >= sharp_edge_length:
                num_new_vertices_per_pair = num_new_vertices // sharp_edge_length
                new_vertices = np.zeros((sharp_edge_length, num_new_vertices_per_pair, 6))

                start_vertex = sharp_vertices_pair[:, 0]
                end_vertex = sharp_vertices_pair[:, 1]
                for j in range(1, num_new_vertices_per_pair+1):
                    t = j / float(num_new_vertices_per_pair+1)
                    new_vertices[:, j - 1 , :3] = (1 - t) * start_vertex + t * end_vertex
                    
                    new_vertices[:, j - 1 , 3:] = edge_normal
                new_vertices= new_vertices.reshape(-1,6)

                remaining_vertices = num_new_vertices % sharp_edge_length
                if remaining_vertices>0:
                    rng = np.random.default_rng()
                    ind = rng.choice(sharp_edge_length, remaining_vertices, replace=False)
                    new_vertices_remain = np.zeros((remaining_vertices, 6))# Initialize new vertex array
                    start_vertex = sharp_vertices_pair[ind, 0]
                    end_vertex = sharp_vertices_pair[ind, 1]
                    t = np.random.rand(remaining_vertices).reshape(-1,1)
                    new_vertices_remain[:,:3] = (1 - t) * start_vertex + t * end_vertex

                    edge_normal =  0.5*sharp_edge_link_normal[ind,:3] + 0.5 * sharp_edge_link_normal[ind,3:]
                    edge_normal = edge_normal / np.linalg.norm(edge_normal,axis=1,keepdims=True)
                    new_vertices_remain[:,3:] = edge_normal

                    new_vertices = np.concatenate([new_vertices,new_vertices_remain], axis=0)
            else:
                remaining_vertices = num_new_vertices % sharp_edge_length
                if remaining_vertices>0:
                    rng = np.random.default_rng() 
                    ind = rng.choice(sharp_edge_length, remaining_vertices, replace=False)
                    new_vertices_remain = np.zeros((remaining_vertices, 6))# Initialize new vertex array
                    start_vertex = sharp_vertices_pair[ind, 0]
                    end_vertex = sharp_vertices_pair[ind, 1]
                    t = np.random.rand(remaining_vertices).reshape(-1,1)
                    new_vertices_remain[:,:3] = (1 - t) * start_vertex + t * end_vertex

                    edge_normal =  0.5*sharp_edge_link_normal[ind,:3] + 0.5 * sharp_edge_link_normal[ind,3:]
                    edge_normal = edge_normal / np.linalg.norm(edge_normal,axis=1,keepdims=True)
                    new_vertices_remain[:,3:] = edge_normal

                    new_vertices = new_vertices_remain


            target_vertices = np.concatenate([new_vertices,known_vertices], axis=0)
        else:
            target_vertices = known_vertices

        sharp_surface = target_vertices

        sharp_surface_points = sharp_surface[:,:3]

        sharp_near_surface_points= [
                        sharp_surface_points + np.random.normal(scale=0.001, size=(len(sharp_surface_points), 3)),
                        sharp_surface_points + np.random.normal(scale=0.005, size=(len(sharp_surface_points),3)),
                        sharp_surface_points + np.random.normal(scale=0.007, size=(len(sharp_surface_points),3)),
                        sharp_surface_points + np.random.normal(scale=0.01, size=(len(sharp_surface_points),3))
            ]
        sharp_near_surface_points = np.concatenate(sharp_near_surface_points)

        f = SDF(mesh.vertices, mesh.faces)
        sharp_sdf = f(sharp_near_surface_points).reshape(-1,1)
        sharp_near_surface = np.concatenate([sharp_near_surface_points, sharp_sdf], axis=1)

        fps_sharp_surface_list=[]
        if sharp_surface.shape[0]>num_target_sharp_vertices:
            for _ in range(1):
                kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(sharp_surface_points, num_target_sharp_vertices, h=5)
                fps_sharp_surface = sharp_surface[kdline_fps_samples_idx].reshape(-1,1,6)
                fps_sharp_surface_list.append(fps_sharp_surface) 

            fps_sharp_surface = np.concatenate(fps_sharp_surface_list, axis=1)
        else:
            logger.info(f"sharp_surface.shape: {sharp_surface.shape}")
            fps_sharp_surface = sharp_surface[:,None]
        
        sharp_surface[np.isinf(sharp_surface)] = 1
        sharp_surface[np.isnan(sharp_surface)] = 1

    return sharp_near_surface, fps_sharp_surface

def process_coarse_surface(mesh_path, point_number=200000):
    point_number = point_number // 2
    # Disable mesh processing to preserve the input topology.
    mesh = trimesh.load(mesh_path, process=False)
    
    # Sample points on the surface
    coarse_surface_points, faces = mesh.sample(point_number, return_index=True)
    normals = mesh.face_normals[faces]
    coarse_surface = np.concatenate([coarse_surface_points, normals], axis=1)
    
    # Generate near-surface points with different noise levels
    coarse_near_surface_points = []
    
    # First set with moderate noise (scale=0.01)
    resampled_points1, faces1 = mesh.sample(len(coarse_surface_points), return_index=True)
    coarse_near_surface_points.append(
        resampled_points1 + np.random.normal(scale=0.001, size=(len(resampled_points1), 3))
    )
    
    # Second set with smaller noise (scale=0.005)
    resampled_points2, faces2 = mesh.sample(len(coarse_surface_points), return_index=True)
    coarse_near_surface_points.append(
        resampled_points2 + np.random.normal(scale=0.005, size=(len(resampled_points2), 3))
    )
    
    # Concatenate all near-surface points
    coarse_near_surface_points = np.concatenate(coarse_near_surface_points)
    
    # Sample points in space (slightly beyond normalized bounds)
    space_points = np.random.uniform(-1.05, 1.05, (point_number, 3))
    rand_points = np.concatenate([coarse_near_surface_points, space_points], axis=0)
    
    # Calculate SDF values for all random points
    f = SDF(mesh.vertices, mesh.faces)
    coarse_sdf = f(rand_points).reshape(-1, 1)
    rand_points = np.concatenate([rand_points, coarse_sdf], axis=1)
    rand_indices = np.random.permutation(rand_points.shape[0])
    rand_points = rand_points[rand_indices]
    
    # Perform FPS (Farthest Point Sampling) on surface points
    fps_coarse_surface_list = []
    for _ in range(1):
        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(coarse_surface_points, point_number, h=5)
        fps_coarse_surface = coarse_surface[kdline_fps_samples_idx].reshape(-1, 1, 6)
        fps_coarse_surface_list.append(fps_coarse_surface)
    fps_coarse_surface = np.concatenate(fps_coarse_surface_list, axis=1)
    
    # Clean up invalid values
    fps_coarse_surface[np.isinf(fps_coarse_surface)] = 1
    fps_coarse_surface[np.isnan(fps_coarse_surface)] = 1
    return fps_coarse_surface, rand_points

def split_save_data(save_dir, num_splits, **kwargs):
    """Split and save data arrays into multiple NPZ files."""
    os.makedirs(save_dir, exist_ok=True)
    base_path = os.path.join(save_dir, "sample")
    
    # Ensure at least one split
    actual_splits = max(1, num_splits)
    
    for i in range(actual_splits):
        split_data = {}
        
        for key, value in kwargs.items():
            # Split arrays if they have shape and multiple splits requested
            if hasattr(value, 'shape') and len(value.shape) > 0 and num_splits > 1:
                points_per_split = value.shape[0] // num_splits
                start_idx = i * points_per_split
                end_idx = start_idx + points_per_split if i < num_splits - 1 else value.shape[0]
                split_data[key] = value[start_idx:end_idx].astype(np.float32)
            else:
                # Keep as is for non-array data or single split
                split_data[key] = value.astype(np.float32) if hasattr(value, 'astype') else value
        
        # Save the split
        split_path = f"{base_path}_{i}.npz"
        np.savez(split_path, **split_data)
        logger.info(f"Saved split {i+1}/{actual_splits} to {split_path}")
    

def main(cfg) -> None:
    os.makedirs(cfg.output.save_dir, exist_ok=True)
    try:
        processed_data = {}
        if cfg.sampling.use_sharp_sample:   
            sharp_near_surface_points, sharp_surface = process_sharp_surface(cfg.input.mesh_path, cfg.sampling.angle_threshold, cfg.sampling.point_number)
            logger.info(f"sharp_near_surface_points.shape: {sharp_near_surface_points.shape}")
            logger.info(f"sharp_surface.shape: {sharp_surface.shape}")
            processed_data['fps_sharp_surface'] = sharp_surface
            processed_data['sharp_near_surface_points'] = sharp_near_surface_points

        if cfg.sampling.use_coarse_sample:
            fps_coarse_surface, rand_points = process_coarse_surface(cfg.input.mesh_path, cfg.sampling.point_number)
            logger.info(f"fps_coarse_surface.shape: {fps_coarse_surface.shape}")
            logger.info(f"rand_points.shape: {rand_points.shape}")
            processed_data['fps_coarse_surface'] = fps_coarse_surface
            processed_data['rand_points'] = rand_points

        split_save_data(cfg.output.save_dir, cfg.output.num_split, **processed_data)

        os.makedirs(os.path.join(cfg.output.save_dir), exist_ok=True)
        ply_output_path = os.path.join(cfg.output.save_dir, "sample_surface.ply")
        save_vertices_as_ply_open3d(fps_coarse_surface[:, 0, :3], ply_output_path)

        return True

    except Exception as e:
        logger.error(f"ERROR: in processing path: {cfg.input.mesh_path}. Error: {e}")
        gc.collect()
        return False


if __name__ == "__main__":
    print("code start")
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "mesh_sample.yaml",
    )
    cfg = OmegaConf.load(config_path)
    cli_cfg = OmegaConf.from_dotlist(sys.argv[1:])
    cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
