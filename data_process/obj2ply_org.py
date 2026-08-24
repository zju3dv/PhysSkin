"""Convert an OBJ mesh to PLY without mesh processing."""

import argparse

import trimesh


def convert_obj_to_ply(input_obj_path: str, output_ply_path: str):
    mesh = trimesh.load_mesh(input_obj_path, process=False)
    if mesh is None:
        raise ValueError(f"Unable to load OBJ mesh: {input_obj_path}")
    mesh.export(output_ply_path, file_type="ply")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_obj", required=True)
    parser.add_argument("--output_ply", required=True)
    args = parser.parse_args()
    convert_obj_to_ply(args.input_obj, args.output_ply)
