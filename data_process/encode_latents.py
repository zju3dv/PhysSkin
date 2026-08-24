"""Encode normalized point clouds into the shape latents used by PhysSkin.

Michelangelo and its pretrained shape encoder checkpoint are required.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch


def natural_key(path: Path):
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--category", default="00000001")
    parser.add_argument(
        "--michelangelo-root",
        type=Path,
        required=True,
        help="Directory containing the Michelangelo Python package directory",
    )
    parser.add_argument(
        "--encoder-checkpoint", type=Path, required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute existing latents.npz"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    category_dir = args.dataset_root / args.category
    if not category_dir.is_dir():
        raise FileNotFoundError(f"Category directory not found: {category_dir}")
    if not args.encoder_checkpoint.is_file():
        raise FileNotFoundError(
            f"Michelangelo encoder checkpoint not found: {args.encoder_checkpoint}"
        )

    sys.path.insert(0, str(args.michelangelo_root.resolve()))
    from Michelangelo.michelangelo.models.tsal.sal_perceiver import (  # noqa: E402
        AlignedShapeLatentEncoder,
    )

    device = torch.device(args.device)
    encoder = AlignedShapeLatentEncoder().to(device)
    checkpoint = torch.load(args.encoder_checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()

    model_dirs = sorted(
        (path for path in category_dir.iterdir() if path.is_dir()), key=natural_key
    )
    encoded = 0
    skipped = 0
    for model_dir in model_dirs:
        sample_dir = model_dir / "models" / "samples"
        input_path = sample_dir / "norm_pc_normals.npz"
        output_path = sample_dir / "latents.npz"
        if output_path.is_file() and not args.overwrite:
            print(f"SKIP: {output_path}")
            skipped += 1
            continue
        if not input_path.is_file():
            print(f"SKIP: missing {input_path}")
            skipped += 1
            continue

        with np.load(input_path) as data:
            points = torch.as_tensor(
                data["points"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            normals = torch.as_tensor(
                data["normals"], dtype=torch.float32, device=device
            ).unsqueeze(0)
        with torch.no_grad():
            _, latents = encoder(points, normals)
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, latents=latents.cpu().numpy())
        print(f"WROTE: {output_path} {tuple(latents.shape)}")
        encoded += 1

    print(f"Complete: encoded={encoded}, skipped={skipped}")


if __name__ == "__main__":
    main()
