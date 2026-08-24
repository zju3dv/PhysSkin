"""RigNet dataset loader for PhysSkin training."""

import os
import re
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


def natural_sort_key(value: str):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", value)
    ]


class RigNetDataset(Dataset):
    """Load preprocessed RigNet latents and interior samples.

    Expected layout::

        dataset_root/00000001/<model_id>/models/samples/latents.npz
        dataset_root/00000001/<model_id>/models/samples/internal_filled.npz
    """

    def __init__(
        self,
        dataset_root,
        split: str = "train",
        num_points: int = 1000,
        split_dir=None,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = split
        self.num_points = num_points
        self.split_dir = (
            Path(split_dir).expanduser().resolve()
            if split_dir is not None
            else Path(__file__).resolve().parents[1] / "data" / "splits"
        )
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        split_file = self.split_dir / f"{split}_final.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        valid_ids = {
            line.strip() for line in split_file.read_text().splitlines()
            if line.strip()
        }

        category = "00000001"
        category_folder = self.dataset_root / category
        if not category_folder.is_dir():
            raise FileNotFoundError(
                f"RigNet category directory not found: {category_folder}"
            )
        model_ids = [
            path.name for path in category_folder.iterdir()
            if path.is_dir() and path.name in valid_ids
        ]
        model_ids.sort(key=natural_sort_key)
        self.models = [(55, category, model_id) for model_id in model_ids]
        print(
            f"[INFO] Loaded {len(self.models)} models for split {split!r} "
            f"from {split_file}"
        )

    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        category_id, category, model_id = self.models[index]
        sample_dir = (
            self.dataset_root / category / model_id / "models" / "samples"
        )
        latent_file = sample_dir / "latents.npz"
        points_file = sample_dir / "internal_filled.npz"
        if not latent_file.is_file():
            raise FileNotFoundError(f"Missing latent set: {latent_file}")
        if not points_file.is_file():
            raise FileNotFoundError(f"Missing interior points: {points_file}")

        latents = np.load(latent_file)["latents"]
        points = np.load(points_file)["points"] * 2.0
        if points.shape[0] >= self.num_points:
            sample_indices = np.random.randint(
                low=0, high=points.shape[0], size=(self.num_points,)
            )
            points = points[sample_indices]
        return latents, points, category_id


# Compatibility alias used by the training script.
RigNetSplitShapeNetPCLatentDataset = RigNetDataset
