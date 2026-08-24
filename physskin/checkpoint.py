"""Checkpoint loading and batched skinning-weight inference."""

from pathlib import Path

import numpy as np
import torch

from .model import PhysSkinPointShapeHandleModel


def load_model(checkpoint_path, device="cuda"):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    config = checkpoint["config"]
    model = PhysSkinPointShapeHandleModel(
        spatial_dimensions=3,
        layer_width=config["layer_width"],
        num_handles=config["num_handles"],
        num_layers=config["model_layers"],
        multires=config["multires"],
        num_self_attn=config["num_self_attn"],
        num_handle_tokens=config.get(
            "num_handle_tokens", config["num_handles"]
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, config


def load_latents(latent_path, device="cuda"):
    values = np.load(Path(latent_path).expanduser().resolve())["latents"]
    return torch.tensor(values, dtype=torch.float32, device=device)


@torch.no_grad()
def predict_weights(model, latents: torch.Tensor, points: torch.Tensor,
                    chunk_size=None):
    """Predict globally normalized weights.

    The default evaluates all points together, exactly like the research
    notebook. Set ``chunk_size`` only when memory is constrained; raw fields
    are then concatenated before the same global normalization is applied.
    """
    model.set_latent_feature(latents)
    if chunk_size is None:
        return model(points)
    raw_chunks = [
        model.forward_raw_weights(points[start:start + chunk_size])
        for start in range(0, points.shape[0], chunk_size)
    ]
    weights = torch.cat(raw_chunks, dim=0)
    weights = weights / (weights.norm(dim=0, keepdim=True) + 1e-8)
    rigid = torch.ones(
        weights.shape[0], 1, device=weights.device, dtype=weights.dtype
    )
    rigid = rigid / rigid.norm(dim=0, keepdim=True)
    return torch.cat([weights, rigid], dim=1)
