# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# Licensed under the Apache License, Version 2.0.

"""Pure-PyTorch linear blend skinning used for training and visualization."""

import torch


def weight_function_lbs(
    points: torch.Tensor, transforms: torch.Tensor, weight_function
) -> torch.Tensor:
    return standard_lbs(points, transforms, weight_function(points))


def standard_lbs(
    points: torch.Tensor,
    transforms: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Apply rest-relative LBS.

    Args:
        points: ``(N, 3)`` rest points.
        transforms: ``(B, H, 3, 4)`` rest-relative affine transforms.
        weights: ``(N, H)`` skinning weights.

    Returns:
        Tensor of shape ``(N, B, 1, 3)``.
    """
    num_points = points.shape[0]
    transform_batch = transforms.shape[0]
    num_handles = transforms.shape[1]
    flattened_handles = transform_batch * num_handles

    points_batched = points.unsqueeze(1)
    homogeneous = torch.cat(
        (points_batched, points_batched.new_ones(num_points, 1, 1)), dim=2
    ).transpose(1, 2)
    expanded_transforms = transforms.reshape(flattened_handles, 3, 4)
    expanded_transforms = expanded_transforms[None].expand(
        num_points, flattened_handles, 3, 4
    )
    expanded_points = homogeneous[:, None].expand(
        num_points, flattened_handles, 4, 1
    )
    weight_map = weights.unsqueeze(2)
    expanded_weights = weight_map[:, None, :, None, :].expand(
        num_points, transform_batch, num_handles, 3, 1
    ).reshape(num_points, flattened_handles, 3, 1)

    deformed = expanded_weights * expanded_transforms @ expanded_points
    deformed = deformed.reshape(
        num_points, transform_batch, num_handles, 3, 1
    ).sum(2)
    deformed = deformed.transpose(2, 3)
    deformed += points_batched[:, None].expand(
        num_points, transform_batch, 1, 3
    )
    return deformed

