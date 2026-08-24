# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# Licensed under the Apache License, Version 2.0.

"""Physics-informed objectives for PhysSkin training."""

from functools import partial

import torch
import torch.nn as nn

from .skinning import weight_function_lbs


def finite_diff_jac(function, points: torch.Tensor, eps: float = 1e-7):
    """Compute a Jacobian with central finite differences."""
    delta = torch.sqrt(torch.tensor(eps, device=points.device))
    offsets = delta * torch.eye(
        points.shape[1], device=points.device, dtype=points.dtype
    )
    bounds = torch.cat([
        points + offsets[0],
        points + offsets[1],
        points + offsets[2],
        points - offsets[0],
        points - offsets[1],
        points - offsets[2],
    ], dim=0)
    jacobian = function(bounds)
    jacobian = jacobian.reshape(2, 3, -1, *jacobian.shape[1:])
    jacobian = jacobian[0] - jacobian[1]
    jacobian = jacobian.permute(*range(1, jacobian.dim()), 0)
    return jacobian / (2.0 * delta)


def to_lame(youngs_modulus: torch.Tensor, poisson_ratio: torch.Tensor):
    mu = youngs_modulus / (2 * (1 + poisson_ratio))
    lam = (
        youngs_modulus * poisson_ratio
        / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    )
    return mu, lam


def cauchy_strain(deformation_gradient: torch.Tensor) -> torch.Tensor:
    dimensions = deformation_gradient.shape
    return (
        0.5 * (
            deformation_gradient.transpose(-2, -1) + deformation_gradient
        )
        - torch.eye(3, device=deformation_gradient.device)[None].expand(dimensions)
    )


def linear_elastic_energy(
    mu: torch.Tensor, lam: torch.Tensor, deformation_gradient: torch.Tensor
) -> torch.Tensor:
    dimensions = deformation_gradient.shape
    batch_dimensions = dimensions[:-2]
    strain = cauchy_strain(deformation_gradient)
    batched_trace = torch.vmap(torch.trace)
    trace_strain = batched_trace(
        strain.reshape(batch_dimensions.numel(), 3, 3)
    ).reshape(batch_dimensions).unsqueeze(-1)
    strain_outer = torch.matmul(strain.transpose(-2, -1), strain)
    return (
        mu
        * batched_trace(
            strain_outer.reshape(batch_dimensions.numel(), 3, 3)
        ).reshape(batch_dimensions).unsqueeze(-1)
        + (lam / 2) * trace_strain * trace_strain
    )


def neohookean_energy(
    mu: torch.Tensor, lam: torch.Tensor, deformation_gradient: torch.Tensor
) -> torch.Tensor:
    c1 = mu / 2
    d1 = lam / 2
    dimensions = deformation_gradient.shape
    batch_dimensions = dimensions[:-2]
    batched_trace = torch.vmap(torch.trace)
    right_cauchy_green = torch.matmul(
        torch.transpose(deformation_gradient, -2, -1), deformation_gradient
    )
    invariant = batched_trace(
        right_cauchy_green.reshape(batch_dimensions.numel(), 3, 3)
    ).reshape(batch_dimensions).unsqueeze(-1)
    determinant = torch.det(deformation_gradient).unsqueeze(-1)
    return (
        c1 * (invariant - 3)
        + d1 * (determinant - 1) * (determinant - 1)
        - mu * (determinant - 1.0)
    )


def loss_elastic(
    model,
    points: torch.Tensor,
    youngs_modulus: torch.Tensor,
    poisson_ratio: torch.Tensor,
    density: torch.Tensor,
    transforms: torch.Tensor,
    approximate_volume: torch.Tensor,
    interpolation_step: float,
) -> torch.Tensor:
    del density
    mu, lam = to_lame(youngs_modulus, poisson_ratio)
    lbs_function = partial(weight_function_lbs, transforms=transforms,
                           weight_function=model)
    deformation_gradients = finite_diff_jac(lbs_function, points)[:, :, 0]
    num_points, transform_batch = deformation_gradients.shape[:2]

    try:
        mu = mu[:, None].expand(num_points, transform_batch).unsqueeze(-1)
        lam = lam[:, None].expand(num_points, transform_batch).unsqueeze(-1)
    except Exception:
        mu = mu.expand(num_points, transform_batch).unsqueeze(-1)
        lam = lam.expand(num_points, transform_batch).unsqueeze(-1)

    linear = (1 - interpolation_step) * linear_elastic_energy(
        mu, lam, deformation_gradients
    )
    neo = interpolation_step * neohookean_energy(
        mu, lam, deformation_gradients
    )
    return (approximate_volume / points.shape[0]) * torch.sum(linear + neo)


def loss_ortho_neural(
    weights: torch.Tensor, include_rigid: bool = False
) -> torch.Tensor:
    neural_weights = weights if include_rigid else weights[:, :-1]
    gram = neural_weights.T @ neural_weights
    return nn.MSELoss()(
        gram,
        torch.eye(neural_weights.shape[1], device=neural_weights.device),
    )


def compute_losses_batched_obj_loop(
    model,
    normalized_pts: torch.Tensor,
    yms,
    prs,
    rhos,
    en_interp: float,
    batch_size: int,
    num_handles: int,
    appx_vol,
    le_coeff: float,
    lo_coeff: float,
    include_rigid: bool = False,
):
    """Compute elastic and orthogonality objectives one object at a time."""
    object_batch, num_samples, _ = normalized_pts.shape
    device, dtype = normalized_pts.device, normalized_pts.dtype
    if not torch.is_tensor(yms):
        yms = torch.full((object_batch, num_samples), yms,
                         dtype=dtype, device=device)
    if not torch.is_tensor(prs):
        prs = torch.full((object_batch, num_samples), prs,
                         dtype=dtype, device=device)
    if not torch.is_tensor(rhos):
        rhos = torch.full((object_batch, num_samples), rhos,
                          dtype=dtype, device=device)
    if not torch.is_tensor(appx_vol):
        appx_vol = torch.tensor([appx_vol], dtype=dtype, device=device)
    yms, prs, rhos, appx_vol = (
        yms.to(device), prs.to(device), rhos.to(device), appx_vol.to(device)
    )

    total_elastic = 0.0
    total_ortho = 0.0
    transforms = 0.2 * torch.randn(
        batch_size, num_handles + 1, 3, 4, dtype=dtype, device=device
    )
    for object_index in range(object_batch):
        points = normalized_pts[object_index]
        weights = model(points)
        total_elastic += le_coeff * loss_elastic(
            model,
            points,
            yms[object_index],
            prs[object_index],
            rhos[object_index],
            transforms,
            appx_vol,
            en_interp,
        )
        total_ortho += lo_coeff * loss_ortho_neural(
            weights, include_rigid=include_rigid
        )
    return total_elastic / object_batch, total_ortho / object_batch
