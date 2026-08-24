"""K3D widgets for inspecting PhysSkin fields and LBS deformations."""

from pathlib import Path

import numpy as np
import torch

from .skinning import standard_lbs


def load_rignet_sample(dataset_root, model_id, device="cuda"):
    sample_dir = (
        Path(dataset_root).expanduser().resolve()
        / "00000001" / str(model_id) / "models" / "samples"
    )
    latents = np.load(sample_dir / "latents.npz")["latents"]
    points = np.load(sample_dir / "internal_filled.npz")["points"] * 2.0
    return (
        torch.tensor(latents, dtype=torch.float32, device=device),
        torch.tensor(points, dtype=torch.float32, device=device),
    )


def _rgb_to_uint32(rgb):
    rgb = np.asarray(rgb, dtype=np.uint32)
    return (rgb[:, 0] << 16) + (rgb[:, 1] << 8) + rgb[:, 2]


def _mode_colors(values, signed=True, clip_percentile=99.0):
    values = np.asarray(values, dtype=np.float32)
    if signed:
        scale = np.percentile(np.abs(values), clip_percentile)
        scale = 1.0 if scale < 1e-12 else scale
        normalized = np.clip(values / scale, -1.0, 1.0)
        rgb = np.ones((values.shape[0], 3), dtype=np.float32)
        negative = normalized < 0
        rgb[negative, 0] = 1.0 + normalized[negative]
        rgb[negative, 1] = 1.0 + normalized[negative]
        positive = ~negative
        rgb[positive, 1] = 1.0 - normalized[positive]
        rgb[positive, 2] = 1.0 - normalized[positive]
    else:
        low = np.percentile(values, 100.0 - clip_percentile)
        high = np.percentile(values, clip_percentile)
        high = low + 1.0 if abs(high - low) < 1e-12 else high
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
        rgb = np.stack([
            normalized,
            0.25 + 0.5 * (1.0 - np.abs(normalized - 0.5) * 2.0),
            1.0 - normalized,
        ], axis=1)
    return _rgb_to_uint32(np.clip(rgb * 255.0, 0, 255).astype(np.uint32))


def show_weight_modes(points: torch.Tensor, weights: torch.Tensor):
    import ipywidgets as widgets
    import k3d
    from IPython.display import display

    show_points = points.detach().cpu().numpy()
    show_weights = weights.detach().cpu().numpy()
    mode = widgets.IntSlider(
        value=0, min=0, max=show_weights.shape[1] - 1, description="mode"
    )
    signed = widgets.Checkbox(value=True, description="signed colors")
    clip = widgets.FloatSlider(
        value=99.0, min=90.0, max=100.0, step=0.1, description="clip %"
    )
    size = widgets.FloatLogSlider(
        value=0.01, base=10, min=-3.0, max=-0.5, step=0.05,
        description="pt size",
    )
    plot = k3d.plot(grid_visible=True, height=720)
    point_object = k3d.points(
        show_points,
        colors=_mode_colors(show_weights[:, 0]),
        point_size=0.01,
        shader="flat",
    )
    plot += point_object
    info = widgets.HTML()

    def update(*_):
        index = int(mode.value)
        values = show_weights[:, index]
        point_object.colors = _mode_colors(
            values, bool(signed.value), float(clip.value)
        )
        point_object.point_size = float(size.value)
        info.value = (
            f"<b>mode {index}</b> &nbsp; min={values.min():.6g}, "
            f"max={values.max():.6g}, mean={values.mean():.6g}, "
            f"l2={np.linalg.norm(values):.6g}"
        )

    for widget in (mode, signed, clip, size):
        widget.observe(update, names="value")
    update()
    display(widgets.VBox([widgets.HBox([mode, signed, clip, size]), info, plot]))
    return plot


def _euler_xyz_to_matrix(angles, device, dtype):
    rx, ry, rz = [
        torch.as_tensor(np.deg2rad(value), device=device, dtype=dtype)
        for value in angles
    ]
    sx, cx = torch.sin(rx), torch.cos(rx)
    sy, cy = torch.sin(ry), torch.cos(ry)
    sz, cz = torch.sin(rz), torch.cos(rz)
    rx_matrix = torch.stack([
        torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)]),
        torch.stack([torch.zeros_like(cx), cx, -sx]),
        torch.stack([torch.zeros_like(cx), sx, cx]),
    ])
    ry_matrix = torch.stack([
        torch.stack([cy, torch.zeros_like(cy), sy]),
        torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)]),
        torch.stack([-sy, torch.zeros_like(cy), cy]),
    ])
    rz_matrix = torch.stack([
        torch.stack([cz, -sz, torch.zeros_like(cz)]),
        torch.stack([sz, cz, torch.zeros_like(cz)]),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)]),
    ])
    return rz_matrix @ ry_matrix @ rx_matrix


def _build_transforms(rotations, translations, device, dtype):
    transforms = torch.zeros(
        (1, rotations.shape[0], 3, 4), device=device, dtype=dtype
    )
    identity = torch.eye(3, device=device, dtype=dtype)
    for handle in range(rotations.shape[0]):
        transforms[0, handle, :3, :3] = (
            _euler_xyz_to_matrix(rotations[handle], device, dtype) - identity
        )
        transforms[0, handle, :3, 3] = torch.as_tensor(
            translations[handle], device=device, dtype=dtype
        )
    return transforms


def show_lbs_deformation(points: torch.Tensor, weights: torch.Tensor):
    import ipywidgets as widgets
    import k3d
    from IPython.display import display

    rotations = np.zeros((weights.shape[1], 3), dtype=np.float32)
    translations = np.zeros((weights.shape[1], 3), dtype=np.float32)
    handle = widgets.IntSlider(
        value=0, min=0, max=weights.shape[1] - 1, description="handle"
    )
    rotation_widgets = [
        widgets.FloatSlider(value=0, min=-270, max=270, step=2, description=name)
        for name in ("rx", "ry", "rz")
    ]
    translation_widgets = [
        widgets.FloatSlider(value=0, min=-1, max=1, step=0.02, description=name)
        for name in ("tx", "ty", "tz")
    ]
    reset = widgets.Button(description="Reset")
    plot = k3d.plot(grid_visible=True, height=720)
    show_points = points.detach().cpu().numpy()
    original = k3d.points(
        show_points, color=0x888888, point_size=0.0065, shader="flat"
    )
    deformed = k3d.points(
        show_points.copy(), color=0xFFA15C, point_size=0.0095, shader="flat"
    )
    plot += original
    plot += deformed

    def sync_from_state(*_):
        index = int(handle.value)
        for axis in range(3):
            rotation_widgets[axis].value = float(rotations[index, axis])
            translation_widgets[axis].value = float(translations[index, axis])

    def update(*_):
        index = int(handle.value)
        rotations[index] = [widget.value for widget in rotation_widgets]
        translations[index] = [widget.value for widget in translation_widgets]
        with torch.no_grad():
            transforms = _build_transforms(
                rotations, translations, points.device, points.dtype
            )
            output = standard_lbs(points, transforms, weights).squeeze(1).squeeze(1)
        deformed.positions = output.detach().cpu().numpy()

    def reset_state(_):
        rotations[:] = 0
        translations[:] = 0
        sync_from_state()
        update()

    handle.observe(sync_from_state, names="value")
    for widget in rotation_widgets + translation_widgets:
        widget.observe(update, names="value")
    reset.on_click(reset_state)
    display(widgets.VBox([
        widgets.HBox([handle, reset]),
        widgets.HBox(rotation_widgets),
        widgets.HBox(translation_widgets),
        plot,
    ]))
    return plot

