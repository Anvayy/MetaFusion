"""CT slice selection, orientation verification, HU windowing, preprocessing, and QA overlays.

Standing Rule 3:
The segmentation mask is used ONLY inside `select_largest_tumor_slice` (to pick z*)
and `save_qa_overlay` (for Phase 3 visual QA). Model preprocessing functions NEVER
accept or touch any segmentation mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SliceSelectionResult:
    """Result of selecting the largest-tumor axial slice from a 3D segmentation mask."""

    z_star: int
    depth_axis: int
    n_slices: int
    tumor_pixels_at_z_star: int
    total_tumor_voxels: int
    has_tumor_mask: bool


def find_axial_depth_axis(affine: np.ndarray) -> tuple[int, tuple[str, str, str]]:
    """Determine which array axis corresponds to the Superior-Inferior (axial depth) axis.

    Uses nibabel's affine-to-axis-codes mapping (`'S'` or `'I'`). Raises ValueError if
    no unique S/I axis can be determined.
    """
    axcodes = tuple(str(c) for c in nib.aff2axcodes(affine))
    si_axes = [i for i, code in enumerate(axcodes) if code in ("S", "I")]
    if len(si_axes) != 1:
        raise ValueError(
            f"Expected exactly one Superior/Inferior axis in affine axcodes, got {axcodes}"
        )
    return si_axes[0], (axcodes[0], axcodes[1], axcodes[2])


def select_largest_tumor_slice(
    mask_3d: np.ndarray,
    tumor_label: int = 2,
    depth_axis: int = 0,
) -> SliceSelectionResult:
    """Select axial slice index z* = argmax_z sum(mask[z] == tumor_label).

    Mask is ONLY used here to find the slice index z*.
    """
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected 3D segmentation mask, got shape {mask_3d.shape}")
    if depth_axis not in (0, 1, 2):
        raise ValueError(f"Invalid depth_axis={depth_axis}; expected 0, 1, or 2.")

    spatial_axes = tuple(ax for ax in (0, 1, 2) if ax != depth_axis)
    tumor_mask = mask_3d == tumor_label
    per_slice_counts = np.sum(tumor_mask, axis=spatial_axes, dtype=np.int64)
    total_tumor = int(np.sum(per_slice_counts))
    n_slices = int(mask_3d.shape[depth_axis])

    if total_tumor > 0:
        z_star = int(np.argmax(per_slice_counts))
        max_pixels = int(per_slice_counts[z_star])
        return SliceSelectionResult(
            z_star=z_star,
            depth_axis=depth_axis,
            n_slices=n_slices,
            tumor_pixels_at_z_star=max_pixels,
            total_tumor_voxels=total_tumor,
            has_tumor_mask=True,
        )

    # Fallback ONLY if a mask genuinely has zero tumor voxels: pick max-foreground or mid-slice
    # and explicitly mark has_tumor_mask=False so QA flags it immediately.
    fg_counts = np.sum(mask_3d > 0, axis=spatial_axes, dtype=np.int64)
    z_fallback = int(np.argmax(fg_counts)) if int(np.sum(fg_counts)) > 0 else n_slices // 2
    return SliceSelectionResult(
        z_star=z_fallback,
        depth_axis=depth_axis,
        n_slices=n_slices,
        tumor_pixels_at_z_star=0,
        total_tumor_voxels=0,
        has_tumor_mask=False,
    )


def extract_axial_slice(volume_3d: np.ndarray, z_idx: int, depth_axis: int = 0) -> np.ndarray:
    """Extract a single 2D axial slice from a 3D volume along `depth_axis`."""
    if volume_3d.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume_3d.shape}")
    return np.take(volume_3d, indices=z_idx, axis=depth_axis)


def window_hu_to_unit(
    slice_hu: np.ndarray,
    hu_min: float = -160.0,
    hu_max: float = 240.0,
) -> np.ndarray:
    """Clip CT Hounsfield Units to [hu_min, hu_max] and scale linearly to [0.0, 1.0]."""
    if hu_max <= hu_min:
        raise ValueError(f"hu_max ({hu_max}) must be greater than hu_min ({hu_min}).")
    clipped = np.clip(np.asarray(slice_hu, dtype=np.float32), hu_min, hu_max)
    return (clipped - float(hu_min)) / float(hu_max - hu_min)


def preprocess_slice_for_resnet(
    slice_hu: np.ndarray,
    hu_min: float = -160.0,
    hu_max: float = 240.0,
    image_size: int = 224,
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    augment_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Convert raw 2D HU slice into a normalized [3, image_size, image_size] tensor for ResNet-18.

    Pipeline:
      1. HU window [hu_min, hu_max] -> [0, 1]
      2. Resize to (image_size, image_size) via bilinear interpolation
      3. Repeat single grayscale channel to 3 channels
      4. Optional training-only augmentation (applied on [3, H, W] in [0, 1])
      5. ImageNet mean/std normalization

    NOTE: Strictly accepts CT HU slice ONLY (no mask input is possible).
    """
    if slice_hu.ndim != 2:
        raise ValueError(f"Expected 2D HU slice, got shape {slice_hu.shape}")

    unit = window_hu_to_unit(slice_hu, hu_min=hu_min, hu_max=hu_max)
    tensor = torch.from_numpy(unit).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    resized = F.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # [1, 224, 224]
    rgb = resized.repeat(3, 1, 1)  # [3, 224, 224]

    if augment_fn is not None:
        rgb = augment_fn(rgb)

    mean = torch.tensor(imagenet_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(imagenet_std, dtype=torch.float32).view(3, 1, 1)
    return (rgb - mean) / std


def save_qa_overlay(
    slice_hu: np.ndarray,
    mask_2d: np.ndarray,
    out_path: str | Path,
    title: str,
    hu_min: float = -160.0,
    hu_max: float = 240.0,
    kidney_label: int = 1,
    tumor_label: int = 2,
    cyst_label: int = 3,
) -> Path:
    """Save a side-by-side QA figure: windowed CT (left) and CT + mask overlay (right)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ct_unit = window_hu_to_unit(slice_hu, hu_min=hu_min, hu_max=hu_max)
    rgb_overlay = np.stack([ct_unit, ct_unit, ct_unit], axis=-1)

    # Blend subtle colors: kidney=green, cyst=cyan/blue, tumor=red
    kidney_px = mask_2d == kidney_label
    cyst_px = mask_2d == cyst_label
    tumor_px = mask_2d == tumor_label

    alpha = 0.45
    rgb_overlay[kidney_px] = (1 - alpha) * rgb_overlay[kidney_px] + alpha * np.array([0.1, 0.85, 0.2])
    rgb_overlay[cyst_px] = (1 - alpha) * rgb_overlay[cyst_px] + alpha * np.array([0.1, 0.65, 1.0])
    rgb_overlay[tumor_px] = (1 - alpha) * rgb_overlay[tumor_px] + alpha * np.array([1.0, 0.15, 0.15])
    rgb_overlay = np.clip(rgb_overlay, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=130)
    axes[0].imshow(ct_unit, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Windowed CT [-160, 240] HU")
    axes[0].axis("off")

    axes[1].imshow(rgb_overlay)
    if np.any(tumor_px):
        axes[1].contour(tumor_px.astype(float), levels=[0.5], colors=["yellow"], linewidths=1.0)
    axes[1].set_title("Overlay (Red/Yellow=Tumor, Green=Kidney, Blue=Cyst)")
    axes[1].axis("off")

    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    return out
