"""Phase 3 unit tests: slice selection correctness, mask boundary isolation, and HU preprocessing."""

from __future__ import annotations

import csv
import inspect
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from metafusion_knight.config import load_config
from metafusion_knight.data.imaging import (
    find_axial_depth_axis,
    preprocess_slice_for_resnet,
    select_largest_tumor_slice,
    window_hu_to_unit,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


def test_select_largest_tumor_slice_ignores_larger_kidney_and_cyst() -> None:
    """Required Test: selected slice has max TUMOR area (label=2), ignoring larger kidney(1)/cyst(3)."""
    mask = np.zeros((10, 32, 32), dtype=np.uint8)
    # Slice 2 has huge kidney (label=1) and cyst (label=3), but NO tumor
    mask[2, 2:30, 2:30] = 1
    mask[2, 5:25, 5:25] = 3
    # Slice 5 has small tumor (4x4 = 16 px)
    mask[5, 10:14, 10:14] = 2
    # Slice 7 has largest tumor (6x6 = 36 px)
    mask[7, 10:16, 10:16] = 2

    res = select_largest_tumor_slice(mask, tumor_label=2, depth_axis=0)
    assert res.has_tumor_mask is True
    assert res.z_star == 7
    assert res.tumor_pixels_at_z_star == 36
    assert res.total_tumor_voxels == 52


def test_select_largest_tumor_slice_handles_different_depth_axes() -> None:
    mask = np.zeros((16, 16, 8), dtype=np.uint8)
    mask[4:10, 4:10, 3] = 2  # 36 px at z=3 along axis 2
    mask[4:7, 4:7, 6] = 2  # 9 px at z=6 along axis 2

    res = select_largest_tumor_slice(mask, tumor_label=2, depth_axis=2)
    assert res.z_star == 3
    assert res.depth_axis == 2
    assert res.n_slices == 8
    assert res.tumor_pixels_at_z_star == 36


def test_select_largest_tumor_slice_flags_missing_tumor() -> None:
    mask = np.zeros((6, 16, 16), dtype=np.uint8)
    mask[2, 4:10, 4:10] = 1  # kidney only, no tumor

    res = select_largest_tumor_slice(mask, tumor_label=2, depth_axis=0)
    assert res.has_tumor_mask is False
    assert res.tumor_pixels_at_z_star == 0
    assert res.total_tumor_voxels == 0


def test_hu_windowing_and_resnet_preprocessing_never_takes_mask() -> None:
    """Required Test: mask never reaches the model preprocessing pipeline."""
    sig = inspect.signature(preprocess_slice_for_resnet)
    param_names = set(sig.parameters.keys())
    for forbidden in ("mask", "seg", "segmentation", "mask_2d", "mask_3d"):
        assert forbidden not in param_names

    raw_hu = np.array([[-1000.0, -160.0], [40.0, 240.0], [1000.0, 40.0]], dtype=np.float32)
    unit = window_hu_to_unit(raw_hu, hu_min=-160.0, hu_max=240.0)
    assert np.allclose(unit, [[0.0, 0.0], [0.5, 1.0], [1.0, 0.5]], atol=1e-6)

    synthetic_ct = np.linspace(-500.0, 500.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    tensor = preprocess_slice_for_resnet(synthetic_ct)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_extracted_slice_index_matches_real_masks() -> None:
    """Verify slice_index.csv covers all 300 KNIGHT cases (0 KiTS23 extras) and z* matches real masks."""
    cfg = load_config(BASE_CONFIG_PATH)
    csv_path = REPO_ROOT / cfg["paths"]["slice_index_csv"]
    if not csv_path.exists():
        pytest.skip("slice_index.csv not yet generated")

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 300
    case_ids = [r["case_id"] for r in rows]
    assert len(set(case_ids)) == 300
    # Strict KNIGHT-only boundary: case_00000..case_00299
    assert all(0 <= int(cid.split("_")[1]) < 300 for cid in case_ids)

    seg_dir = REPO_ROOT / cfg["paths"]["kits23_seg_dir"]
    tumor_label = int(cfg["imaging"]["tumor_label"])

    # Spot-check 5 spread-out real masks against slice_index.csv
    for idx in [0, 50, 150, 225, 299]:
        r = rows[idx]
        cid = r["case_id"]
        mask_path = seg_dir / cid / "segmentation.nii.gz"
        assert mask_path.exists()
        nib_mask = nib.load(str(mask_path))
        mask_arr = np.asarray(nib_mask.dataobj)
        depth_axis, _ = find_axial_depth_axis(nib_mask.affine)
        sel = select_largest_tumor_slice(mask_arr, tumor_label=tumor_label, depth_axis=depth_axis)
        assert int(r["z_star"]) == sel.z_star
        assert int(r["tumor_pixels_at_z_star"]) == sel.tumor_pixels_at_z_star
        slice_file = REPO_ROOT / r["slice_path"]
        assert slice_file.exists()
        arr = np.load(str(slice_file))
        assert arr.ndim == 2
        assert np.isfinite(arr).all()
