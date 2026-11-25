"""Verify TUMOR_LABEL, axial depth axis, CT/mask alignment, and download speed on 3 sample KNIGHT cases."""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metafusion_knight.config import load_config
from metafusion_knight.data.imaging import (
    extract_axial_slice,
    find_axial_depth_axis,
    save_qa_overlay,
    select_largest_tumor_slice,
)
from metafusion_knight.data.labels import load_knight_records


def download_file(url: str, dst: Path) -> float:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".part.{dst.name}"
    t0 = time.perf_counter()
    urllib.request.urlretrieve(url, str(tmp))
    tmp.replace(dst)
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--cases", nargs="+", default=["case_00000", "case_00001", "case_00002"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    records = {r["case_id"]: r for r in load_knight_records(REPO_ROOT / cfg["paths"]["knight_json"])}
    seg_dir = REPO_ROOT / cfg["paths"]["kits23_seg_dir"]
    tmp_ct_dir = REPO_ROOT / cfg["paths"]["tmp_ct_dir"]
    qa_dir = REPO_ROOT / cfg["paths"]["qa_figures_dir"] / "_sample_verify"

    ct_tmpl = cfg["imaging"]["ct_url_template"]
    mask_tmpl = cfg["imaging"]["mask_url_template"]
    tumor_label = int(cfg["imaging"]["tumor_label"])

    for cid in args.cases:
        rec = records[cid]
        mask_path = seg_dir / cid / "segmentation.nii.gz"
        ct_path = tmp_ct_dir / f"{cid}.nii.gz"

        t_mask = 0.0
        if not mask_path.exists():
            t_mask = download_file(mask_tmpl.format(case_id=cid), mask_path)
        t_ct = 0.0
        if not ct_path.exists():
            t_ct = download_file(ct_tmpl.format(case_id=cid), ct_path)

        ct_mb = ct_path.stat().st_size / (1024 * 1024)
        mask_mb = mask_path.stat().st_size / (1024 * 1024)

        nib_mask = nib.load(str(mask_path))
        nib_ct = nib.load(str(ct_path))
        mask_arr = np.asarray(nib_mask.dataobj)
        ct_arr = np.asarray(nib_ct.dataobj, dtype=np.float32)

        depth_axis, axcodes = find_axial_depth_axis(nib_mask.affine)
        ct_depth_axis, ct_axcodes = find_axial_depth_axis(nib_ct.affine)

        sitk_mask = sitk.ReadImage(str(mask_path))
        sitk_arr = sitk.GetArrayFromImage(sitk_mask)

        uniq, cnts = np.unique(mask_arr, return_counts=True)
        sel = select_largest_tumor_slice(mask_arr, tumor_label=tumor_label, depth_axis=depth_axis)

        ct_slice = extract_axial_slice(ct_arr, sel.z_star, depth_axis=depth_axis)
        mask_slice = extract_axial_slice(mask_arr, sel.z_star, depth_axis=depth_axis)

        save_qa_overlay(
            ct_slice,
            mask_slice,
            qa_dir / f"{cid}_z{sel.z_star}.png",
            title=(
                f"{cid} | AUA={rec['aua_risk_group']} | z*={sel.z_star}/{sel.n_slices} "
                f"(axis={depth_axis}, {axcodes}) | tumor_px={sel.tumor_pixels_at_z_star}"
            ),
            hu_min=float(cfg["imaging"]["hu_window_min"]),
            hu_max=float(cfg["imaging"]["hu_window_max"]),
            tumor_label=tumor_label,
        )

        print(f"\n=== {cid} ===")
        print(f"  Sizes: mask={mask_mb:.2f} MB ({t_mask:.1f}s), CT={ct_mb:.2f} MB ({t_ct:.1f}s, {ct_mb/max(t_ct,1e-3):.1f} MB/s)")
        print(f"  nibabel shape: ct={ct_arr.shape}, mask={mask_arr.shape}, match={ct_arr.shape == mask_arr.shape}")
        print(f"  nibabel affine match: {np.allclose(nib_ct.affine, nib_mask.affine, atol=1e-4)}")
        print(f"  nibabel axcodes: ct={ct_axcodes} (depth_axis={ct_depth_axis}), mask={axcodes} (depth_axis={depth_axis})")
        print(f"  nibabel zooms: {tuple(round(float(z), 4) for z in nib_ct.header.get_zooms())}")
        print(f"  knight.json voxel_spacing: {rec['voxel_spacing']}")
        print(f"  SimpleITK array shape: {sitk_arr.shape}, spacing (xyz): {tuple(round(s, 4) for s in sitk_mask.GetSpacing())}")
        print(f"  Unique mask labels: {dict(zip([int(u) for u in uniq], [int(c) for c in cnts]))}")
        print(f"  Selected slice: z*={sel.z_star}/{sel.n_slices}, tumor_px={sel.tumor_pixels_at_z_star}, total_tumor={sel.total_tumor_voxels}")
        print(f"  CT slice HU range: [{float(ct_slice.min()):.1f}, {float(ct_slice.max()):.1f}]")

        # Clean up temporary 3D CT immediately to preserve X: free disk space
        ct_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
