"""Phase 3: Resumable streaming slice extraction + visual QA overlay generation for all 300 KNIGHT patients.

Strict Standing Rules enforced:
- ONLY KNIGHT patients from `knight.json` (300 patients, case_00000..case_00299). No KiTS23 extra cases.
- Segmentation mask is used ONLY to choose z* = argmax_z sum(mask[z] == TUMOR_LABEL) and for Phase 3 QA overlays.
- Disk-safe streaming: temporary 3D CT `.nii.gz` files are deleted immediately after verifying the saved 2D HU `.npy`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metafusion_knight.config import load_config, set_seed
from metafusion_knight.data.imaging import (
    extract_axial_slice,
    find_axial_depth_axis,
    save_qa_overlay,
    select_largest_tumor_slice,
)
from metafusion_knight.data.labels import load_knight_records


def download_with_retry(url: str, dst: Path, retries: int = 3) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".part.{dst.name}"
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, str(tmp))
            tmp.replace(dst)
            return
        except Exception as exc:
            last_err = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_err}")


def process_one_patient(
    rec: dict[str, Any],
    cfg: dict[str, Any],
    seg_dir: Path,
    tmp_ct_dir: Path,
    slices_dir: Path,
    qa_masks_dir: Path,
) -> dict[str, Any]:
    cid = str(rec["case_id"])
    # Hard guardrail against KiTS23 extra cases (Standing Rule 1)
    case_num = int(cid.split("_")[1])
    if not (0 <= case_num < 300):
        raise ValueError(f"Refusing non-KNIGHT case_id {cid!r}; only case_00000..case_00299 permitted.")

    ct_url = cfg["imaging"]["ct_url_template"].format(case_id=cid)
    mask_url = cfg["imaging"]["mask_url_template"].format(case_id=cid)
    tumor_label = int(cfg["imaging"]["tumor_label"])

    mask_path = seg_dir / cid / "segmentation.nii.gz"
    slice_hu_path = slices_dir / f"{cid}_hu.npy"
    qa_mask_path = qa_masks_dir / f"{cid}_mask2d.npy"
    meta_cache_path = slices_dir / f"{cid}_meta.json"

    # Resume cleanly if already extracted & verified
    if slice_hu_path.exists() and qa_mask_path.exists() and meta_cache_path.exists() and mask_path.exists():
        with meta_cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if not mask_path.exists():
        download_with_retry(mask_url, mask_path)

    nib_mask = nib.load(str(mask_path))
    mask_arr = np.asarray(nib_mask.dataobj)
    depth_axis, axcodes = find_axial_depth_axis(nib_mask.affine)
    uniq_labels = [int(u) for u in np.unique(mask_arr)]
    sel = select_largest_tumor_slice(mask_arr, tumor_label=tumor_label, depth_axis=depth_axis)

    # Compute in-plane pixel spacing (the two axes != depth_axis)
    zooms = tuple(float(z) for z in nib_mask.header.get_zooms()[:3])
    in_plane_zooms = [zooms[i] for i in range(3) if i != depth_axis]
    pixel_area_mm2 = float(in_plane_zooms[0] * in_plane_zooms[1])
    tumor_area_mm2 = round(sel.tumor_pixels_at_z_star * pixel_area_mm2, 2)

    # Download temporary 3D CT, verify alignment, extract 2D slice, delete temp 3D CT immediately
    tmp_ct_path = tmp_ct_dir / f"{cid}.nii.gz"
    try:
        download_with_retry(ct_url, tmp_ct_path)
        nib_ct = nib.load(str(tmp_ct_path))
        ct_shape = tuple(int(s) for s in nib_ct.shape)
        mask_shape = tuple(int(s) for s in mask_arr.shape)
        shape_match = ct_shape == mask_shape
        affine_match = bool(np.allclose(nib_ct.affine, nib_mask.affine, atol=1e-3))
        if not shape_match:
            raise ValueError(
                f"Shape mismatch for {cid}: CT={ct_shape} vs Mask={mask_shape}"
            )

        # Slice lazily through dataobj to save RAM & time
        slicer: list[Any] = [slice(None), slice(None), slice(None)]
        slicer[depth_axis] = sel.z_star
        ct_slice = np.asarray(nib_ct.dataobj[tuple(slicer)], dtype=np.float32)
        mask_slice = extract_axial_slice(mask_arr, sel.z_star, depth_axis=depth_axis).astype(np.uint8)

        # Atomic-ish save & readback verification before unlinking temp 3D CT
        slices_dir.mkdir(parents=True, exist_ok=True)
        qa_masks_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(slice_hu_path), ct_slice)
        np.save(str(qa_mask_path), mask_slice)

        # Verify saved arrays load cleanly
        check_ct = np.load(str(slice_hu_path))
        if check_ct.shape != ct_slice.shape or not np.isfinite(check_ct).all():
            raise RuntimeError(f"Saved slice verification failed for {cid}")
    finally:
        # Release file handle (Windows) before unlink
        nib_ct = None
        tmp_ct_path.unlink(missing_ok=True)

    row = {
        "case_id": cid,
        "aua_risk_group": str(rec["aua_risk_group"]),
        "radiographic_size_cm": float(rec["radiographic_size"]),
        "z_star": int(sel.z_star),
        "depth_axis": int(sel.depth_axis),
        "axcodes": "".join(axcodes),
        "n_slices": int(sel.n_slices),
        "volume_shape": "x".join(str(s) for s in mask_arr.shape),
        "slice_shape": "x".join(str(s) for s in ct_slice.shape),
        "zooms_mm": ",".join(f"{z:.4f}" for z in zooms),
        "unique_mask_labels": ",".join(str(u) for u in uniq_labels),
        "has_tumor_mask": bool(sel.has_tumor_mask),
        "tumor_pixels_at_z_star": int(sel.tumor_pixels_at_z_star),
        "tumor_area_mm2": float(tumor_area_mm2),
        "total_tumor_voxels": int(sel.total_tumor_voxels),
        "ct_mask_shape_match": bool(shape_match),
        "ct_mask_affine_match": bool(affine_match),
        "slice_hu_min": round(float(np.min(ct_slice)), 2),
        "slice_hu_max": round(float(np.max(ct_slice)), 2),
        "slice_path": str(slice_hu_path.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    with meta_cache_path.open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    return row


def generate_qa_overlays(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    slices_dir: Path,
    qa_masks_dir: Path,
    qa_figures_dir: Path,
) -> dict[str, Any]:
    """Save visual QA overlays for 20 random patients + extreme cases + a contact sheet."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    qa_figures_dir.mkdir(parents=True, exist_ok=True)
    qa_seed = int(cfg["imaging"].get("qa_seed", 42))
    n_random = int(cfg["imaging"].get("qa_num_random", 20))
    hu_min = float(cfg["imaging"]["hu_window_min"])
    hu_max = float(cfg["imaging"]["hu_window_max"])
    kidney_label = int(cfg["imaging"]["kidney_label"])
    tumor_label = int(cfg["imaging"]["tumor_label"])
    cyst_label = int(cfg["imaging"]["cyst_label"])

    rows_by_id = {r["case_id"]: r for r in rows}

    # Extreme cases: missing tumor mask, 3 smallest tumors, 3 largest tumors
    missing_tumor = [r["case_id"] for r in rows if not r["has_tumor_mask"]]
    with_tumor = [r for r in rows if r["has_tumor_mask"]]
    smallest_3 = [r["case_id"] for r in sorted(with_tumor, key=lambda x: x["tumor_area_mm2"])[:3]]
    largest_3 = [r["case_id"] for r in sorted(with_tumor, key=lambda x: x["tumor_area_mm2"], reverse=True)[:3]]

    rng = random.Random(qa_seed)
    all_ids = sorted(rows_by_id.keys())
    random_20 = sorted(rng.sample(all_ids, min(n_random, len(all_ids))))

    tagged_cases: list[tuple[str, str]] = []
    seen: set[str] = set()
    for cid in missing_tumor:
        tagged_cases.append((cid, "EXTREME_MISSING_TUMOR"))
        seen.add(cid)
    for rank, cid in enumerate(smallest_3, start=1):
        if cid not in seen:
            tagged_cases.append((cid, f"EXTREME_SMALLEST_{rank}"))
            seen.add(cid)
    for rank, cid in enumerate(largest_3, start=1):
        if cid not in seen:
            tagged_cases.append((cid, f"EXTREME_LARGEST_{rank}"))
            seen.add(cid)
    for cid in random_20:
        if cid not in seen:
            tagged_cases.append((cid, "RANDOM"))
            seen.add(cid)

    saved_files: list[str] = []
    for cid, tag in tagged_cases:
        r = rows_by_id[cid]
        ct_slice = np.load(str(slices_dir / f"{cid}_hu.npy"))
        mask_slice = np.load(str(qa_masks_dir / f"{cid}_mask2d.npy"))
        out_file = qa_figures_dir / f"{tag}_{cid}_z{r['z_star']}.png"
        title = (
            f"[{tag}] {cid} | AUA={r['aua_risk_group']} | z*={r['z_star']}/{r['n_slices']} "
            f"| tumor={r['tumor_pixels_at_z_star']} px ({r['tumor_area_mm2']:.1f} mm^2) "
            f"| rad_size={r['radiographic_size_cm']} cm"
        )
        save_qa_overlay(
            ct_slice,
            mask_slice,
            out_file,
            title=title,
            hu_min=hu_min,
            hu_max=hu_max,
            kidney_label=kidney_label,
            tumor_label=tumor_label,
            cyst_label=cyst_label,
        )
        saved_files.append(str(out_file.relative_to(REPO_ROOT)).replace("\\", "/"))

    # Build a compact visual contact sheet of extreme cases + first 12 random cases for quick inspection
    grid_files = [REPO_ROOT / f for f in saved_files[:18]]
    if grid_files:
        ncols = 3
        nrows = (len(grid_files) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.2 * nrows), dpi=110)
        axes_flat = np.atleast_1d(axes).ravel()
        for ax, img_path in zip(axes_flat, grid_files):
            ax.imshow(Image.open(img_path))
            ax.axis("off")
        for ax in axes_flat[len(grid_files) :]:
            ax.axis("off")
        fig.tight_layout(pad=0.3)
        contact_path = qa_figures_dir / "QA_CONTACT_SHEET.png"
        fig.savefig(str(contact_path), bbox_inches="tight")
        plt.close(fig)
        saved_files.append(str(contact_path.relative_to(REPO_ROOT)).replace("\\", "/"))

    return {
        "missing_tumor_cases": missing_tumor,
        "smallest_tumor_cases": smallest_3,
        "largest_tumor_cases": largest_3,
        "random_sample_cases": random_20,
        "total_overlays_saved": len(tagged_cases),
        "saved_files": saved_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract largest-tumor axial slices for all 300 KNIGHT patients.")
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--workers", type=int, default=4, help="Parallel download/extract workers (keep small for disk safety).")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for testing (0 = all 300 KNIGHT cases).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))

    records = sorted(
        load_knight_records(REPO_ROOT / cfg["paths"]["knight_json"]),
        key=lambda r: str(r["case_id"]),
    )
    if args.limit > 0:
        records = records[: args.limit]

    seg_dir = REPO_ROOT / cfg["paths"]["kits23_seg_dir"]
    tmp_ct_dir = REPO_ROOT / cfg["paths"]["tmp_ct_dir"]
    slices_dir = REPO_ROOT / cfg["paths"]["slices_dir"]
    qa_masks_dir = REPO_ROOT / cfg["paths"]["qa_masks_dir"]
    qa_figures_dir = REPO_ROOT / cfg["paths"]["qa_figures_dir"]
    csv_out = REPO_ROOT / cfg["paths"]["slice_index_csv"]
    report_out = REPO_ROOT / cfg["paths"]["reports_dir"] / "phase3_slice_extraction.json"

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_one_patient, rec, cfg, seg_dir, tmp_ct_dir, slices_dir, qa_masks_dir
            ): rec["case_id"]
            for rec in records
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting KNIGHT slices"):
            cid = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:
                errors.append({"case_id": str(cid), "error": str(exc)})
                print(f"\n[ERROR] {cid}: {exc}", file=sys.stderr)

    rows.sort(key=lambda r: r["case_id"])
    if errors:
        print(f"\nCompleted with {len(errors)} errors! Inspect errors before continuing.", file=sys.stderr)
        sys.exit(1)

    # Save slice_index.csv
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    qa_summary = generate_qa_overlays(rows, cfg, slices_dir, qa_masks_dir, qa_figures_dir)

    summary = {
        "n_processed": len(rows),
        "n_missing_tumor_mask": sum(not r["has_tumor_mask"] for r in rows),
        "n_shape_mismatch": sum(not r["ct_mask_shape_match"] for r in rows),
        "n_affine_mismatch": sum(not r["ct_mask_affine_match"] for r in rows),
        "depth_axes_seen": sorted(set(r["depth_axis"] for r in rows)),
        "axcodes_seen": sorted(set(r["axcodes"] for r in rows)),
        "min_tumor_area_mm2": min(r["tumor_area_mm2"] for r in rows),
        "max_tumor_area_mm2": max(r["tumor_area_mm2"] for r in rows),
        "qa_summary": qa_summary,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with report_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved slice index: {csv_out} ({len(rows)} patients)")
    print(f"Saved Phase 3 summary: {report_out}")
    print(json.dumps({k: v for k, v in summary.items() if k != "qa_summary"}, indent=2))
    print(
        f"QA overlays saved: {qa_summary['total_overlays_saved']} "
        f"(missing_tumor={qa_summary['missing_tumor_cases']}, "
        f"smallest={qa_summary['smallest_tumor_cases']}, "
        f"largest={qa_summary['largest_tumor_cases']})"
    )


if __name__ == "__main__":
    main()
