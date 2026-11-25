"""Phase 1: Reproducible inspection of KNIGHT clinical metadata and local imaging/mask files.

Reads configuration from YAML (default: configs/base.yaml), inspects knight.json thoroughly,
checks for any local CT/segmentation NIfTI files (without downloading anything), and writes
both JSON and Markdown reports to outputs/reports/.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Allow running directly as `python scripts/00_inspect_knight.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metafusion_knight.config import load_config, set_seed


def try_parse_float(value: Any) -> tuple[bool, float | None]:
    """Return (is_numeric, parsed_float) excluding booleans."""
    if isinstance(value, bool) or value is None:
        return False, None
    if isinstance(value, (int, float)):
        if np.isnan(value):
            return False, None
        return True, float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return False, None
        try:
            f = float(s)
            if np.isnan(f):
                return False, None
            return True, f
        except ValueError:
            return False, None
    return False, None


def summarize_scalar_values(values: list[Any]) -> dict[str, Any]:
    """Summarize types, missingness, numeric stats, and non-numeric values for a field."""
    n = len(values)
    py_types = Counter(type(v).__name__ for v in values)
    null_count = sum(v is None for v in values)

    numeric_vals: list[float] = []
    non_numeric_non_null: Counter[str] = Counter()
    bool_counts: Counter[str] = Counter()

    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            bool_counts[str(v)] += 1
            continue
        ok, f = try_parse_float(v)
        if ok and f is not None:
            numeric_vals.append(f)
        else:
            non_numeric_non_null[repr(v)] += 1

    summary: dict[str, Any] = {
        "n_total": n,
        "python_types": dict(sorted(py_types.items())),
        "null_count": null_count,
        "null_pct": round(100.0 * null_count / n, 2) if n else 0.0,
        "numeric_parseable_count": len(numeric_vals),
        "non_numeric_non_null_count": sum(non_numeric_non_null.values()),
        "non_numeric_values": dict(non_numeric_non_null.most_common()),
    }
    if bool_counts:
        summary["bool_counts"] = dict(bool_counts)
    if numeric_vals:
        arr = np.asarray(numeric_vals, dtype=float)
        summary["numeric_stats"] = {
            "min": float(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "p75": float(np.percentile(arr, 75)),
            "max": float(np.max(arr)),
        }
    return summary


def inspect_local_nifti(imaging_dir: Path, seg_dir: Path) -> dict[str, Any]:
    """Check locally available CT volumes and segmentation masks without downloading."""
    imaging_files = sorted(imaging_dir.glob("case_*/imaging.nii.gz")) if imaging_dir.exists() else []
    seg_candidates: list[Path] = []
    for search_root in [imaging_dir, seg_dir]:
        if search_root.exists():
            seg_candidates.extend(sorted(search_root.rglob("*segmentation*.nii.gz")))
            seg_candidates.extend(sorted(search_root.rglob("*aggregated*seg*.nii.gz")))
    # Deduplicate preserving order
    seen: set[Path] = set()
    seg_files: list[Path] = []
    for p in seg_candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            seg_files.append(p)

    info: dict[str, Any] = {
        "imaging_dir_checked": str(imaging_dir),
        "seg_dir_checked": str(seg_dir),
        "local_imaging_count": len(imaging_files),
        "local_segmentation_count": len(seg_files),
        "sample_segmentation_inspection": [],
    }

    if seg_files:
        import nibabel as nib

        for p in seg_files[:5]:
            img = nib.load(str(p))
            arr = np.asarray(img.dataobj)
            uniq, counts = np.unique(arr, return_counts=True)
            info["sample_segmentation_inspection"].append(
                {
                    "path": str(p),
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "axcodes": list(nib.aff2axcodes(img.affine)),
                    "unique_labels": {str(int(u)): int(c) for u, c in zip(uniq, counts)},
                }
            )
    return info


def build_report(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run full Phase 1 inspection of knight.json and local files."""
    knight_json_path = REPO_ROOT / cfg["paths"]["knight_json"]
    if not knight_json_path.exists():
        raise FileNotFoundError(f"knight.json not found at {knight_json_path}")

    with knight_json_path.open("r", encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)

    n_cases = len(records)
    id_col = cfg["data"]["id_column"]
    target_col = cfg["data"]["target_column"]
    permitted = list(cfg["data"]["permitted_clinical_features"])
    five_map: dict[str, int] = cfg["data"]["five_class_map"]
    bin_map: dict[str, int] = cfg["data"]["binary_map"]

    case_ids = [r.get(id_col) for r in records]
    unique_case_ids = len(set(case_ids))

    # Top-level schema across all records
    all_keys: list[str] = []
    key_set: set[str] = set()
    for r in records:
        for k in r.keys():
            if k not in key_set:
                key_set.add(k)
                all_keys.append(k)

    key_presence = {k: sum(k in r for r in records) for k in all_keys}
    top_level_schema: dict[str, Any] = {}
    for k in all_keys:
        vals = [r.get(k) for r in records]
        role = (
            "identifier"
            if k == id_col
            else "target"
            if k == target_col
            else "permitted_clinical"
            if k in permitted
            else "FORBIDDEN"
        )
        py_types = dict(sorted(Counter(type(v).__name__ for v in vals).items()))
        null_count = sum(v is None for v in vals)
        top_level_schema[k] = {
            "role": role,
            "present_in_cases": key_presence[k],
            "python_types": py_types,
            "null_count": null_count,
            "null_pct": round(100.0 * null_count / n_cases, 2),
        }

    # Permitted feature presence check
    missing_permitted = [f for f in permitted if f not in key_set]

    # Target distribution
    raw_targets = [r.get(target_col) for r in records]
    target_counts = Counter(raw_targets)
    unexpected_targets = [t for t in target_counts if t not in five_map]

    five_class_dist = {
        label: {
            "id": five_map[label],
            "count": target_counts.get(label, 0),
            "pct": round(100.0 * target_counts.get(label, 0) / n_cases, 2),
        }
        for label in five_map
    }
    binary_counts = Counter(bin_map[t] for t in raw_targets if t in bin_map)
    binary_dist = {
        "0_non_high_risk (benign/low/intermediate)": {
            "count": binary_counts.get(0, 0),
            "pct": round(100.0 * binary_counts.get(0, 0) / n_cases, 2),
        },
        "1_high_or_very_high_risk": {
            "count": binary_counts.get(1, 0),
            "pct": round(100.0 * binary_counts.get(1, 0) / n_cases, 2),
        },
    }

    # Detailed inspection of all 12 permitted clinical features (including nested dicts)
    permitted_details: dict[str, Any] = {}
    for feat in permitted:
        vals = [r.get(feat) for r in records]
        non_null_vals = [v for v in vals if v is not None]
        is_nested_dict = any(isinstance(v, dict) for v in non_null_vals)

        if not is_nested_dict:
            permitted_details[feat] = {
                "structure": "scalar",
                **summarize_scalar_values(vals),
            }
        else:
            subkeys: list[str] = []
            subkey_set: set[str] = set()
            for v in non_null_vals:
                if isinstance(v, dict):
                    for sk in v.keys():
                        if sk not in subkey_set:
                            subkey_set.add(sk)
                            subkeys.append(sk)

            sub_details: dict[str, Any] = {}
            for sk in subkeys:
                sub_vals = [v.get(sk) if isinstance(v, dict) else None for v in vals]
                sub_details[sk] = summarize_scalar_values(sub_vals)

            extra: dict[str, Any] = {
                "structure": "nested_dict",
                "outer_null_count": sum(v is None for v in vals),
                "subkey_count": len(subkeys),
                "subkeys": sub_details,
            }
            if feat == "comorbidities":
                counts_per_patient = [
                    sum(1 for x in v.values() if x is True) if isinstance(v, dict) else 0
                    for v in vals
                ]
                extra["true_count_per_patient_distribution"] = dict(
                    sorted(Counter(counts_per_patient).items())
                )
            permitted_details[feat] = extra

    # Cross-checks / edge cases
    edge_cases: dict[str, Any] = {}
    smoking_vs_quit: dict[str, Counter[str]] = defaultdict(Counter)
    smoking_vs_pack: dict[str, Counter[str]] = defaultdict(Counter)
    for r in records:
        sh = str(r.get("smoking_history"))
        aq = r.get("age_when_quit_smoking")
        py = r.get("pack_years")
        ok_aq, _ = try_parse_float(aq)
        ok_py, _ = try_parse_float(py)
        smoking_vs_quit[sh]["numeric" if ok_aq else repr(aq)] += 1
        smoking_vs_pack[sh]["numeric" if ok_py else repr(py)] += 1
    edge_cases["smoking_history_vs_age_when_quit"] = {
        k: dict(v) for k, v in smoking_vs_quit.items()
    }
    edge_cases["smoking_history_vs_pack_years"] = {
        k: dict(v) for k, v in smoking_vs_pack.items()
    }

    local_nifti = inspect_local_nifti(
        REPO_ROOT / cfg["paths"]["knight_imaging_dir"],
        REPO_ROOT / cfg["paths"].get("kits23_seg_dir", "data/raw/kits23_segmentations"),
    )

    return {
        "source_file": str(knight_json_path),
        "n_cases": n_cases,
        "unique_case_ids": unique_case_ids,
        "first_case_id": case_ids[0] if case_ids else None,
        "last_case_id": case_ids[-1] if case_ids else None,
        "total_top_level_fields": len(all_keys),
        "permitted_feature_count": len(permitted),
        "missing_permitted_features": missing_permitted,
        "forbidden_field_count": sum(
            1 for v in top_level_schema.values() if v["role"] == "FORBIDDEN"
        ),
        "target_column": target_col,
        "unexpected_target_values": unexpected_targets,
        "five_class_distribution": five_class_dist,
        "binary_distribution": binary_dist,
        "top_level_schema": top_level_schema,
        "permitted_feature_details": permitted_details,
        "edge_cases": edge_cases,
        "local_nifti_status": local_nifti,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a concise, readable Phase 1 Markdown report."""
    lines: list[str] = []
    lines.append("# Phase 1: KNIGHT Dataset Inspection Report\n")
    lines.append(f"- **Source file:** `{report['source_file']}`")
    lines.append(
        f"- **Patients:** `{report['n_cases']}` total (`{report['unique_case_ids']}` unique `case_id`s, "
        f"`{report['first_case_id']}` to `{report['last_case_id']}`)"
    )
    lines.append(
        f"- **Top-level JSON fields:** `{report['total_top_level_fields']}` total "
        f"(`1` identifier, `1` target, `{report['permitted_feature_count']}` permitted clinical, "
        f"`{report['forbidden_field_count']}` forbidden)"
    )
    lines.append(
        f"- **Unexpected target values:** `{report['unexpected_target_values']}`\n"
    )

    lines.append("## 1. Target Distribution (`aua_risk_group`)\n")
    lines.append("### Five-Class Task")
    lines.append("| Raw Label | Class ID | Count | Percentage |")
    lines.append("| :--- | :---: | :---: | :---: |")
    for label, info in report["five_class_distribution"].items():
        lines.append(f"| `{label}` | {info['id']} | {info['count']} | {info['pct']:.2f}% |")
    lines.append("")

    lines.append("### Binary Task")
    lines.append("| Binary Class | Count | Percentage |")
    lines.append("| :--- | :---: | :---: |")
    for label, info in report["binary_distribution"].items():
        lines.append(f"| `{label}` | {info['count']} | {info['pct']:.2f}% |")
    lines.append("")

    lines.append("## 2. Detailed Inspection of the 12 Permitted Clinical Features\n")
    lines.append("| # | Feature | Structure | Python Types | Nulls (%) | Numeric Parseable | Non-Numeric / Categories |")
    lines.append("| :---: | :--- | :--- | :--- | :---: | :---: | :--- |")
    for idx, (feat, d) in enumerate(report["permitted_feature_details"].items(), start=1):
        if d["structure"] == "scalar":
            non_num = d["non_numeric_values"]
            non_num_str = ", ".join(f"{k}: {v}" for k, v in non_num.items()) if non_num else "—"
            types_str = ", ".join(f"{k}({v})" for k, v in d["python_types"].items())
            lines.append(
                f"| {idx} | `{feat}` | scalar | `{types_str}` | "
                f"{d['null_count']} ({d['null_pct']:.1f}%) | {d['numeric_parseable_count']} | {non_num_str} |"
            )
        else:
            lines.append(
                f"| {idx} | `{feat}` | nested_dict ({d['subkey_count']} keys) | `dict` | "
                f"{d['outer_null_count']} (0.0%) | see subkeys below | see subkeys below |"
            )
    lines.append("")

    lines.append("### Numeric Summary (Scalar Permitted Features)\n")
    lines.append("| Feature | Valid N | Min | P25 | Median | Mean | Std | P75 | Max |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for feat, d in report["permitted_feature_details"].items():
        if d["structure"] == "scalar" and "numeric_stats" in d:
            s = d["numeric_stats"]
            lines.append(
                f"| `{feat}` | {d['numeric_parseable_count']} | {s['min']} | {s['p25']} | "
                f"{s['median']} | {s['mean']} | {s['std']} | {s['p75']} | {s['max']} |"
            )
    lines.append("")

    lines.append("### Nested Dict Sub-Fields\n")
    for feat, d in report["permitted_feature_details"].items():
        if d["structure"] != "nested_dict":
            continue
        lines.append(f"#### `{feat}` ({d['subkey_count']} sub-keys)")
        lines.append("| Sub-key | Python Types | Nulls (%) | Summary |")
        lines.append("| :--- | :--- | :---: | :--- |")
        for sk, sd in d["subkeys"].items():
            types_str = ", ".join(f"{k}({v})" for k, v in sd["python_types"].items())
            if "bool_counts" in sd:
                t_cnt = sd["bool_counts"].get("True", 0)
                f_cnt = sd["bool_counts"].get("False", 0)
                summ = f"True={t_cnt} ({100.0 * t_cnt / sd['n_total']:.1f}%), False={f_cnt}"
            elif "numeric_stats" in sd:
                s = sd["numeric_stats"]
                summ = (
                    f"num={sd['numeric_parseable_count']} "
                    f"(min={s['min']}, med={s['median']}, max={s['max']})"
                )
                if sd["non_numeric_values"]:
                    summ += f"; non-num={sd['non_numeric_values']}"
            else:
                summ = f"non-num={sd['non_numeric_values']}"
            lines.append(
                f"| `{sk}` | `{types_str}` | {sd['null_count']} ({sd['null_pct']:.1f}%) | {summ} |"
            )
        if "true_count_per_patient_distribution" in d:
            lines.append(
                f"\n- Active comorbidities per patient distribution: "
                f"`{d['true_count_per_patient_distribution']}`"
            )
        lines.append("")

    lines.append("## 3. Full Top-Level Schema & Forbidden Fields\n")
    lines.append("| Field | Role | Present | Types | Nulls (%) |")
    lines.append("| :--- | :--- | :---: | :--- | :---: |")
    for k, info in report["top_level_schema"].items():
        types_str = ", ".join(f"{t}({c})" for t, c in info["python_types"].items())
        lines.append(
            f"| `{k}` | `{info['role']}` | {info['present_in_cases']} | `{types_str}` | "
            f"{info['null_count']} ({info['null_pct']:.1f}%) |"
        )
    lines.append("")

    lines.append("## 4. Local CT Imaging & Segmentation Mask Status\n")
    nifti = report["local_nifti_status"]
    lines.append(f"- Local CT volumes found (`imaging.nii.gz`): `{nifti['local_imaging_count']}`")
    lines.append(f"- Local segmentation masks found: `{nifti['local_segmentation_count']}`")
    if not nifti["sample_segmentation_inspection"]:
        lines.append(
            "- **Status:** No NIfTI CT volumes or KiTS21 segmentation masks are downloaded locally yet. "
            "`TUMOR_LABEL` and axis orientation cannot be verified on actual NIfTI files until data download is approved."
        )
    else:
        for item in nifti["sample_segmentation_inspection"]:
            lines.append(f"- `{item['path']}`: shape={item['shape']}, axcodes={item['axcodes']}, labels={item['unique_labels']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect KNIGHT dataset metadata and local imaging files.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "base.yaml"),
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))

    report = build_report(cfg)
    reports_dir = REPO_ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_out = reports_dir / "phase1_knight_inspection.json"
    md_out = reports_dir / "phase1_knight_inspection.md"

    with json_out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with md_out.open("w", encoding="utf-8") as f:
        f.write(render_markdown(report))

    print(f"Wrote JSON report: {json_out}")
    print(f"Wrote Markdown report: {md_out}")
    print(f"Total cases: {report['n_cases']} (unique IDs: {report['unique_case_ids']})")
    print("Five-class distribution:")
    for k, v in report["five_class_distribution"].items():
        print(f"  {k:20s} (id={v['id']}): {v['count']:3d} ({v['pct']:5.2f}%)")
    print("Binary distribution:")
    for k, v in report["binary_distribution"].items():
        print(f"  {k:45s}: {v['count']:3d} ({v['pct']:5.2f}%)")
    print(
        f"Local CT volumes: {report['local_nifti_status']['local_imaging_count']}, "
        f"Local masks: {report['local_nifti_status']['local_segmentation_count']}"
    )


if __name__ == "__main__":
    main()
