"""Frozen stratified 5-fold cross-validation split creation, integrity checks, and loading."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from metafusion_knight.data.labels import (
    five_class_to_binary,
    map_binary_label,
    map_five_class_label,
)


def compute_assignments_sha256(patients_map: dict[str, dict[str, Any]]) -> str:
    """Compute a deterministic SHA-256 fingerprint of patient fold & label assignments."""
    canonical_rows = [
        {
            "case_id": cid,
            "aua_risk_group": patients_map[cid]["aua_risk_group"],
            "five_class": int(patients_map[cid]["five_class"]),
            "binary": int(patients_map[cid]["binary"]),
            "val_fold": int(patients_map[cid]["val_fold"]),
        }
        for cid in sorted(patients_map.keys())
    ]
    payload = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_stratified_folds(
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Generate frozen 5-fold stratified splits stratified on the five-class AUA risk label."""
    id_col = str(cfg["data"]["id_column"])
    target_col = str(cfg["data"]["target_column"])
    five_class_map: dict[str, int] = dict(cfg["data"]["five_class_map"])
    binary_map: dict[str, int] = dict(cfg["data"]["binary_map"])

    n_splits = int(cfg["cv"]["n_splits"])
    shuffle = bool(cfg["cv"]["shuffle"])
    seed = int(cfg["cv"]["seed"])
    stratify_on = str(cfg["cv"]["stratify_on"])
    if stratify_on != "five_class":
        raise ValueError(
            f"Standing Rule 5 requires stratification on 'five_class', got {stratify_on!r}."
        )

    # Sort deterministically by case_id so record order in JSON can never alter folds.
    sorted_records = sorted(records, key=lambda r: str(r[id_col]))
    case_ids: list[str] = []
    raw_labels: list[str] = []
    five_labels: list[int] = []
    bin_labels: list[int] = []

    for rec in sorted_records:
        cid = str(rec[id_col])
        raw = rec.get(target_col)
        y5 = map_five_class_label(raw, five_class_map)
        y2 = map_binary_label(raw, binary_map)
        if five_class_to_binary(y5) != y2:
            raise ValueError(
                f"Inconsistent binary/five-class mapping for {cid}: "
                f"raw={raw!r}, five_class={y5}, binary={y2}"
            )
        case_ids.append(cid)
        raw_labels.append(str(raw))
        five_labels.append(y5)
        bin_labels.append(y2)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    X_dummy = np.zeros((len(case_ids), 1), dtype=np.int8)
    y_strat = np.asarray(five_labels, dtype=np.int64)

    val_fold_for_patient: dict[str, int] = {}
    folds_dict: dict[str, dict[str, Any]] = {}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dummy, y_strat)):
        train_ids = sorted(case_ids[i] for i in train_idx)
        val_ids = sorted(case_ids[i] for i in val_idx)

        for cid in val_ids:
            if cid in val_fold_for_patient:
                raise RuntimeError(f"Patient {cid} assigned to multiple validation folds.")
            val_fold_for_patient[cid] = fold_idx

        train_y5 = [five_labels[i] for i in train_idx]
        val_y5 = [five_labels[i] for i in val_idx]
        train_y2 = [bin_labels[i] for i in train_idx]
        val_y2 = [bin_labels[i] for i in val_idx]

        folds_dict[str(fold_idx)] = {
            "train_ids": train_ids,
            "val_ids": val_ids,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "train_five_class_counts": {str(k): int(v) for k, v in sorted(Counter(train_y5).items())},
            "val_five_class_counts": {str(k): int(v) for k, v in sorted(Counter(val_y5).items())},
            "train_binary_counts": {str(k): int(v) for k, v in sorted(Counter(train_y2).items())},
            "val_binary_counts": {str(k): int(v) for k, v in sorted(Counter(val_y2).items())},
        }

    patients_map: dict[str, dict[str, Any]] = {}
    for cid, raw, y5, y2 in zip(case_ids, raw_labels, five_labels, bin_labels):
        patients_map[cid] = {
            "aua_risk_group": raw,
            "five_class": int(y5),
            "binary": int(y2),
            "val_fold": int(val_fold_for_patient[cid]),
        }

    fingerprint = compute_assignments_sha256(patients_map)

    splits_data = {
        "metadata": {
            "dataset": "KNIGHT",
            "n_patients": len(case_ids),
            "n_splits": n_splits,
            "shuffle": shuffle,
            "seed": seed,
            "stratify_on": stratify_on,
            "five_class_map": five_class_map,
            "binary_map": binary_map,
            "sha256_patient_assignments": fingerprint,
        },
        "patients": patients_map,
        "folds": folds_dict,
    }
    validate_splits_data(splits_data)
    return splits_data


def validate_splits_data(splits_data: dict[str, Any]) -> None:
    """Strictly verify fold disjointness, completeness, label consistency, and SHA-256 hash."""
    meta = splits_data["metadata"]
    patients = splits_data["patients"]
    folds = splits_data["folds"]

    n_splits = int(meta["n_splits"])
    n_patients = int(meta["n_patients"])
    if len(patients) != n_patients:
        raise ValueError(f"Expected {n_patients} patients, found {len(patients)}.")
    if len(folds) != n_splits:
        raise ValueError(f"Expected {n_splits} folds, found {len(folds)}.")

    expected_hash = meta["sha256_patient_assignments"]
    actual_hash = compute_assignments_sha256(patients)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Frozen folds checksum mismatch! Expected {expected_hash}, got {actual_hash}."
        )

    all_patient_ids = set(patients.keys())
    all_val_seen: list[str] = []

    for fold_idx in range(n_splits):
        key = str(fold_idx)
        if key not in folds:
            raise ValueError(f"Missing fold key {key!r}.")
        fold = folds[key]
        train_set = set(fold["train_ids"])
        val_set = set(fold["val_ids"])

        if len(train_set) != len(fold["train_ids"]):
            raise ValueError(f"Duplicate patient in train_ids for fold {fold_idx}.")
        if len(val_set) != len(fold["val_ids"]):
            raise ValueError(f"Duplicate patient in val_ids for fold {fold_idx}.")
        if train_set & val_set:
            raise ValueError(f"Leakage: overlap between train and val in fold {fold_idx}.")
        if (train_set | val_set) != all_patient_ids:
            raise ValueError(f"Fold {fold_idx} train+val union does not match all patients.")
        all_val_seen.extend(fold["val_ids"])

    if len(all_val_seen) != n_patients or set(all_val_seen) != all_patient_ids:
        raise ValueError("Validation folds do not form an exact partition of all patients.")

    for cid, pdata in patients.items():
        if five_class_to_binary(int(pdata["five_class"])) != int(pdata["binary"]):
            raise ValueError(f"Inconsistent binary/five-class label for {cid}.")


def save_frozen_folds(
    splits_data: dict[str, Any],
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Write frozen folds JSON to disk, refusing to overwrite unless explicitly requested."""
    validate_splits_data(splits_data)
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Frozen splits file already exists at {path}. "
            "Pass --overwrite only if you intentionally want to regenerate splits."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)
    return path


def load_frozen_folds(splits_path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen folds JSON file."""
    path = Path(splits_path)
    if not path.exists():
        raise FileNotFoundError(f"Frozen splits file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        splits_data = json.load(f)
    validate_splits_data(splits_data)
    return splits_data


def get_fold_split(
    splits_data: dict[str, Any],
    fold_idx: int,
    task: str,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Return (train_ids, train_y, val_ids, val_y) for the requested fold and task.

    Guarantees identical train_ids and val_ids across all models and both tasks.
    """
    if task not in ("binary", "five_class"):
        raise ValueError(f"Invalid task {task!r}. Expected 'binary' or 'five_class'.")
    key = str(fold_idx)
    if key not in splits_data["folds"]:
        raise KeyError(f"Fold {fold_idx} not found in frozen splits.")

    fold = splits_data["folds"][key]
    patients = splits_data["patients"]

    train_ids = list(fold["train_ids"])
    val_ids = list(fold["val_ids"])
    train_y = np.asarray([int(patients[cid][task]) for cid in train_ids], dtype=np.int64)
    val_y = np.asarray([int(patients[cid][task]) for cid in val_ids], dtype=np.int64)
    return train_ids, train_y, val_ids, val_y
