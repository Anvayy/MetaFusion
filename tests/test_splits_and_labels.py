"""Phase 2 unit tests: label mapping correctness and frozen 5-fold split invariants."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from metafusion_knight.config import load_config
from metafusion_knight.data.labels import (
    five_class_to_binary,
    load_knight_records,
    map_binary_label,
    map_five_class_label,
)
from metafusion_knight.data.splits import (
    create_stratified_folds,
    get_fold_split,
    load_frozen_folds,
    validate_splits_data,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(BASE_CONFIG_PATH)


@pytest.fixture(scope="module")
def records(cfg: dict) -> list[dict]:
    return load_knight_records(REPO_ROOT / cfg["paths"]["knight_json"])


@pytest.fixture(scope="module")
def frozen_splits(cfg: dict) -> dict:
    return load_frozen_folds(REPO_ROOT / cfg["paths"]["splits_file"])


def test_label_mapping_exact_definitions(cfg: dict) -> None:
    five_map = cfg["data"]["five_class_map"]
    bin_map = cfg["data"]["binary_map"]

    expected_five = {
        "benign": 0,
        "low_risk": 1,
        "intermediate_risk": 2,
        "high_risk": 3,
        "very_high_risk": 4,
    }
    expected_bin = {
        "benign": 0,
        "low_risk": 0,
        "intermediate_risk": 0,
        "high_risk": 1,
        "very_high_risk": 1,
    }

    for raw, exp5 in expected_five.items():
        y5 = map_five_class_label(raw, five_map)
        y2 = map_binary_label(raw, bin_map)
        assert y5 == exp5
        assert y2 == expected_bin[raw]
        assert five_class_to_binary(y5) == y2


def test_label_mapping_rejects_invalid_values(cfg: dict) -> None:
    five_map = cfg["data"]["five_class_map"]
    bin_map = cfg["data"]["binary_map"]

    for bad in [None, "", "unknown_risk", "High Risk", 3]:
        with pytest.raises(ValueError):
            map_five_class_label(bad, five_map)
        with pytest.raises(ValueError):
            map_binary_label(bad, bin_map)

    for bad_id in [-1, 5, 99]:
        with pytest.raises(ValueError):
            five_class_to_binary(bad_id)


def test_all_knight_records_map_consistently(cfg: dict, records: list[dict]) -> None:
    five_map = cfg["data"]["five_class_map"]
    bin_map = cfg["data"]["binary_map"]
    target_col = cfg["data"]["target_column"]

    assert len(records) == 300
    five_counts = Counter()
    bin_counts = Counter()

    for rec in records:
        raw = rec[target_col]
        y5 = map_five_class_label(raw, five_map)
        y2 = map_binary_label(raw, bin_map)
        assert five_class_to_binary(y5) == y2
        five_counts[y5] += 1
        bin_counts[y2] += 1

    assert dict(sorted(five_counts.items())) == {0: 25, 1: 142, 2: 48, 3: 40, 4: 45}
    assert dict(sorted(bin_counts.items())) == {0: 215, 1: 85}


def test_frozen_folds_disjoint_and_complete(frozen_splits: dict, records: list[dict]) -> None:
    all_case_ids = {r["case_id"] for r in records}
    n_splits = int(frozen_splits["metadata"]["n_splits"])
    assert n_splits == 5

    all_val_ids: list[str] = []
    for fold_idx in range(n_splits):
        train_ids, train_y, val_ids, val_y = get_fold_split(frozen_splits, fold_idx, "five_class")
        train_set = set(train_ids)
        val_set = set(val_ids)

        # No patient in both train and val
        assert train_set.isdisjoint(val_set), f"Overlap in fold {fold_idx}"
        # No duplicates inside train or val
        assert len(train_set) == len(train_ids)
        assert len(val_set) == len(val_ids)
        # Union equals full KNIGHT cohort
        assert (train_set | val_set) == all_case_ids
        assert len(train_y) == len(train_ids)
        assert len(val_y) == len(val_ids)

        all_val_ids.extend(val_ids)

    # Every patient appears in validation in exactly one fold
    assert len(all_val_ids) == len(all_case_ids)
    assert set(all_val_ids) == all_case_ids


def test_frozen_folds_stratified_for_five_class_and_binary(frozen_splits: dict) -> None:
    total_five = Counter(p["five_class"] for p in frozen_splits["patients"].values())
    total_bin = Counter(p["binary"] for p in frozen_splits["patients"].values())
    n_splits = int(frozen_splits["metadata"]["n_splits"])

    for fold_idx in range(n_splits):
        _, _, _, val_y5 = get_fold_split(frozen_splits, fold_idx, "five_class")
        _, _, _, val_y2 = get_fold_split(frozen_splits, fold_idx, "binary")
        c5 = Counter(val_y5.tolist())
        c2 = Counter(val_y2.tolist())

        # Each five-class count in val must be within floor/ceil of total / 5
        for cls_id, total_cnt in total_five.items():
            lo = total_cnt // n_splits
            hi = (total_cnt + n_splits - 1) // n_splits
            assert lo <= c5[cls_id] <= hi, (
                f"Fold {fold_idx} class {cls_id} count {c5[cls_id]} outside [{lo}, {hi}]"
            )

        # Binary counts should also remain tightly balanced across folds
        for cls_id, total_cnt in total_bin.items():
            lo = total_cnt // n_splits
            hi = (total_cnt + n_splits - 1) // n_splits
            assert lo <= c2[cls_id] <= hi, (
                f"Fold {fold_idx} binary class {cls_id} count {c2[cls_id]} outside [{lo}, {hi}]"
            )


def test_same_folds_reused_across_tasks_and_deterministic(
    cfg: dict, records: list[dict], frozen_splits: dict
) -> None:
    # Identical fold indices across binary and five-class tasks
    for fold_idx in range(5):
        tr_ids_5, tr_y5, va_ids_5, va_y5 = get_fold_split(frozen_splits, fold_idx, "five_class")
        tr_ids_2, tr_y2, va_ids_2, va_y2 = get_fold_split(frozen_splits, fold_idx, "binary")
        assert tr_ids_5 == tr_ids_2
        assert va_ids_5 == va_ids_2
        assert np.array_equal([five_class_to_binary(int(y)) for y in tr_y5], tr_y2)
        assert np.array_equal([five_class_to_binary(int(y)) for y in va_y5], va_y2)

    # Re-running split generation (even on reversed input order) yields identical SHA-256 hash
    recreated = create_stratified_folds(list(reversed(records)), cfg)
    assert (
        recreated["metadata"]["sha256_patient_assignments"]
        == frozen_splits["metadata"]["sha256_patient_assignments"]
    )


def test_checksum_detects_tampering(frozen_splits: dict) -> None:
    tampered = copy.deepcopy(frozen_splits)
    first_cid = next(iter(tampered["patients"]))
    tampered["patients"][first_cid]["val_fold"] = (
        tampered["patients"][first_cid]["val_fold"] + 1
    ) % 5
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_splits_data(tampered)
