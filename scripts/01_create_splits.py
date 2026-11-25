"""Phase 2: Create and freeze stratified 5-fold cross-validation splits (data/splits/folds.json).

Stratifies on the five-class AUA risk label (StratifiedKFold, shuffle=True, fixed seed from YAML).
The resulting frozen folds.json is shared by EVERY model and BOTH tasks (binary & five-class).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metafusion_knight.config import load_config, set_seed
from metafusion_knight.data.labels import load_knight_records
from metafusion_knight.data.splits import create_stratified_folds, save_frozen_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Create frozen 5-fold stratified splits for KNIGHT.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "base.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frozen splits file if present.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["cv"]["seed"]))

    knight_json = REPO_ROOT / cfg["paths"]["knight_json"]
    splits_out = REPO_ROOT / cfg["paths"]["splits_file"]

    records = load_knight_records(knight_json)
    splits_data = create_stratified_folds(records, cfg)
    saved_path = save_frozen_folds(splits_data, splits_out, overwrite=args.overwrite)

    meta = splits_data["metadata"]
    print(f"Saved frozen 5-fold splits: {saved_path}")
    print(
        f"Patients: {meta['n_patients']} | Folds: {meta['n_splits']} | "
        f"Seed: {meta['seed']} | Stratify: {meta['stratify_on']}"
    )
    print(f"SHA-256 fingerprint: {meta['sha256_patient_assignments']}")
    print("\nPer-fold validation class distribution:")
    print(f"{'Fold':<6} {'N_train':<8} {'N_val':<7} {'5-class (0..4)':<28} {'Binary (0, 1)'}")
    for fold_idx in range(int(meta["n_splits"])):
        f = splits_data["folds"][str(fold_idx)]
        c5 = [f["val_five_class_counts"].get(str(c), 0) for c in range(5)]
        c2 = [f["val_binary_counts"].get(str(c), 0) for c in range(2)]
        print(f"{fold_idx:<6} {f['n_train']:<8} {f['n_val']:<7} {str(c5):<28} {str(c2)}")


if __name__ == "__main__":
    main()
