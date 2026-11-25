"""Data loading, label mapping, frozen splits, slice selection, and metadata pipelines."""

from metafusion_knight.data.labels import (
    five_class_to_binary,
    load_knight_records,
    map_binary_label,
    map_five_class_label,
)
from metafusion_knight.data.splits import (
    compute_assignments_sha256,
    create_stratified_folds,
    get_fold_split,
    load_frozen_folds,
    save_frozen_folds,
    validate_splits_data,
)

__all__ = [
    "compute_assignments_sha256",
    "create_stratified_folds",
    "five_class_to_binary",
    "get_fold_split",
    "load_frozen_folds",
    "load_knight_records",
    "map_binary_label",
    "map_five_class_label",
    "save_frozen_folds",
    "validate_splits_data",
]
