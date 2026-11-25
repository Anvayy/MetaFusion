"""Label loading and verified binary / five-class mappings for KNIGHT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_knight_records(knight_json_path: str | Path) -> list[dict[str, Any]]:
    """Load and validate raw KNIGHT patient records from knight.json."""
    path = Path(knight_json_path)
    if not path.exists():
        raise FileNotFoundError(f"KNIGHT metadata file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(f"Expected non-empty JSON list in {path}")

    seen_ids: set[str] = set()
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"Record at index {idx} is not a JSON object.")
        cid = rec.get("case_id")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError(f"Record at index {idx} is missing a valid case_id.")
        if cid in seen_ids:
            raise ValueError(f"Duplicate case_id detected in {path}: {cid}")
        seen_ids.add(cid)

    return records


def map_five_class_label(raw_label: Any, five_class_map: dict[str, int]) -> int:
    """Map raw `aua_risk_group` string to five-class integer ID {0..4}."""
    if not isinstance(raw_label, str) or raw_label not in five_class_map:
        raise ValueError(
            f"Unexpected aua_risk_group value: {raw_label!r}. "
            f"Expected one of {sorted(five_class_map.keys())}."
        )
    return int(five_class_map[raw_label])


def map_binary_label(raw_label: Any, binary_map: dict[str, int]) -> int:
    """Map raw `aua_risk_group` string to binary integer ID {0, 1}."""
    if not isinstance(raw_label, str) or raw_label not in binary_map:
        raise ValueError(
            f"Unexpected aua_risk_group value: {raw_label!r}. "
            f"Expected one of {sorted(binary_map.keys())}."
        )
    return int(binary_map[raw_label])


def five_class_to_binary(five_class_id: int) -> int:
    """Convert five-class ID {0,1,2,3,4} to binary risk ID ({0,1,2}->0, {3,4}->1)."""
    if five_class_id in (0, 1, 2):
        return 0
    if five_class_id in (3, 4):
        return 1
    raise ValueError(f"Invalid five_class_id: {five_class_id!r}. Expected 0..4.")
