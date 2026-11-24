"""Configuration loading and deterministic seeding utilities."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return cfg


def set_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch (if installed)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
