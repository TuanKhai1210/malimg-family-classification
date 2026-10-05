"""Persist feature matrices with explicit sample alignment metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import validate_sample_ids


def save_feature_set(
    directory: str | Path,
    split: str,
    x: np.ndarray,
    y: np.ndarray,
    sample_ids: list[str],
    metadata: dict[str, Any],
) -> None:
    """Save one split; the caller must supply provenance in metadata."""
    validate_sample_ids(sample_ids)
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(y) != len(sample_ids):
        raise ValueError("X must be 2D and X/y/sample_ids must have matching rows")
    if not np.isfinite(x).all():
        raise ValueError("feature matrix contains non-finite values")
    if not split or any(char in split for char in "/\\."):
        raise ValueError("split must be a plain name")
    required = {"manifest_sha256", "extractor", "weights", "transform"}
    if not required.issubset(metadata) or any(not metadata[key] for key in required):
        raise ValueError(f"metadata requires {sorted(required)}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"X_{split}.npy", x, allow_pickle=False)
    np.save(directory / f"y_{split}.npy", y, allow_pickle=False)
    (directory / f"sample_ids_{split}.json").write_text(
        json.dumps(sample_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stored = dict(metadata, split=split, feature_shape=list(x.shape), feature_dtype=str(x.dtype))
    (directory / f"metadata_{split}.json").write_text(
        json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_feature_set(directory: str | Path, split: str) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Load files written by save_feature_set and recheck row alignment."""
    directory = Path(directory)
    x = np.load(directory / f"X_{split}.npy", allow_pickle=False)
    y = np.load(directory / f"y_{split}.npy", allow_pickle=False)
    sample_ids = json.loads((directory / f"sample_ids_{split}.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / f"metadata_{split}.json").read_text(encoding="utf-8"))
    validate_sample_ids(sample_ids)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(y) != len(sample_ids):
        raise ValueError("stored feature rows do not align")
    if metadata.get("feature_shape") != list(x.shape) or metadata.get("feature_dtype") != str(x.dtype):
        raise ValueError("stored feature metadata differs from files")
    return x, y, sample_ids, metadata
