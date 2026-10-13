"""Persist aligned feature matrices with checksums and atomic file replacement."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import validate_sample_ids


FEATURE_SCHEMA_VERSION = 2


def file_sha256(path: str | Path) -> str:
    """Hash file bytes without loading a feature matrix into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_split(split: str) -> None:
    if not split or any(char in split for char in "/\\."):
        raise ValueError("split must be a plain name")


def _aligned(x: np.ndarray, y: np.ndarray, sample_ids: list[str]) -> None:
    if not isinstance(sample_ids, list) or any(not isinstance(value, str) for value in sample_ids):
        raise ValueError("sample_ids must be a list of strings")
    validate_sample_ids(sample_ids)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(y) != len(sample_ids):
        raise ValueError("X must be 2D and X/y/sample_ids must have matching rows")
    if x.shape[1] < 1 or not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError("feature matrix must contain finite numeric vectors")
    if not np.issubdtype(y.dtype, np.integer) or (y < 0).any():
        raise ValueError("labels must be nonnegative integers")


def atomic_json(path: str | Path, value: Any) -> None:
    """Publish complete JSON by replacing a temporary file in the same directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_feature_set(
    directory: str | Path,
    split: str,
    x: np.ndarray,
    y: np.ndarray,
    sample_ids: list[str],
    metadata: dict[str, Any],
) -> None:
    """Save one split, writing its integrity metadata only after all data files.

    The public API is unchanged. Schema 2 metadata adds checksums so a partially
    replaced, reordered or modified feature artifact is rejected when loaded.
    """
    _plain_split(split)
    x = np.asarray(x)
    y = np.asarray(y)
    _aligned(x, y, sample_ids)
    required = {"manifest_sha256", "extractor", "weights", "transform"}
    if not required.issubset(metadata) or any(not metadata[key] for key in required):
        raise ValueError(f"metadata requires {sorted(required)}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / f"metadata_{split}.json"
    # Invalidate the previous transaction before replacing its first file.
    metadata_path.unlink(missing_ok=True)
    _atomic_npy(directory / f"X_{split}.npy", x)
    _atomic_npy(directory / f"y_{split}.npy", y)
    atomic_json(directory / f"sample_ids_{split}.json", sample_ids)
    stored = dict(
        metadata,
        schema_version=FEATURE_SCHEMA_VERSION,
        split=split,
        feature_shape=list(x.shape),
        feature_dtype=str(x.dtype),
        label_dtype=str(y.dtype),
        sample_count=len(sample_ids),
        file_sha256={
            "X": file_sha256(directory / f"X_{split}.npy"),
            "y": file_sha256(directory / f"y_{split}.npy"),
            "sample_ids": file_sha256(directory / f"sample_ids_{split}.json"),
        },
    )
    atomic_json(metadata_path, stored)


def load_feature_set(
    directory: str | Path, split: str
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Load features and verify checksums, numeric validity and row alignment."""
    _plain_split(split)
    directory = Path(directory)
    metadata = json.loads((directory / f"metadata_{split}.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("stored feature metadata must be an object")
    if metadata.get("split") != split:
        raise ValueError("stored feature split differs from requested split")
    version = metadata.get("schema_version", 1)
    if version not in (1, FEATURE_SCHEMA_VERSION):
        raise ValueError(f"unsupported feature schema: {version}")
    if version == FEATURE_SCHEMA_VERSION:
        expected = metadata.get("file_sha256")
        if not isinstance(expected, dict) or set(expected) != {"X", "y", "sample_ids"}:
            raise ValueError("stored feature checksums are missing")
        paths = {"X": f"X_{split}.npy", "y": f"y_{split}.npy", "sample_ids": f"sample_ids_{split}.json"}
        for name, filename in paths.items():
            if expected[name] != file_sha256(directory / filename):
                raise ValueError(f"stored {name} checksum differs for {split}")
    x = np.load(directory / f"X_{split}.npy", allow_pickle=False)
    y = np.load(directory / f"y_{split}.npy", allow_pickle=False)
    sample_ids = json.loads((directory / f"sample_ids_{split}.json").read_text(encoding="utf-8"))
    _aligned(x, y, sample_ids)
    if metadata.get("feature_shape") != list(x.shape) or metadata.get("feature_dtype") != str(x.dtype):
        raise ValueError("stored feature metadata differs from files")
    if version == FEATURE_SCHEMA_VERSION:
        if metadata.get("label_dtype") != str(y.dtype) or metadata.get("sample_count") != len(sample_ids):
            raise ValueError("stored label or sample metadata differs from files")
    return x, y, sample_ids, metadata
