"""Load one validated configuration for the notebook and command line."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

EXTRACTORS = {"resnet50", "convnext_tiny", "image_size", "hog"}
CLASSIFIERS = {"logistic_regression", "linear_svm"}


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _positive_int(value: object, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {"dataset_name", "dataset_url", "archive_bytes", "archive_sha256",
                "excluded_families", "split_seed", "split_target", "features", "training", "experiments"}
    if required - config.keys():
        raise ValueError(f"missing configuration keys: {sorted(required - config.keys())}")
    if not str(config["dataset_url"]).startswith("https://"):
        raise ValueError("dataset_url must use HTTPS")
    digest = config["archive_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("archive_sha256 must be a lowercase SHA-256 digest")
    _positive_int(config["archive_bytes"], "archive_bytes")
    _positive_int(config["split_seed"], "split_seed", 0)
    targets = config["split_target"]
    if set(targets) != {"train", "val", "test"}:
        raise ValueError("split_target must contain train, val and test")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in targets.values()):
        raise ValueError("split ratios must be positive finite numbers")
    if not math.isclose(sum(targets.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to one")
    features = config["features"]
    _positive_int(features.get("image_size"), "features.image_size", 32)
    _positive_int(features.get("batch_size"), "features.batch_size")
    _positive_int(features.get("num_workers", 0), "features.num_workers", 0)
    if features.get("resize_policy") not in {"direct", "pad"}:
        raise ValueError("resize_policy must be direct or pad")
    if features.get("device", "auto") not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda or mps")
    training = config["training"]
    _positive_int(training.get("seed"), "training.seed", 0)
    _positive_int(training.get("max_iter"), "training.max_iter")
    if training.get("train_unit") not in {"file", "representative"}:
        raise ValueError("train_unit must be file or representative")
    if training.get("class_weight") not in {None, "balanced"}:
        raise ValueError("class_weight must be null or balanced")
    pca = training.get("pca_components")
    if pca is not None and (isinstance(pca, bool) or not isinstance(pca, (int, float)) or pca <= 0 or
                            (isinstance(pca, float) and not 0 < pca < 1)):
        raise ValueError("pca_components must be null, a positive integer or a variance fraction in (0,1)")
    experiments = config["experiments"]
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("at least one experiment is required")
    ids = set()
    for experiment in experiments:
        exp_id = experiment.get("id", "")
        if not isinstance(exp_id, str) or not exp_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in exp_id):
            raise ValueError("experiment id must be a plain nonempty name")
        if exp_id in ids:
            raise ValueError(f"duplicate experiment id: {exp_id}")
        ids.add(exp_id)
        if experiment.get("extractor") not in EXTRACTORS or experiment.get("classifier") not in CLASSIFIERS:
            raise ValueError(f"unsupported extractor/classifier in {exp_id}")
        values = experiment.get("c_values", [])
        if not values or any(isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or c <= 0 for c in values):
            raise ValueError(f"c_values must be positive finite numbers in {exp_id}")
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate C values in {exp_id}")
    for name, default in (("data_dir", "data/images"), ("results_dir", "results"), ("features_dir", "features")):
        path = Path(config.get(name, default))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{name} must remain inside the repository")
    config_hash(config)
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))
