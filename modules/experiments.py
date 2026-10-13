"""Validation selection and one-time final testing of classical classifiers.

Only train rows fit the scaler, optional PCA, and classifier. Validation never
loads the test arrays. Test feature files are hashed as opaque bytes to lock
their identity before the final test. Joblib files produced here are executable
Python serialization: load only this project's locally generated, trusted
models, never a model downloaded from an unknown source. SHA-256 detects changes
but does not authenticate the author of a downloaded model or selection file.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .classifiers import build_classifier
from .contracts import validate_split_manifest
from .evaluation import classification_metrics
from .feature_io import load_feature_set


SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")
FEATURE_FILES = {
    "X": "X_{split}.npy", "y": "y_{split}.npy",
    "sample_ids": "sample_ids_{split}.json", "metadata": "metadata_{split}.json",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _object_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                         + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _read_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["true_label"] = int(row["true_label"])
        row["predicted_label"] = int(row["predicted_label"])
        row["eval_representative"] = _flag(row["eval_representative"])
        row["score_value"] = float(row["score_value"])
    return rows


def _flag(value: Any) -> bool:
    if value is True or value == 1 or str(value).lower() in {"true", "1"}:
        return True
    if value is False or value == 0 or str(value).lower() in {"false", "0", ""}:
        return False
    raise ValueError(f"invalid boolean flag: {value!r}")


def _manifest_rows(rows: list[dict]) -> tuple[list[dict], list[int], dict[int, str]]:
    included = [row for row in rows if _flag(row.get("included", True))]
    validate_split_manifest(included)
    id_names: dict[int, str] = {}
    for row in included:
        label_id = int(row["label_id"])
        if label_id in id_names and id_names[label_id] != row["label"]:
            raise ValueError("different class names share a label_id")
        id_names[label_id] = str(row["label"])
    labels = sorted(id_names)
    if labels != list(range(len(labels))):
        raise ValueError("label IDs must be contiguous from zero")
    if len(labels) < 2:
        raise ValueError("classification needs at least two classes")
    # Representatives are locked by image identity, never predictions.
    for split in SPLITS:
        groups: dict[str, list[dict]] = {}
        for row in included:
            if row["split"] == split:
                groups.setdefault(str(row["pixel_sha256"]), []).append(row)
        for group in groups.values():
            chosen = min(group, key=lambda row: (str(row["relative_path"]), str(row["sample_id"])))
            marked = [row for row in group if _flag(row.get("eval_representative", False))]
            if len(marked) != 1 or marked[0]["sample_id"] != chosen["sample_id"]:
                raise ValueError("manifest needs the deterministic representative for every pixel group")
    return included, labels, id_names


def _split_rows(rows: list[dict], split: str) -> list[dict]:
    return [row for row in rows if row["split"] == split]


def _root_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _reference(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _output_dir(root: Path, config: dict) -> Path:
    return _root_path(root, config.get("results_dir", "results"))


def _archive_sha(config: dict) -> str:
    result = config.get("archive_sha256", config.get("dataset", {}).get("archive_sha256"))
    if not result:
        raise ValueError("config must contain archive_sha256 for data provenance")
    return str(result)


def _provenance(metadata: dict, extractor: str, manifest_sha256: str, config: dict) -> dict:
    required = {"manifest_sha256", "extractor", "weights", "transform", "dataset_archive_sha256"}
    if not required.issubset(metadata) or any(not metadata[key] for key in required):
        raise ValueError(f"feature provenance requires {sorted(required)}")
    if metadata["manifest_sha256"] != manifest_sha256:
        raise ValueError("feature manifest SHA-256 does not match the locked manifest")
    if metadata["extractor"] != extractor:
        raise ValueError("feature extractor does not match the experiment")
    if metadata["dataset_archive_sha256"] != _archive_sha(config):
        raise ValueError("feature dataset archive SHA-256 does not match config")
    if "extraction_signature" in metadata:
        signature = metadata["extraction_signature"]
        if metadata.get("extraction_signature_sha256") != _object_sha(signature):
            raise ValueError("feature extraction signature SHA-256 is invalid")
        declared = signature.get("extractor_provenance", {})
        expected = dict(declared, manifest_sha256=manifest_sha256, dataset_archive_sha256=_archive_sha(config))
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("feature metadata contradicts its extraction signature")
        if signature.get("manifest_sha256") != manifest_sha256 or signature.get("dataset_archive_sha256") != _archive_sha(config):
            raise ValueError("feature extraction signature refers to different source data")
        settings = config.get("features", {})
        if extractor in {"resnet50", "convnext_tiny", "hog"}:
            transform = metadata["transform"]
            size = int(settings.get("image_size", 224))
            if transform.get("image_size") != [size, size]:
                raise ValueError("feature image_size differs from the current config")
            if transform.get("resize_policy") != settings.get("resize_policy", "direct"):
                raise ValueError("feature resize_policy differs from the current config")
        if extractor in {"resnet50", "convnext_tiny"}:
            default_weights = {"resnet50": "IMAGENET1K_V2", "convnext_tiny": "IMAGENET1K_V1"}
            requested = settings.get("weights", {}).get(extractor, default_weights[extractor])
            if str(metadata["weights"]).rsplit(".", 1)[-1] != requested:
                raise ValueError("feature weights differ from the current config")
    keys = sorted(required | ({"extraction_signature_sha256"} if "extraction_signature_sha256" in metadata else set()))
    return {key: metadata[key] for key in keys}


def _feature_signatures(directory: Path) -> dict:
    # This reads bytes, including test bytes, without deserializing test arrays.
    result = {split: {key: _file_sha(directory / pattern.format(split=split))
                      for key, pattern in FEATURE_FILES.items()} for split in SPLITS}
    if (directory / "metadata.json").exists() or (directory / "complete.json").exists():
        result["common"] = {"metadata": _file_sha(directory / "metadata.json"),
                            "completion": _file_sha(directory / "complete.json")}
    return result


def _verify_store_metadata(directory: Path, signatures: dict) -> None:
    """Validate extraction transactions and test checksums without opening arrays."""
    metadata = {split: _read_json(directory / f"metadata_{split}.json") for split in SPLITS}
    for split, item in metadata.items():
        if item.get("schema_version") == 2:
            declared = item.get("file_sha256", {})
            if set(declared) != {"X", "y", "sample_ids"} or any(
                declared[key] != signatures[split][key] for key in declared
            ):
                raise ValueError(f"feature array/ID hashes contradict {split} metadata")
    signed = [item for item in metadata.values() if "extraction_signature" in item]
    if signed:
        if len(signed) != len(SPLITS) or "common" not in signatures:
            raise ValueError("signed feature store requires all splits and a completion marker")
        complete = _read_json(directory / "complete.json")
        common = _read_json(directory / "metadata.json")
        signature = signed[0]["extraction_signature"]
        signature_sha = _object_sha(signature)
        for item in [*signed, common, complete]:
            if item.get("extraction_signature") != signature or item.get("extraction_signature_sha256") != signature_sha:
                raise ValueError("feature store contains inconsistent extraction signatures")
        if complete.get("metadata_sha256") != signatures["common"]["metadata"]:
            raise ValueError("feature common metadata differs from its completion checksum")
        if complete.get("split_metadata_sha256") != {split: signatures[split]["metadata"] for split in SPLITS}:
            raise ValueError("feature split metadata differs from its completion checksums")
        expected_common = dict(signature.get("extractor_provenance", {}),
                               manifest_sha256=signature.get("manifest_sha256"),
                               dataset_archive_sha256=signature.get("dataset_archive_sha256"))
        if any(common.get(key) != value for key, value in expected_common.items()):
            raise ValueError("common feature provenance contradicts its extraction signature")


def _load_aligned(
    directory: Path, split: str, rows: list[dict], extractor: str,
    manifest_sha256: str, config: dict, provenance: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    x, y, ids, metadata = load_feature_set(directory, split)
    expected = _split_rows(rows, split)
    if ids != [str(row["sample_id"]) for row in expected]:
        raise ValueError(f"{extractor}/{split} sample IDs/order do not match manifest")
    if not np.issubdtype(y.dtype, np.integer) or not np.array_equal(
        y, np.asarray([int(row["label_id"]) for row in expected], dtype=np.int64)
    ):
        raise ValueError(f"{extractor}/{split} labels do not match manifest")
    if not np.isfinite(x).all() or not np.issubdtype(x.dtype, np.number):
        raise ValueError("features must be finite numeric values")
    current = _provenance(metadata, extractor, manifest_sha256, config)
    if "extraction_signature" in metadata:
        row_signature = {name: [{key: row.get(key) for key in (
            "sample_id", "relative_path", "label_id", "eval_representative", "pixel_sha256", "file_sha256")}
            for row in _split_rows(rows, name)] for name in SPLITS}
        if metadata["extraction_signature"].get("rows_sha256") != _object_sha(row_signature):
            raise ValueError("feature extraction row signature differs from the manifest")
    if provenance is not None and current != provenance:
        raise ValueError("feature provenance differs between partitions")
    return x, y, ids, current


def _environment() -> dict:
    packages = {}
    for package in ("numpy", "scikit-learn", "joblib", "matplotlib", "torch", "torchvision"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version, "software": packages,
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                     "classifier_device": "CPU"},
    }


def _implementation_signature() -> dict:
    directory = Path(__file__).resolve().parent
    return {name: _file_sha(directory / name) for name in
            ("classifiers.py", "evaluation.py", "experiments.py")}


def _training_options(config: dict) -> dict:
    training = config.get("training", {})
    options = {
        "seed": int(training.get("seed", config.get("split_seed", 42))),
        "max_iter": int(training.get("max_iter", 5000)),
        "class_weight": training.get("class_weight", "balanced"),
        "pca_components": training.get("pca_components"),
    }
    if training.get("train_unit", "file") not in {"file", "representative"}:
        raise ValueError("train_unit must be file or representative")
    return options


def _experiments(config: dict) -> list[dict]:
    experiments = config.get("experiments", [])
    if not experiments or not isinstance(experiments, list):
        raise ValueError("config must define at least one experiment")
    seen = set()
    for experiment in experiments:
        experiment_id = experiment.get("id", "")
        if not experiment_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                                    for character in experiment_id) or experiment_id in seen:
            raise ValueError("experiment IDs must be unique plain names")
        seen.add(experiment_id)
        if experiment.get("classifier") not in {"logistic_regression", "linear_svm"}:
            raise ValueError("only logistic_regression and linear_svm are supported")
        values = experiment.get("c_values", [0.1, 1.0, 10.0])
        if not values or any(isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0
                             for value in values):
            raise ValueError("c_values must be finite positive numbers")
        if len(set(float(value) for value in values)) != len(values):
            raise ValueError("c_values must not repeat")
        if not experiment.get("extractor"):
            raise ValueError("each experiment must name its extractor")
    return experiments


def _prediction_rows(model, x: np.ndarray, y: np.ndarray, ids: list[str], rows: list[dict],
                     experiment_id: str) -> tuple[np.ndarray, list[dict], float]:
    started = time.perf_counter()
    predictions = np.asarray(model.predict(x), dtype=np.int64)
    scores = model.predict_proba(x) if hasattr(model, "predict_proba") else model.decision_function(x)
    elapsed = time.perf_counter() - started
    class_positions = {int(label): index for index, label in enumerate(model.classes_)}
    result = []
    for index, row in enumerate(rows):
        label = int(predictions[index])
        if np.ndim(scores) == 1:
            score = float(scores[index]) * (1 if label == int(model.classes_[1]) else -1)
        else:
            score = float(scores[index, class_positions[label]])
        result.append({
            "experiment_id": experiment_id, "sample_id": ids[index],
            "true_label": int(y[index]), "predicted_label": label,
            "relative_path": str(row["relative_path"]), "pixel_sha256": str(row["pixel_sha256"]),
            "eval_representative": _flag(row["eval_representative"]),
            "score_value": score,
            "score_type": "class_probability" if hasattr(model, "predict_proba") else "decision_function",
        })
    return predictions, result, elapsed


PREDICTION_FIELDS = ["experiment_id", "sample_id", "true_label", "predicted_label", "relative_path",
                     "pixel_sha256", "eval_representative", "score_value", "score_type"]


def _metrics(y: np.ndarray, predictions: np.ndarray, rows: list[dict], labels: list[int]) -> dict:
    return {
        "primary": classification_metrics(y, predictions, labels=labels,
                     representative=[_flag(row["eval_representative"]) for row in rows]),
        "file_secondary": classification_metrics(y, predictions, labels=labels),
    }


def _analysis(directory: Path, metrics: dict, predictions: list[dict], names: dict[int, str],
              data_dir: Path, *, figures: bool) -> None:
    class_rows = []
    for label, values in metrics["primary"]["per_class"].items():
        class_rows.append({"label_id": int(label), "label": names[int(label)], **values})
    _write_csv(directory / "per_class.csv", class_rows,
               ["label_id", "label", "precision", "recall", "f1", "support"])
    labels = metrics["primary"]["labels"]
    for unit, suffix in (("primary", ""), ("file_secondary", "_files")):
        confusion = metrics[unit]["confusion_matrix"]
        fields = ["true_label"] + [names[label] for label in labels]
        _write_csv(directory / f"confusion_matrix{suffix}.csv", [
            {"true_label": names[label], **{names[predicted]: int(confusion[i][j])
             for j, predicted in enumerate(labels)}} for i, label in enumerate(labels)], fields)
    errors = [row for row in predictions if row["true_label"] != row["predicted_label"]]
    _write_csv(directory / "errors.csv", errors, PREDICTION_FIELDS)
    pairs: dict[tuple[int, int], int] = {}
    for row in errors:
        if row["eval_representative"]:
            pair = (row["true_label"], row["predicted_label"])
            pairs[pair] = pairs.get(pair, 0) + 1
    _write_json(directory / "error_analysis.json", {
        "n_file_errors": len(errors),
        "n_representative_errors": sum(row["eval_representative"] for row in errors),
        "confused_pairs": [{"true_label": names[a], "predicted_label": names[b], "count": count}
                           for (a, b), count in sorted(pairs.items(), key=lambda item: (-item[1], item[0]))],
        "note": "Exact-pixel representatives do not establish independence of malware variants.",
    })
    if figures:
        _figures(directory, metrics, class_rows, errors, names, data_dir)


def _figures(directory: Path, metrics: dict, class_rows: list[dict], errors: list[dict],
             names: dict[int, str], data_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = directory / "figures"
    directory.mkdir(exist_ok=True)
    labels = metrics["primary"]["labels"]
    class_names = [names[label] for label in labels]
    confusion = np.asarray(metrics["primary"]["confusion_matrix"])
    figure, axes = plt.subplots(figsize=(max(7, len(labels) * 0.48), max(6, len(labels) * 0.43)))
    image = axes.imshow(confusion, cmap="Blues")
    axes.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=class_names,
             yticklabels=class_names, xlabel="Predicted family", ylabel="True family",
             title="Confusion matrix: one representative per pixel group")
    plt.setp(axes.get_xticklabels(), rotation=90)
    figure.colorbar(image, ax=axes, label="Samples")
    figure.tight_layout()
    figure.savefig(directory / "confusion_matrix.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(figsize=(max(8, len(labels) * 0.45), 4.8))
    axes.bar(class_names, [row["f1"] for row in class_rows], color="#3066BE")
    axes.set(ylim=(0, 1.05), ylabel="F1", title="Per-family F1: pixel-group representatives")
    plt.setp(axes.get_xticklabels(), rotation=90)
    figure.tight_layout()
    figure.savefig(directory / "per_class_f1.png", dpi=180)
    plt.close(figure)
    # Deterministic examples; missing source images do not invalidate feature evaluation.
    examples = []
    for row in sorted(errors, key=lambda row: row["sample_id"]):
        if not row["eval_representative"]:
            continue
        path = (data_dir / row["relative_path"]).resolve()
        if path.is_relative_to(data_dir.resolve()) and path.is_file():
            examples.append((row, path))
        if len(examples) == 12:
            break
    if examples:
        from PIL import Image
        columns = min(4, len(examples))
        figure, axes = plt.subplots(math.ceil(len(examples) / columns), columns,
                                    figsize=(columns * 3, math.ceil(len(examples) / columns) * 3), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (row, path) in zip(axes.ravel(), examples):
            with Image.open(path) as source:
                ax.imshow(source.convert("L"), cmap="gray", aspect="auto")
            ax.set_title(f"True: {names[row['true_label']]}\nPred: {names[row['predicted_label']]}", fontsize=9)
        figure.tight_layout()
        figure.savefig(directory / "error_examples.png", dpi=180)
        plt.close(figure)


def _registry(output: Path, record: dict) -> None:
    path = output / "registry.json"
    current = _read_json(path) if path.exists() else {"schema_version": SCHEMA_VERSION, "runs": []}
    current["runs"] = [item for item in current["runs"] if item["run_id"] != record["run_id"]] + [record]
    _write_json(path, current)


def run_validation(root: Path, config: dict, rows: list[dict], manifest_sha256: str,
                   feature_dirs: dict[str, Path]) -> dict:
    """Fit candidates on train, select C on validation, and lock final-test inputs.

    The returned selection_path is the explicit artifact consumed by final testing.
    Re-running validation creates a preserved run and replaces the latest selection
    before test has been opened. After a final-test marker exists, use a new results
    directory for a separately documented protocol; never tune this test run again.
    """
    root = Path(root).resolve()
    output = _output_dir(root, config)
    if (output / "final_test.lock.json").exists() or (output / "final_test").exists():
        raise ValueError("final test is locked; validation cannot change its selection")
    included, labels, names = _manifest_rows(rows)
    experiments = _experiments(config)
    options = _training_options(config)
    _archive_sha(config)
    if not manifest_sha256:
        raise ValueError("manifest SHA-256 is required")
    config_sha = _object_sha(config)
    run_id = "validation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + config_sha[:8]
    run_dir = output / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    environment = _environment()
    environment["implementation_sha256"] = _implementation_signature()
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "environment.json", environment)
    started_at = _now()
    registry = {"run_id": run_id, "stage": "validation", "status": "running",
                "started_at": started_at, "config_sha256": config_sha,
                "manifest_sha256": manifest_sha256, "output_dir": _reference(root, run_dir),
                "experiments": []}
    _registry(output, registry)
    selection = {"schema_version": SCHEMA_VERSION, "created_at": started_at,
                 "run_id": run_id, "config_sha256": config_sha,
                 "manifest_sha256": manifest_sha256, "manifest_rows_sha256": _object_sha(rows),
                 "config_snapshot": _reference(root, run_dir / "config.json"),
                 "validation_run_dir": _reference(root, run_dir),
                 "implementation_sha256": environment["implementation_sha256"],
                 "selection_metric": "validation macro_f1 on unique_pixel_representative",
                 "tie_break": "largest macro_f1, then smallest C; no test scores are consulted",
                 "trusted_local_models_only": True, "labels": labels,
                 "class_names": {str(label): name for label, name in names.items()},
                 "feature_stores": {}, "experiments": []}
    stores = {}
    try:
        for experiment in experiments:
            extractor = experiment["extractor"]
            if extractor not in stores:
                if extractor not in feature_dirs:
                    raise ValueError(f"no feature directory for extractor {extractor}")
                directory = _root_path(root, feature_dirs[extractor])
                signatures = _feature_signatures(directory)
                _verify_store_metadata(directory, signatures)
                x_train, y_train, _, provenance = _load_aligned(
                    directory, "train", included, extractor, manifest_sha256, config)
                x_val, y_val, val_ids, _ = _load_aligned(
                    directory, "val", included, extractor, manifest_sha256, config, provenance)
                test_metadata = _read_json(directory / "metadata_test.json")
                if _provenance(test_metadata, extractor, manifest_sha256, config) != provenance:
                    raise ValueError("test feature provenance differs from train/validation")
                if x_train.shape[1] != x_val.shape[1] or test_metadata.get("feature_shape", [0, 0])[1] != x_train.shape[1]:
                    raise ValueError("feature dimension differs between partitions")
                stores[extractor] = (x_train, y_train, x_val, y_val, val_ids)
                selection["feature_stores"][extractor] = {
                    "directory": _reference(root, directory), "file_sha256": signatures,
                    "provenance": provenance, "feature_dim": int(x_train.shape[1]),
                    "feature_array_bytes": sum((directory / f"X_{split}.npy").stat().st_size for split in SPLITS),
                }
            x_train, y_train, x_val, y_val, val_ids = stores[extractor]
            train_rows = _split_rows(included, "train")
            val_rows = _split_rows(included, "val")
            train_mask = np.ones(len(y_train), dtype=bool)
            if config.get("training", {}).get("train_unit", "file") == "representative":
                train_mask = np.array([_flag(row["eval_representative"]) for row in train_rows])
            experiment_dir = run_dir / experiment["id"]
            experiment_dir.mkdir()
            trials = []
            for trial_index, c in enumerate(sorted(float(value) for value in experiment.get("c_values", [0.1, 1, 10]))):
                trial_dir = experiment_dir / f"trial_{trial_index:02d}"
                trial_dir.mkdir()
                trial = {"c": c, "status": "running", "warnings": [], "started_at": _now()}
                captured = []
                try:
                    model = build_classifier(experiment["classifier"], c=c, **options)
                    fit_started = time.perf_counter()
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        model.fit(x_train[train_mask], y_train[train_mask])
                    trial["fit_seconds"] = time.perf_counter() - fit_started
                    predictions, prediction_rows, predict_seconds = _prediction_rows(
                        model, x_val, y_val, val_ids, val_rows, experiment["id"])
                    trial.update({"status": "completed", "predict_seconds": predict_seconds,
                                  "metrics": _metrics(y_val, predictions, val_rows, labels),
                                  "n_train_fit": int(train_mask.sum()), "n_train_files": len(y_train),
                                  "feature_dim": int(x_train.shape[1]),
                                  "n_iterations": np.asarray(model.steps[-1][1].n_iter_).tolist()})
                    model_path = trial_dir / "model.joblib"
                    joblib.dump(model, model_path)
                    trial["model_path"] = _reference(root, model_path)
                    trial["model_sha256"] = _file_sha(model_path)
                    trial["model_bytes"] = model_path.stat().st_size
                    trial["model_classes"] = [int(label) for label in model.classes_]
                    _write_csv(trial_dir / "predictions.csv", prediction_rows, PREDICTION_FIELDS)
                    _write_json(trial_dir / "metrics.json", trial["metrics"])
                except Exception as error:
                    trial.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
                trial["warnings"] = [{"category": warning.category.__name__, "message": str(warning.message)}
                                     for warning in captured]
                trial["converged"] = trial["status"] == "completed" and not any(
                    warning["category"] == "ConvergenceWarning" for warning in trial["warnings"])
                trial["completed_at"] = _now()
                _write_json(trial_dir / "trial.json", trial)
                _write_json(trial_dir / "warnings.json", trial["warnings"])
                trials.append(trial)
            completed = [trial for trial in trials if trial["status"] == "completed"]
            record = {"experiment_id": experiment["id"], "extractor": extractor,
                      "classifier": experiment["classifier"], "trials": trials}
            _write_json(experiment_dir / "trials.json", record)
            if not completed:
                registry["experiments"].append(record)
                raise RuntimeError(f"all candidates failed for {experiment['id']}; inspect trials.json")
            best = min(completed, key=lambda trial: (-trial["metrics"]["primary"]["macro_f1"], trial["c"]))
            record["selected_c"] = best["c"]
            selected = {"experiment_id": experiment["id"], "extractor": extractor,
                        "classifier": experiment["classifier"], "selected_c": best["c"],
                        "model_path": best["model_path"], "model_sha256": best["model_sha256"],
                        "model_classes": best["model_classes"], "validation_metrics": best["metrics"],
                        "fit_seconds": best["fit_seconds"], "validation_predict_seconds": best["predict_seconds"],
                        "model_bytes": best["model_bytes"], "convergence_warnings": best["warnings"],
                        "converged": best["converged"], "n_iterations": best["n_iterations"],
                        "n_train_fit": best["n_train_fit"], "feature_dim": best["feature_dim"],
                        "feature_array_bytes": selection["feature_stores"][extractor]["feature_array_bytes"],
                        "validation_output_dir": _reference(root, experiment_dir),
                        "trials_path": _reference(root, experiment_dir / "trials.json")}
            selected_predictions = _read_predictions(_root_path(root, best["model_path"]).parent / "predictions.csv")
            _write_csv(experiment_dir / "predictions.csv", selected_predictions, PREDICTION_FIELDS)
            _write_json(experiment_dir / "metrics.json", best["metrics"])
            _analysis(experiment_dir, best["metrics"], selected_predictions, names,
                      _root_path(root, config.get("data_dir", "data/images")),
                      figures=config.get("evaluation", {}).get("figures", True))
            selection["experiments"].append(selected)
            registry["experiments"].append(record)
            _registry(output, registry)
        # Recheck identity after fitting to reject edits during a long run.
        for store in selection["feature_stores"].values():
            if _feature_signatures(_root_path(root, store["directory"])) != store["file_sha256"]:
                raise ValueError("feature files changed during validation")
        audit_path = run_dir / "selection_audit.json"
        selection["selection_audit_path"] = _reference(root, audit_path)
        _write_json(audit_path, {"schema_version": SCHEMA_VERSION, "selection_metric": selection["selection_metric"],
                    "tie_break": selection["tie_break"], "selection": selection["experiments"],
                    "test_arrays_loaded": False, "test_scores_computed": False,
                    "train_unit": config.get("training", {}).get("train_unit", "file"),
                    "environment": environment, "manifest_sha256": manifest_sha256, "config_sha256": config_sha})
        selection["selection_audit_sha256"] = _file_sha(audit_path)
        selection_path = output / "selection.json"
        _write_json(run_dir / "selection.json", selection)
        _write_json(selection_path, selection)
        registry.update({"status": "completed", "completed_at": _now(),
                         "selection_path": _reference(root, selection_path)})
        _registry(output, registry)
        summarize_results(output)
        return {"status": "completed", "selection_path": str(selection_path),
                "run_dir": str(run_dir), "registry_path": str(output / "registry.json"),
                "comparison_path": str(output / "comparison.csv"), "experiments": selection["experiments"]}
    except Exception as error:
        registry.update({"status": "failed", "completed_at": _now(),
                         "error_type": type(error).__name__, "error": str(error)})
        _registry(output, registry)
        raise


def _verify_selection(root: Path, config: dict, rows: list[dict], manifest_sha256: str,
                      selection_path: Path) -> tuple[dict, Path]:
    selection_path = _root_path(root, selection_path)
    selection = _read_json(selection_path)
    output = _output_dir(root, config)
    if selection.get("schema_version") != SCHEMA_VERSION or not selection.get("trusted_local_models_only"):
        raise ValueError("selection must be produced locally by run_validation")
    if selection.get("config_sha256") != _object_sha(config):
        raise ValueError("config changed after validation selection")
    if selection.get("implementation_sha256") != _implementation_signature():
        raise ValueError("training/evaluation implementation changed after selection")
    if selection.get("manifest_sha256") != manifest_sha256 or selection.get("manifest_rows_sha256") != _object_sha(rows):
        raise ValueError("manifest changed after validation selection")
    run_dir = _root_path(root, selection["validation_run_dir"])
    if not run_dir.is_relative_to(output / "runs"):
        raise ValueError("selected models must remain in this results directory's trusted local runs")
    if _object_sha(_read_json(_root_path(root, selection["config_snapshot"]))) != selection["config_sha256"]:
        raise ValueError("config snapshot changed after validation")
    audit_path = _root_path(root, selection["selection_audit_path"])
    if _file_sha(audit_path) != selection["selection_audit_sha256"]:
        raise ValueError("selection audit changed after validation")
    trained_software = _read_json(audit_path).get("environment", {}).get("software", {})
    current_software = _environment()["software"]
    for package in ("numpy", "scikit-learn", "joblib"):
        if not trained_software.get(package) or trained_software[package] != current_software.get(package):
            raise ValueError(
                f"{package} version differs from validation: "
                f"trained={trained_software.get(package)!r}, current={current_software.get(package)!r}; "
                "restore the validation environment before loading its trusted local model"
            )
    saved_selection = _read_json(run_dir / "selection.json")
    if saved_selection != selection:
        raise ValueError("selection differs from the preserved validation run")
    for store in selection["feature_stores"].values():
        if _feature_signatures(_root_path(root, store["directory"])) != store["file_sha256"]:
            raise ValueError("feature files changed after validation selection")
    for experiment in selection["experiments"]:
        model_path = _root_path(root, experiment["model_path"])
        if not model_path.is_relative_to(run_dir) or _file_sha(model_path) != experiment["model_sha256"]:
            raise ValueError("trusted local selected model is missing or has changed")
    return selection, output


def run_final_test(root: Path, config: dict, rows: list[dict], manifest_sha256: str,
                   selection_path: Path) -> dict:
    """Evaluate locked locally generated candidates exactly once, without fitting.

    A complete unchanged run can be read again without loading models/test arrays.
    A started/failed marker requires review rather than an automatic second test.
    """
    root = Path(root).resolve()
    included, labels, names = _manifest_rows(rows)
    selection, output = _verify_selection(root, config, rows, manifest_sha256, selection_path)
    selection_sha = _file_sha(_root_path(root, selection_path))
    lock_path = output / "final_test.lock.json"
    result_path = output / "final_test" / "results.json"
    if lock_path.exists():
        marker = _read_json(lock_path)
        if marker.get("selection_sha256") != selection_sha:
            raise ValueError("final-test lock belongs to a different selection")
        if marker.get("status") == "completed" and result_path.exists():
            if marker.get("results_sha256") != _file_sha(result_path):
                raise ValueError("final-test results changed after completion")
            return {**_read_json(result_path), "reused": True, "results_path": str(result_path)}
        raise ValueError("final test already started; inspect its lock and failure before retrying")
    if (output / "final_test").exists():
        raise ValueError("pre-existing final-test outputs cannot be overwritten; review their provenance")
    marker = {"schema_version": SCHEMA_VERSION, "status": "started", "started_at": _now(),
              "selection_sha256": selection_sha, "config_sha256": selection["config_sha256"],
              "manifest_sha256": manifest_sha256, "selection_path": _reference(root, _root_path(root, selection_path))}
    output.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2)
    final_dir = output / "final_test"
    final_dir.mkdir(exist_ok=False)
    final_result = {"schema_version": SCHEMA_VERSION, "status": "running", "stage": "final_test",
                    "selection_sha256": selection_sha, "manifest_sha256": manifest_sha256,
                    "config_sha256": selection["config_sha256"], "started_at": marker["started_at"],
                    "environment": _environment(), "experiments": []}
    cached = {}
    try:
        for experiment in selection["experiments"]:
            extractor = experiment["extractor"]
            if extractor not in cached:
                store = selection["feature_stores"][extractor]
                cached[extractor] = _load_aligned(_root_path(root, store["directory"]), "test",
                    included, extractor, manifest_sha256, config, store["provenance"])
                if cached[extractor][0].shape[1] != store["feature_dim"]:
                    raise ValueError("test feature dimension does not match selected model")
            x, y, ids, _ = cached[extractor]
            model = joblib.load(_root_path(root, experiment["model_path"]))
            if [int(label) for label in model.classes_] != experiment["model_classes"] or list(model.classes_) != labels:
                raise ValueError("selected model classes do not match the manifest")
            test_rows = _split_rows(included, "test")
            predictions, prediction_rows, predict_seconds = _prediction_rows(
                model, x, y, ids, test_rows, experiment["experiment_id"])
            metrics = _metrics(y, predictions, test_rows, labels)
            experiment_dir = final_dir / experiment["experiment_id"]
            experiment_dir.mkdir()
            record = {"experiment_id": experiment["experiment_id"], "extractor": extractor,
                      "classifier": experiment["classifier"], "selected_c": experiment["selected_c"],
                      "metrics": metrics, "predict_seconds": predict_seconds,
                      "model_sha256": experiment["model_sha256"],
                      "output_dir": _reference(root, experiment_dir)}
            _write_csv(experiment_dir / "predictions.csv", prediction_rows, PREDICTION_FIELDS)
            _write_json(experiment_dir / "metrics.json", metrics)
            _analysis(experiment_dir, metrics, prediction_rows, names,
                      _root_path(root, config.get("data_dir", "data/images")),
                      figures=config.get("evaluation", {}).get("figures", True))
            final_result["experiments"].append(record)
            _write_json(result_path, final_result)
        final_result.update({"status": "completed", "completed_at": _now(), "reused": False})
        _write_json(result_path, final_result)
        marker.update({"status": "completed", "completed_at": final_result["completed_at"],
                       "results_sha256": _file_sha(result_path)})
        _write_json(lock_path, marker)
        _registry(output, {"run_id": "final_test_" + selection["run_id"], "stage": "final_test",
                  "status": "completed", "selection_sha256": selection_sha,
                  "config_sha256": selection["config_sha256"], "manifest_sha256": manifest_sha256,
                  "output_dir": _reference(root, final_dir), "experiments": final_result["experiments"]})
        summarize_results(output)
        return {**final_result, "results_path": str(result_path)}
    except Exception as error:
        marker.update({"status": "failed", "error_type": type(error).__name__, "error": str(error),
                       "failed_at": _now()})
        _write_json(lock_path, marker)
        final_result.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        _write_json(result_path, final_result)
        raise


def summarize_results(output_dir: Path) -> Path:
    """Create the comparison table from real saved metrics; blank means not run."""
    output_dir = Path(output_dir)
    selection_path = output_dir / "selection.json"
    if not selection_path.exists():
        raise ValueError("validation selection is required before summarizing results")
    selection = _read_json(selection_path)
    test_path = output_dir / "final_test" / "results.json"
    test = _read_json(test_path) if test_path.exists() else {}
    if test and test.get("selection_sha256") != _file_sha(selection_path):
        raise ValueError("saved final-test results belong to a different selection")
    if test:
        lock_path = output_dir / "final_test.lock.json"
        if not lock_path.exists():
            raise ValueError("saved final-test results have no completion lock")
        lock = _read_json(lock_path)
        if lock.get("selection_sha256") != _file_sha(selection_path):
            raise ValueError("final-test completion lock belongs to a different selection")
        if test.get("status") == "completed":
            if lock.get("status") != "completed" or lock.get("results_sha256") != _file_sha(test_path):
                raise ValueError("final-test results do not match their completed lock")
        elif lock.get("status") == "completed":
            raise ValueError("completed final-test lock has incomplete results")
    test_records = {record["experiment_id"]: record for record in test.get("experiments", [])}
    records = []
    for experiment in selection["experiments"]:
        metrics = experiment["validation_metrics"]
        record = {"experiment_id": experiment["experiment_id"], "extractor": experiment["extractor"],
                  "classifier": experiment["classifier"], "selected_c": experiment["selected_c"],
                  "manifest_sha256": selection["manifest_sha256"], "config_sha256": selection["config_sha256"],
                  "validation_macro_f1_representative": metrics["primary"]["macro_f1"],
                  "validation_macro_f1_file": metrics["file_secondary"]["macro_f1"],
                  "fit_seconds": experiment["fit_seconds"],
                  "validation_predict_seconds": experiment["validation_predict_seconds"],
                  "model_bytes": experiment["model_bytes"],
                  "feature_dim": experiment["feature_dim"],
                  "feature_array_bytes": experiment["feature_array_bytes"],
                  "n_train_fit": experiment["n_train_fit"], "converged": experiment["converged"],
                  "convergence_warning_count": sum(warning["category"] == "ConvergenceWarning"
                                                   for warning in experiment["convergence_warnings"]),
                  "test_status": test.get("status", "not_run")}
        if test.get("status") == "completed" and record["experiment_id"] in test_records:
            result = test_records[record["experiment_id"]]
            for unit, suffix in (("primary", "representative"), ("file_secondary", "file")):
                for key in ("macro_f1", "macro_precision", "macro_recall", "accuracy", "n_samples"):
                    record[f"test_{key}_{suffix}"] = result["metrics"][unit][key]
            record["test_predict_seconds"] = result["predict_seconds"]
        records.append(record)
    fields = list(records[0])
    for suffix in ("representative", "file"):
        for key in ("macro_f1", "macro_precision", "macro_recall", "accuracy", "n_samples"):
            field = f"test_{key}_{suffix}"
            if field not in fields:
                fields.append(field)
    if "test_predict_seconds" not in fields:
        fields.append("test_predict_seconds")
    _write_json(output_dir / "comparison.json", {"schema_version": SCHEMA_VERSION,
                "primary_metric": "macro_f1_unique_pixel_representatives", "rows": records,
                "note": "Test fields remain absent/blank until explicit final testing succeeds."})
    _write_csv(output_dir / "comparison.csv", records, fields)
    return output_dir / "comparison.csv"
