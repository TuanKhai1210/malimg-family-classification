"""Epoch-resumable SGD on frozen features; separate from historical LR/SVC runs.

Checkpoint only at a completed epoch. A killed minibatch restarts that epoch.
Joblib state contains the estimator's learning-rate counter and the shuffle RNG.
Load only own-group checkpoints. Validation never deserializes test arrays.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import hinge_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.metaestimators import available_if

from . import experiments as ex
from .config import validate_path_option


DEFAULT_SETTINGS = {
    "extractor": "resnet50", "loss": "log_loss", "alpha": 0.0001,
    "epochs": 30, "batch_size": 256, "patience": 6,
    "learning_rate": "optimal", "eta0": 0.01,
}


def _stable_log_probabilities(scores):
    """Normalized OvR sigmoid probabilities in log space, not softmax scores."""
    scores = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("Non-finite SGD decision scores; inspect features and optimizer state before retrying")
    if scores.ndim == 1:
        values = np.column_stack((-np.logaddexp(0.0, scores), -np.logaddexp(0.0, -scores)))
    elif scores.ndim == 2 and scores.shape[1] >= 2:
        values = -np.logaddexp(0.0, -scores)
    else:
        raise ValueError("SGD scores must be binary or have one column per class")
    # Center before reducing: expit(scores) may underflow to zero for EVERY
    # class, while subtracting an uncentered logsum can lose precision.
    centered = values - values.max(axis=1, keepdims=True)
    return centered - np.log(np.exp(centered).sum(axis=1, keepdims=True))


class _StableSGDClassifier(SGDClassifier):
    """Keep SGD fitting/predictions unchanged; stabilize log-loss probabilities."""

    @available_if(lambda self: self.loss == "log_loss")
    def predict_proba(self, x):
        return np.exp(_stable_log_probabilities(self.decision_function(x)))

    @available_if(lambda self: self.loss == "log_loss")
    def predict_log_proba(self, x):
        return _stable_log_probabilities(self.decision_function(x))


def validate_settings(settings: dict) -> dict:
    if set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError("Epoch settings must contain exactly: " + ", ".join(DEFAULT_SETTINGS))
    for key in ("epochs", "batch_size", "patience"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"epoch {key} must be a positive integer")
    for key in ("alpha", "eta0"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"epoch {key} must be positive and finite")
    if settings["loss"] not in {"log_loss", "hinge"}:
        raise ValueError("epoch loss must be log_loss or hinge; these are new SGD experiments")
    if settings["learning_rate"] not in {"optimal", "constant", "invscaling"}:
        raise ValueError("unsupported SGD learning_rate")
    if settings["extractor"] not in {"resnet50", "convnext_tiny", "hog", "image_size"}:
        raise ValueError("unsupported epoch extractor")
    return copy.deepcopy(settings)


def _implementation() -> dict:
    directory = Path(__file__).parent
    return {name: ex._file_sha(directory / name) for name in (
        "epoch_training.py", "experiments.py", "evaluation.py", "feature_io.py", "contracts.py")}


def _identity(config, settings, rows, manifest_sha, feature_dirs, root):
    environment = ex._environment()
    extractor = settings["extractor"]
    directory = ex._root_path(root, feature_dirs[extractor])
    signatures = ex._feature_signatures(directory)
    ex._verify_store_metadata(directory, signatures)
    return {
        "schema": "sgd-epoch-v1", "config": config, "settings": settings,
        "manifest_sha256": manifest_sha, "rows_sha256": ex._object_sha(rows),
        "implementation": _implementation(), "python": list(sys.version_info[:2]),
        "software": {k: environment["software"][k] for k in ("numpy", "scipy", "scikit-learn", "joblib")},
        "feature_files": signatures,
    }


def _run_directory(root, config, run_id, *, create=False):
    parent = (root / config.get("results_dir", "results") / "epoch_runs").resolve()
    if not parent.is_relative_to(root.resolve()):
        raise ValueError("Epoch results must stay inside the workspace")
    value = validate_path_option(run_id, "EPOCH_RUN")
    if value.name != str(value) or value.name in {".", ".."}:
        raise ValueError("EPOCH_RUN must be one run ID, not a path")
    directory = (parent / value).resolve()
    if directory.parent != parent:
        raise ValueError("epoch run must stay inside results/epoch_runs")
    if create:
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir():
        raise FileNotFoundError(f"Epoch run is missing: {directory}; restore its backup first")
    return directory


def _committed(directory):
    """Ignore unpublished staging folders; never silently fall back from corruption."""
    checkpoints = sorted((directory / "epochs").glob("epoch_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    for index, path in enumerate(checkpoints, 1):
        commit = ex._read_json(path / "commit.json")
        if commit.get("epoch") != index or path.name != f"epoch_{index:06d}":
            raise ValueError("Epoch checkpoint sequence is incomplete")
        if set(commit.get("files", {})) != {"state.joblib", "metrics.json"}:
            raise ValueError("Epoch checkpoint is incomplete")
        for name, checksum in commit["files"].items():
            if ex._file_sha(path / name) != checksum:
                raise ValueError(f"Epoch checkpoint checksum mismatch: {path.name}/{name}")
    return checkpoints


def _load_state(path, identity):
    # Caller verifies commit hashes and run identity before executable deserialization.
    state = joblib.load(path / "state.joblib")
    if state.get("identity") != identity or state["epoch"] != len(state["history"]):
        raise ValueError("Epoch state identity/history mismatch")
    return state


def _publish_epoch(directory, state, metrics):
    parent = directory / "epochs"
    parent.mkdir(exist_ok=True)
    staging = parent / (".staging-" + uuid.uuid4().hex)
    staging.mkdir()
    with (staging / "state.joblib").open("wb") as stream:
        joblib.dump(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    ex._write_json(staging / "metrics.json", metrics)
    ex._write_json(staging / "commit.json", {
        "epoch": state["epoch"], "files": {name: ex._file_sha(staging / name)
                                               for name in ("state.joblib", "metrics.json")},
    })
    target = parent / f"epoch_{state['epoch']:06d}"
    if target.exists():
        raise ValueError("Refusing to overwrite a committed epoch")
    os.replace(staging, target)
    return target


def _training_data(root, config, rows, manifest_sha, settings, feature_dirs):
    included, labels, names = ex._manifest_rows(rows)
    extractor = settings["extractor"]
    directory = ex._root_path(root, feature_dirs[extractor])
    x, y, _, provenance = ex._load_aligned(directory, "train", included, extractor, manifest_sha, config)
    xv, yv, ids, _ = ex._load_aligned(directory, "val", included, extractor, manifest_sha, config, provenance)
    test_metadata = ex._read_json(directory / "metadata_test.json")
    if ex._provenance(test_metadata, extractor, manifest_sha, config) != provenance:
        raise ValueError("Test feature provenance differs; test arrays were not opened")
    if x.shape[1] != xv.shape[1] or test_metadata.get("feature_shape", [0, 0])[1] != x.shape[1]:
        raise ValueError("Feature dimension differs between partitions")
    mask = np.ones(len(y), dtype=bool)
    if config.get("training", {}).get("train_unit", "file") == "representative":
        mask = np.array([ex._flag(r["eval_representative"]) for r in ex._split_rows(included, "train")])
    return included, labels, names, x[mask], y[mask], xv, yv, ids, provenance


def _initial_state(x, y, labels, identity, config, settings):
    scaler = StandardScaler().fit(x)
    steps = [("standardscaler", scaler)]
    pca = config.get("training", {}).get("pca_components")
    if pca is not None:
        steps.append(("pca", PCA(n_components=pca, random_state=config["training"]["seed"]).fit(scaler.transform(x))))
    weights = None
    if config.get("training", {}).get("class_weight") == "balanced":
        values = compute_class_weight("balanced", classes=np.asarray(labels), y=y)
        weights = {label: float(weight) for label, weight in zip(labels, values)}
    steps.append(("sgdclassifier", _StableSGDClassifier(
        loss=settings["loss"], alpha=settings["alpha"], learning_rate=settings["learning_rate"],
        eta0=settings["eta0"], class_weight=weights, shuffle=False,
        random_state=config["training"]["seed"], average=False,
    )))
    return {"identity": identity, "model": Pipeline(steps), "epoch": 0, "history": [],
            "rng_state": np.random.default_rng(config["training"]["seed"]).bit_generator.state,
            "best_epoch": 0, "best_score": -1.0, "stale_epochs": 0}


def _write_summary(directory, state, status):
    summary = {"engine": "sgd_epoch", "run_id": directory.name, "status": status,
               "epoch": state["epoch"], "best_epoch": state["best_epoch"],
               "best_validation_macro_f1": state["best_score"], "updated_at": ex._now(),
               "checkpoint": f"epochs/epoch_{state['epoch']:06d}" if state["epoch"] else None}
    if (directory / "selection.json").is_file():
        summary["selection_sha256"] = ex._file_sha(directory / "selection.json")
    ex._write_json(directory / "status.json", summary)
    if state["history"]:
        ex._write_csv(directory / "history.csv", state["history"], list(state["history"][0]))
    return summary


def _export_validation(root, config, directory, state, rows, labels, names, xv, yv, ids):
    checkpoint = directory / "epochs" / f"epoch_{state['best_epoch']:06d}"
    best = _load_state(checkpoint, state["identity"])
    _, predictions, _ = ex._prediction_rows(best["model"], xv, yv, ids, ex._split_rows(rows, "val"), "SGD")
    values = np.asarray([p["predicted_label"] for p in predictions])
    metrics = ex._metrics(yv, values, ex._split_rows(rows, "val"), labels)
    output = directory / "validation"
    output.mkdir(exist_ok=True)
    ex._write_csv(output / "predictions.csv", predictions, ex.PREDICTION_FIELDS)
    ex._write_json(output / "metrics.json", metrics)
    ex._analysis(output, metrics, predictions, names, root / config.get("data_dir", "data/images"),
                 figures=config.get("evaluation", {}).get("figures", True))
    ex._write_json(directory / "selection.json", {
        "engine": "sgd_epoch", "run_id": directory.name, "best_epoch": state["best_epoch"],
        "model_path": ex._reference(root, checkpoint / "state.joblib"),
        "model_sha256": ex._file_sha(checkpoint / "state.joblib"),
        "identity_sha256": ex._object_sha(state["identity"]), "validation_metrics": metrics,
        "selection_rule": "highest validation representative Macro-F1; ties choose earliest epoch",
    })


def run_epoch_training(root, config, rows, manifest_sha256, feature_dirs, settings, *,
                       resume_run=None, checkpoint_callback=None, max_session_seconds=None,
                       epochs_this_session=None):
    """Train SGD and sync AFTER committing each epoch; a sync error stops progress.

    Callback runs also before fitting (and on completed resume), allowing retry of
    a previously failed upload without repeating an epoch. No test scores select a model.
    """
    root = validate_path_option(root, "repo_root").resolve()
    settings = validate_settings(settings)
    ex._training_options(config)
    for key, value in (("max_session_seconds", max_session_seconds), ("epochs_this_session", epochs_this_session)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be positive or None")
    results = root / config.get("results_dir", "results")
    if (results / "final_test.lock.json").exists() or (results / "final_test").exists():
        raise ValueError("Final test is locked; cannot tune or resume training")
    identity = _identity(config, settings, rows, manifest_sha256, feature_dirs, root)
    if resume_run is None:
        stamp = ex._now().replace("-", "").replace(":", "").replace(".", "").replace("+0000", "Z")
        run_id = f"sgd_{stamp}_{ex._object_sha(identity)[:8]}"
    else:
        run_id = resume_run
    directory = _run_directory(root, config, run_id, create=resume_run is None)
    started = time.monotonic()
    with ex._run_lock(directory):
        if resume_run is None:
            ex._write_json(directory / "identity.json", identity)
            ex._write_json(directory / "config.json", config)
            ex._write_json(directory / "settings.json", settings)
            ex._write_json(directory / "environment.json", ex._environment())
        elif ex._read_json(directory / "identity.json") != identity:
            raise ValueError("Epoch resume identity mismatch: source/config/features/Python/libraries changed")
        checkpoints = _committed(directory)
        data = _training_data(root, config, rows, manifest_sha256, settings, feature_dirs)
        included, labels, names, x, y, xv, yv, ids, _ = data
        state = _load_state(checkpoints[-1], identity) if checkpoints else _initial_state(x, y, labels, identity, config, settings)
        print(f"SGD run: {directory.name}; committed epoch {state['epoch']}/{settings['epochs']}", flush=True)
        _write_summary(directory, state, "running")
        if checkpoint_callback:
            checkpoint_callback("epoch_session_started", directory)
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        transformed = state["model"][:-1].transform(x)
        initial_epoch = state["epoch"]
        while state["epoch"] < settings["epochs"] and state["stale_epochs"] < settings["patience"]:
            if ((max_session_seconds is not None and time.monotonic() - started >= max_session_seconds)
                    or (epochs_this_session is not None and state["epoch"] - initial_epoch >= epochs_this_session)):
                break
            epoch_started = time.perf_counter()
            order = rng.permutation(len(y))
            estimator = state["model"].steps[-1][1]
            for offset in range(0, len(order), settings["batch_size"]):
                batch = order[offset:offset + settings["batch_size"]]
                estimator.partial_fit(transformed[batch], y[batch], classes=np.asarray(labels))
            fit_seconds = time.perf_counter() - epoch_started
            train_prediction = state["model"].predict(x)
            if settings["loss"] == "log_loss":
                loss = log_loss(y, state["model"].predict_proba(x), labels=labels)
            else:
                loss = hinge_loss(y, state["model"].decision_function(x), labels=labels)
            val_prediction = state["model"].predict(xv)
            metrics = ex._metrics(yv, val_prediction, ex._split_rows(included, "val"), labels)
            score = metrics["primary"]["macro_f1"]
            state["epoch"] += 1
            improved = score > state["best_score"]
            if improved:
                state.update(best_score=score, best_epoch=state["epoch"], stale_epochs=0)
            else:
                state["stale_epochs"] += 1
            state["history"].append({
                "epoch": state["epoch"], "train_loss_unweighted": float(loss),
                "train_accuracy": float(np.mean(train_prediction == y)),
                "val_macro_f1": score, "val_accuracy": metrics["primary"]["accuracy"],
                "fit_seconds": fit_seconds, "best_epoch": state["best_epoch"],
            })
            state["rng_state"] = copy.deepcopy(rng.bit_generator.state)
            _publish_epoch(directory, state, metrics)
            if improved:
                _export_validation(root, config, directory, state, included, labels, names, xv, yv, ids)
            _write_summary(directory, state, "running")
            print(f"Epoch {state['epoch']:03d}: loss={loss:.5f}, val F1={score:.6f}; "
                  f"best={state['best_epoch']}; local checkpoint committed", flush=True)
            if checkpoint_callback:
                checkpoint_callback(f"epoch_{state['epoch']:06d}", directory)
        done = state["epoch"] >= settings["epochs"] or state["stale_epochs"] >= settings["patience"]
        if state["epoch"]:
            _export_validation(root, config, directory, state, included, labels, names, xv, yv, ids)
        summary = _write_summary(directory, state, "completed" if done else "paused")
        summary["run_dir"] = str(directory)
        if checkpoint_callback:
            checkpoint_callback("epoch_session_completed" if done else "epoch_session_paused", directory)
        return summary


def run_epoch_final_test(root, config, rows, manifest_sha256, feature_dirs, settings, *, run_id,
                         confirm_frozen=False, checkpoint_callback=None):
    """Explicit one-time test; reuse saved results without refitting or retesting."""
    if confirm_frozen is not True:
        raise ValueError("Set FINAL_TEST=True only after freezing the protocol and validation choices")
    root = validate_path_option(root, "repo_root").resolve()
    settings = validate_settings(settings)
    directory = _run_directory(root, config, run_id)
    identity = _identity(config, settings, rows, manifest_sha256, feature_dirs, root)
    if ex._read_json(directory / "identity.json") != identity:
        raise ValueError("Final epoch identity mismatch")
    _committed(directory)
    selection = ex._read_json(directory / "selection.json")
    if selection.get("identity_sha256") != ex._object_sha(identity):
        raise ValueError("Selected identity changed")
    best_dir = directory / "epochs" / f"epoch_{selection['best_epoch']:06d}"
    if selection["model_path"] != ex._reference(root, best_dir / "state.joblib") or selection["model_sha256"] != ex._file_sha(best_dir / "state.joblib"):
        raise ValueError("Selected epoch model changed")
    status = ex._read_json(directory / "status.json")
    if status["status"] != "completed":
        raise ValueError("Complete the planned epoch/early-stopping protocol before final test")
    if (status.get("best_epoch") != selection["best_epoch"] or
            status.get("selection_sha256") != ex._file_sha(directory / "selection.json")):
        raise ValueError("Validation selection changed after completion")
    results = root / config.get("results_dir", "results")
    lock = results / "final_test.lock.json"
    result_path = results / "final_test/results.json"
    selection_sha = ex._file_sha(directory / "selection.json")
    if lock.exists():
        marker = ex._read_json(lock)
        if marker.get("engine") != "sgd_epoch" or marker.get("selection_sha256") != selection_sha:
            raise ValueError("Final test already belongs to a different selection")
        if marker["status"] == "completed" and ex._file_sha(result_path) == marker["results_sha256"]:
            return {**ex._read_json(result_path), "reused": True}
        raise ValueError("Final test previously started/failed; inspect its lock before retrying")
    if result_path.parent.exists():
        raise ValueError("Pre-existing final outputs cannot be overwritten")
    marker = {"engine": "sgd_epoch", "status": "started", "selection_sha256": selection_sha,
              "run_id": run_id, "started_at": ex._now()}
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2)
    if checkpoint_callback:
        checkpoint_callback("final_test_started", directory)
    result_path.parent.mkdir()
    try:
        included, labels, names = ex._manifest_rows(rows)
        feature_dir = ex._root_path(root, feature_dirs[settings["extractor"]])
        x, y, ids, _ = ex._load_aligned(feature_dir, "test", included, settings["extractor"], manifest_sha256, config)
        state = _load_state(best_dir, identity)
        _, predictions, seconds = ex._prediction_rows(state["model"], x, y, ids, ex._split_rows(included, "test"), "SGD")
        metrics = ex._metrics(y, np.asarray([p["predicted_label"] for p in predictions]), ex._split_rows(included, "test"), labels)
        ex._write_csv(result_path.parent / "predictions.csv", predictions, ex.PREDICTION_FIELDS)
        ex._analysis(result_path.parent, metrics, predictions, names, root / config.get("data_dir", "data/images"),
                     figures=config.get("evaluation", {}).get("figures", True))
        result = {"engine": "sgd_epoch", "status": "completed", "run_id": run_id,
                  "best_epoch": selection["best_epoch"], "metrics": metrics,
                  "predict_seconds": seconds, "reused": False, "completed_at": ex._now()}
        ex._write_json(result_path, result)
        marker.update(status="completed", results_sha256=ex._file_sha(result_path))
        ex._write_json(lock, marker)
    except BaseException:
        marker.update(status="failed")
        ex._write_json(lock, marker)
        if checkpoint_callback:
            checkpoint_callback("final_test_failed", directory)
        raise
    if checkpoint_callback:
        checkpoint_callback("final_test_completed", directory)
    return result
