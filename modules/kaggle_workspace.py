"""Workspace recovery, verified feature-only reuse, and compact notebook reports."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from . import experiments as ex
from .checkpoints import restore_backup
from .data import prepare_dataset, read_manifest


def restore_workspace(snapshot, destination, *, upgrade_from=None):
    """Exact recovery by default; upgrading source is explicit and means NEW runs."""
    restored = restore_backup(snapshot, destination)
    if upgrade_from is not None:
        source = Path(upgrade_from).resolve()
        if source == restored or not (source / "modules/epoch_training.py").is_file():
            raise ValueError("Upgrade source must be the current Kaggle implementation")
        # Preserve saved configuration, metadata, caches, and historical run files.
        for folder in ("modules", "scripts", "notebooks", "docs", "tests"):
            for path in (source / folder).rglob("*"):
                if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                    target = restored / path.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
        for name in ("README.md", "requirements.txt", "requirements-dev.txt", "CONTRIBUTING.md", "AGENTS.md"):
            if (source / name).is_file():
                shutil.copyfile(source / name, restored / name)
        revision = None
        if (source / ".git").exists():
            revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        ex._write_json(restored / "MIGRATION.json", {
            "mode": "new_run_from_verified_artifacts", "upgraded_at": ex._now(),
            "source_commit": revision,
            "note": "Old results retained. Scientific source/environment identity still governs any resume.",
        })
    return restored


def saved_manifest(root, config):
    """Verify saved preparation artifacts, without claiming to re-audit raw images."""
    root = Path(root)
    version = config.get("manifest_version", "v1")
    provenance = ex._read_json(root / "metadata" / f"preparation_provenance_{version}.json")
    identity = provenance["identity"]
    for key, value in {
        "archive_sha256": config["archive_sha256"], "archive_bytes": config["archive_bytes"],
        "excluded_families": sorted(config["excluded_families"]), "split_seed": config["split_seed"],
        "split_target": config["split_target"], "manifest_version": version,
    }.items():
        if identity.get(key) != value:
            raise ValueError(f"Saved manifest preparation differs from config: {key}")
    for name, checksum in provenance["artifacts"].items():
        if Path(name).name != name or ex._file_sha(root / "metadata" / name) != checksum:
            raise ValueError("Saved preparation artifact checksum mismatch")
    manifest = root / "metadata" / f"split_manifest_{version}.csv"
    rows = read_manifest(manifest)
    ex._manifest_rows(rows)  # validates splits, representatives, labels and pixel leakage
    return rows, ex._file_sha(manifest)


def prepare_features(root, config, extractors, *, archive_path=None, feature_dirs=None,
                     checkpoint_callback=None, extract_missing=True):
    """Use validated saved features across devices without re-running extraction.

    These are fixed previously extracted inputs to a NEW classifier experiment.
    Calling the extractor is a separate path with its original environment/device
    cache signature. Multiple cached candidates require an explicit directory.
    """
    root = Path(root).resolve()
    feature_dirs = dict(feature_dirs or {})
    manifest_path = root / "metadata" / f"split_manifest_{config.get('manifest_version', 'v1')}.csv"
    if manifest_path.is_file():
        rows, manifest_sha = saved_manifest(root, config)
    else:
        if not extract_missing:
            raise FileNotFoundError("Restore the selected manifest/features before resume or final test")
        prepared = prepare_dataset(root, config, archive_path)
        rows, manifest_sha = read_manifest(prepared["manifest_path"]), prepared["manifest_sha256"]
    included, _, _ = ex._manifest_rows(rows)
    result = {}
    prepared = None
    for extractor in dict.fromkeys(extractors):
        explicit = feature_dirs.get(extractor)
        if explicit:
            candidates = [ex._root_path(root, explicit)]
        else:
            parent = root / config.get("features_dir", "features") / extractor
            candidates = [p.parent for p in parent.glob("*/metadata_train.json")]
        valid = []
        for directory in candidates:
            if not directory.resolve().is_relative_to(root):
                raise ValueError("Copy the feature store inside the workspace before portable backup")
            signatures = ex._feature_signatures(directory)
            ex._verify_store_metadata(directory, signatures)
            ex._load_aligned(directory, "train", included, extractor, manifest_sha, config)
            ex._load_aligned(directory, "val", included, extractor, manifest_sha, config)
            valid.append(directory)
        if len(valid) > 1:
            raise ValueError(f"Multiple feature caches for {extractor}; choose FEATURE_DIRS explicitly")
        if valid:
            result[extractor] = valid[0]
            print(f"Verified saved {extractor} features; no extraction and no test scoring", flush=True)
        else:
            if explicit or not extract_missing:
                raise FileNotFoundError(f"Explicit feature store is missing for {extractor}")
            if prepared is None:
                prepared = prepare_dataset(root, config, archive_path)
                if prepared["manifest_sha256"] != manifest_sha:
                    raise ValueError("Fresh preparation differs from the saved manifest")
            from .features import extract_features
            result[extractor] = extract_features(root, config, rows, Path(prepared["data_root"]), manifest_sha, extractor)
        if checkpoint_callback:
            checkpoint_callback(f"features_{extractor}", None)
    return rows, manifest_sha, result


def analysis_tables(root, config, *, engine, run_id=None):
    """Read reports only; never fit, prepare/download data, or open test arrays."""
    import pandas as pd
    output = Path(root) / config.get("results_dir", "results")
    tables = {}
    if engine == "classical":
        if (output / "comparison.csv").is_file():
            tables["comparison"] = pd.read_csv(output / "comparison.csv")
        selection = output / "selection.json"
        if selection.is_file():
            record = ex._read_json(selection)
            for experiment in record["experiments"]:
                directory = ex._root_path(Path(root), experiment["validation_output_dir"])
                if (directory / "per_class.csv").is_file():
                    tables[experiment["experiment_id"] + "_per_class"] = pd.read_csv(directory / "per_class.csv").sort_values("f1")
    elif engine == "sgd_epoch":
        from .epoch_training import _run_directory
        if run_id is None:
            tables["available_runs"] = pd.DataFrame([ex._read_json(p) for p in sorted((output / "epoch_runs").glob("*/status.json"))])
        else:
            directory = _run_directory(Path(root).resolve(), config, run_id)
            for name, relative in (("history", "history.csv"), ("per_class", "validation/per_class.csv"),
                                    ("confusion_matrix", "validation/confusion_matrix.csv"), ("errors", "validation/errors.csv")):
                if (directory / relative).is_file():
                    tables[name] = pd.read_csv(directory / relative)
            if "per_class" in tables:
                tables["per_class"] = tables["per_class"].sort_values("f1")
            if (directory / "selection.json").is_file():
                selected = ex._read_json(directory / "selection.json")
                metrics = selected["validation_metrics"]["primary"]
                tables["selected_validation"] = pd.DataFrame([{
                    "run_id": run_id, "best_epoch": selected["best_epoch"],
                    "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
                    "n_representatives": metrics["n_samples"],
                }])
    else:
        raise ValueError("ENGINE must be classical or sgd_epoch")
    if (output / "final_test/results.json").is_file():
        record = ex._read_json(output / "final_test/results.json")
        if record.get("engine") == "sgd_epoch":
            tables["locked_final_test"] = pd.DataFrame([record["metrics"]["primary"]]).drop(columns=["per_class", "confusion_matrix", "labels"])
    return tables
