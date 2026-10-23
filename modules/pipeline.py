"""Shared orchestration used by Colab and the command-line front end."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import validate_config


def run_pipeline(
    root: str | Path, config: dict[str, Any], *, stage: str = "prepare",
    archive_path: str | Path | None = None, final_test: bool = False,
    resume_run: str | Path | None = None, backup_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare, extract or train; final test always requires an explicit choice.

    `all` executes preparation, EDA, extraction and validation selection.
    Pass final_test=True only after the group freezes the experimental choices.
    """
    validate_config(config)
    if stage not in {"prepare", "eda", "extract", "train", "final", "all"}:
        raise ValueError(f"unknown pipeline stage: {stage}")
    if resume_run is not None and stage not in {"train", "all"}:
        raise ValueError("resume_run is only used with train or all")
    root = Path(root).resolve()
    def checkpoint(label, run_dir=None):
        if backup_dir is not None:
            from .checkpoints import save_backup
            run_label = f"{run_dir.name}_{label}" if run_dir else label
            save_backup(root, backup_dir, label=run_label, config=config)
    if stage in {"train", "all"}:
        results = root / config.get("results_dir", "results")
        if (results / "final_test.lock.json").exists() or (results / "final_test").exists():
            raise ValueError("Final test artifacts already exist. Use the final stage to read the locked result; review a new protocol before changing the selection.")
    from .data import prepare_dataset, read_manifest

    prepared = prepare_dataset(root, config, Path(archive_path) if archive_path else None)
    rows = read_manifest(prepared["manifest_path"])
    result: dict[str, Any] = {"stage": stage, "dataset": prepared,
                              "instructor_approved": config.get("instructor_approved", False)}
    if stage in {"prepare", "eda", "all"}:
        from .eda import generate_eda

        inventory = read_manifest(prepared["dataset_manifest"])
        result["eda"] = generate_eda(inventory, Path(prepared["data_root"]),
                                     root / config.get("results_dir", "results") / "eda")
    if stage in {"extract", "train", "all"}:
        from .features import extract_features

        feature_dirs = {}
        for name in dict.fromkeys(experiment["extractor"] for experiment in config["experiments"]):
            print(f"Extracting/reusing {name} features", flush=True)
            feature_dirs[name] = extract_features(root, config, rows, Path(prepared["data_root"]),
                                                  prepared["manifest_sha256"], name)
            checkpoint(f"features_{name}")
        result["feature_dirs"] = {name: str(path) for name, path in feature_dirs.items()}
        if stage in {"train", "all"}:
            from .experiments import run_validation

            result["validation"] = run_validation(root, config, rows, prepared["manifest_sha256"], feature_dirs,
                                                   resume_run=resume_run,
                                                   checkpoint_callback=checkpoint if backup_dir else None)
    if stage == "final" or (stage == "all" and final_test):
        from .experiments import run_final_test

        selection = root / config.get("results_dir", "results") / "selection.json"
        if not selection.is_file():
            raise ValueError("No locked validation selection. Run the train stage before final test.")
        result["final"] = run_final_test(root, config, rows, prepared["manifest_sha256"], selection)
    if stage == "all" and not final_test:
        result["test_status"] = "held out; run the final stage after freezing the choices"
    status = root / config.get("results_dir", "results") / "pipeline_status.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    checkpoint(f"pipeline_{stage}_completed")
    return result
