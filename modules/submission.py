"""Create a portable draft or a submission archive from real artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, reference: str) -> Path:
    path = (root / reference).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Package artifact must be a local file inside the repository: {reference}")
    return path


def build_submission(root: str | Path, output: str | Path, *, draft: bool = True,
                     include_features: bool = True) -> Path:
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("output must be a .zip file")
    if not draft:
        config = json.loads((root / "configs/project.json").read_text(encoding="utf-8"))
        if not config.get("instructor_approved", False):
            raise ValueError("Record instructor approval in project.json before creating the final package")
        report = root / "reports/final_report.pdf"
        if not report.is_file() or not report.read_bytes().startswith(b"%PDF-"):
            raise ValueError("Final package requires reports/final_report.pdf")
        results = root / config.get("results_dir", "results")
        if not (results / "final_test.lock.json").is_file() or not (results / "comparison.csv").is_file():
            raise ValueError("Final package requires completed final evaluation and comparison.csv")
        lock = json.loads((results / "final_test.lock.json").read_text(encoding="utf-8"))
        result_file = results / "final_test/results.json"
        if lock.get("status") != "completed" or not result_file.is_file():
            raise ValueError("Final evaluation has not completed")
        if hashlib.sha256(result_file.read_bytes()).hexdigest() != lock.get("results_sha256"):
            raise ValueError("Final results have changed since evaluation was locked")
        selection_path = results / "selection.json"
        if not selection_path.is_file() or _sha256(selection_path) != lock.get("selection_sha256"):
            raise ValueError("Final selection is missing or has changed")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != selection.get("config_sha256"):
            raise ValueError("Current configuration differs from the locked final experiment")
        if _sha256(_inside(root, selection["selection_audit_path"])) != selection["selection_audit_sha256"]:
            raise ValueError("Validation selection audit has changed")
        for store in selection["feature_stores"].values():
            directory = (root / store["directory"]).resolve()
            if not directory.is_relative_to(root / "features"):
                raise ValueError("Final feature stores must be inside features/ for portable packaging")
            for split, checksums in store["file_sha256"].items():
                filenames = {"X": f"X_{split}.npy", "y": f"y_{split}.npy",
                             "sample_ids": f"sample_ids_{split}.json", "metadata": f"metadata_{split}.json"}
                if split == "common":
                    filenames = {"metadata": "metadata.json", "completion": "complete.json"}
                for kind, checksum in checksums.items():
                    path = _inside(root, str(directory / filenames[kind]))
                    if _sha256(path) != checksum:
                        raise ValueError(f"Selected feature artifact has changed: {path.name}")
        for experiment in selection["experiments"]:
            if _sha256(_inside(root, experiment["model_path"])) != experiment["model_sha256"]:
                raise ValueError("Selected model has changed")
        if not include_features:
            raise ValueError("Include the saved features in the final package; external links need manual review")
        if not selection.get("feature_stores") or not any((root / "features").rglob("X_train.npy")):
            raise ValueError("Final package requires saved feature arrays")

    members = {}
    for name in ("README.md", "CONTRIBUTING.md", "AGENTS.md", "requirements.txt", "requirements-dev.txt", ".python-version"):
        path = root / name
        if path.is_file():
            members[name] = path
    folders = ["notebooks", "modules", "scripts", "configs", "docs", "metadata", "reports", "tests"]
    if include_features:
        folders.append("features")
    for folder in folders:
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or path.is_symlink() or path.resolve() == output:
                continue
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or ".ipynb_checkpoints" in relative.parts or path.suffix in {".pyc", ".pyo", ".zip"}:
                continue
            members[relative.as_posix()] = path
    config_file = root / "configs/project.json"
    config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.is_file() else {}
    result_root = root / config.get("results_dir", "results")
    for path in result_root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".png", ".joblib"} and not path.is_symlink():
            members[path.relative_to(root).as_posix()] = path
    manifest = {name: _sha256(path) for name, path in sorted(members.items())}
    source_commit = None
    if (root / ".git").exists():
        source_commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in ("notebooks", "modules", "reports", "features"):
            archive.writestr(folder + "/", b"")
        for name, path in sorted(members.items()):
            archive.write(path, name)
        archive.writestr("PACKAGE.json", json.dumps({"draft": draft, "includes_features": include_features,
                          "source_commit": source_commit, "trusted_generated_models_included": True,
                          "warning": "Draft for review; not a completed course submission" if draft else "Review course information, report, feature alignment and access before submitting",
                          "files_sha256": manifest}, indent=2))
    temporary.replace(output)
    return output
