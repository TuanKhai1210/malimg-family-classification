"""Portable backups to a mounted filesystem; no Drive account is required.

Feature bundles are immutable and shared between snapshots. Each snapshot ZIP
contains source/configuration, manifests and all saved runs, plus the identity
of its feature bundle. Only restore backups produced by your own group.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .submission import build_submission


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(name: str) -> str:
    parts = PurePosixPath(name).parts
    if (not parts or PurePosixPath(name).is_absolute() or "\\" in name
            or any(part in {"..", "."} or ":" in part for part in parts)):
        raise ValueError(f"Unsafe backup member: {name}")
    return name


def _verify(archive: zipfile.ZipFile, checksums: dict) -> None:
    members = [info.filename for info in archive.infolist() if not info.is_dir()]
    if len(members) != len(set(members)):
        raise ValueError("Duplicate backup members")
    for info in archive.infolist():
        _safe_name(info.filename)
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("Backup must not contain symlinks")
    expected = set(checksums) | {"PACKAGE.json", "BACKUP.json", "FEATURES.json"}
    if set(members) - expected:
        raise ValueError("Unlisted backup members")
    for name, checksum in checksums.items():
        _safe_name(name)
        digest = hashlib.sha256()
        with archive.open(name) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != checksum:
            raise ValueError(f"Backup checksum mismatch: {name}")


def _publish(source: Path, destination: Path) -> None:
    """Copy, read back and then publish; a failed write leaves old backups intact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.part")
    try:
        shutil.copyfile(source, temporary)
        if _sha(source) != _sha(temporary):
            raise OSError("Backup read-back checksum mismatch")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_backup(root: str | Path, destination: str | Path, *, label: str = "manual",
                config: dict | None = None) -> Path:
    """Save files, including completed features/models, without raw images/ZIPs.

    Call between pipeline stages or trials, never concurrently with another
    writer to the workspace. A snapshot becomes visible only after read-back.
    No backup is deleted automatically, and an I/O failure is surfaced to caller.
    """
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("Backup destination must be outside the repository")
    config_path = root / "configs/project.json"
    disk_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    config = config if config is not None else disk_config
    for key, default in (("results_dir", "results"), ("features_dir", "features")):
        directory = (root / config.get(key, default)).resolve()
        if not directory.is_relative_to(root):
            raise ValueError(f"Portable backups require {key} inside the repository")
        if directory != (root / disk_config.get(key, default)).resolve():
            raise ValueError(f"Persist {key} in configs/project.json before enabling backups")
    feature_root = (root / config.get("features_dir", "features")).resolve()
    if not feature_root.is_relative_to(root):
        raise ValueError("Portable backups require features_dir inside the repository")
    # Ignore extraction staging directories: they do not represent complete stores.
    feature_files = {path.relative_to(root).as_posix(): path
                     for path in feature_root.rglob("*")
                     if path.is_file() and not path.is_symlink()
                     and not any(part.startswith(".") for part in path.relative_to(feature_root).parts)}
    hashes = {name: _sha(path) for name, path in sorted(feature_files.items())}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    feature_zip = destination / "features" / f"{identity}.zip"
    with tempfile.TemporaryDirectory(prefix="malimg-backup-") as temporary:
        local = Path(temporary)
        if not feature_zip.is_file():
            bundle = local / "features.zip"
            with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, path in sorted(feature_files.items()):
                    archive.write(path, name)
                archive.writestr("FEATURES.json", json.dumps({"files_sha256": hashes}))
            # Detect files changed while copying, before accepting a backup.
            with zipfile.ZipFile(bundle) as archive:
                _verify(archive, hashes)
            _publish(bundle, feature_zip)
        else:
            with zipfile.ZipFile(feature_zip) as archive:
                if json.loads(archive.read("FEATURES.json"))["files_sha256"] != hashes:
                    raise ValueError("Feature bundle identity mismatch")
                _verify(archive, hashes)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", label)[:80] or "manual"
        snapshot = local / "snapshot.zip"
        build_submission(root, snapshot, draft=True, include_features=False)
        packages = {}
        for name in ("numpy", "scipy", "scikit-learn", "joblib", "torch", "torchvision",
                     "matplotlib", "pandas", "Pillow", "scikit-image"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        record = {"schema_version": 1, "created_at": stamp, "label": label,
                  "effective_config": config, "python": sys.version, "software": packages,
                  "feature_bundle": feature_zip.relative_to(destination).as_posix(),
                  "feature_bundle_sha256": _sha(feature_zip),
                  "warning": "Own-group backup; joblib is executable serialization"}
        with zipfile.ZipFile(snapshot, "a", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("BACKUP.json", json.dumps(record, indent=2))
        with zipfile.ZipFile(snapshot) as archive:
            _verify(archive, json.loads(archive.read("PACKAGE.json"))["files_sha256"])
        output = destination / "snapshots" / f"{stamp}_{safe_label}.zip"
        _publish(snapshot, output)
    print(f"Backup verified: {output}", flush=True)
    return output


def restore_backup(snapshot: str | Path, destination: str | Path) -> Path:
    """Verify both archives before restoring into a NEW directory.

    Keep the backup's features/ and snapshots/ directories together. This does
    not import or deserialize any model. Source and environment remain those
    recorded by the snapshot, not whatever happens to be on GitHub today.
    """
    snapshot, destination = Path(snapshot).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Restore destination must be a new directory")
    with zipfile.ZipFile(snapshot) as state:
        record = json.loads(state.read("BACKUP.json"))
        if record.get("schema_version") != 1:
            raise ValueError("Unsupported backup schema")
        package = json.loads(state.read("PACKAGE.json"))
        _verify(state, package["files_sha256"])
        relative = _safe_name(record["feature_bundle"])
        base = snapshot.parent.parent
        features_path = (base / relative).resolve()
        if not features_path.is_relative_to(base):
            raise ValueError("Feature bundle must be inside the backup directory")
        if _sha(features_path) != record["feature_bundle_sha256"]:
            raise ValueError("Feature bundle checksum mismatch")
        with zipfile.ZipFile(features_path) as features:
            checksums = json.loads(features.read("FEATURES.json"))["files_sha256"]
            _verify(features, checksums)
            if set(checksums) & set(package["files_sha256"]):
                raise ValueError("Snapshot and feature bundle overlap")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".restore-", dir=destination.parent) as temporary:
                staging = Path(temporary) / "workspace"
                staging.mkdir()
                for archive, names in ((state, package["files_sha256"]), (features, checksums)):
                    for name in names:
                        target = staging / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(name) as source, target.open("wb") as output:
                            shutil.copyfileobj(source, output)
                (staging / "PACKAGE.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
                (staging / "RESTORED_BACKUP.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                os.replace(staging, destination)
    print(f"Restored workspace: {destination}", flush=True)
    return destination
