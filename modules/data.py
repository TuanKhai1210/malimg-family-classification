"""Verified public ZIP download, selective PNG extraction and versioned manifests.

Only raster images are extracted. Archive integrity, relative paths, image
decoding, exact pixel conflicts, and existing artifact provenance are checked
before a dataset is returned. Reuse never silently trusts an existing folder.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import urllib.request
import uuid
import warnings
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from .splits import DEFAULT_TARGETS, SPLIT_ALGORITHM, assign_group_splits, summarize_split, validate_targets


CHUNK_BYTES = 1024 * 1024
PREPARATION_SCHEMA = "malimg-preparation-v1"
PIXEL_HASH_SCHEME = "sha256(mode:widthxheight: + decoded pixel bytes)"
MANIFEST_FIELDS = (
    "sample_id", "relative_path", "archive_path", "label", "label_id", "width", "height", "mode",
    "file_bytes", "file_sha256", "pixel_sha256", "included", "exclusion_reason", "split", "eval_representative",
)
INTEGER_FIELDS = {"label_id", "width", "height", "file_bytes"}
BOOLEAN_FIELDS = {"included", "eval_representative"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _archive_expectation(config: dict[str, Any]) -> tuple[int, str]:
    expected_bytes = int(config["archive_bytes"])
    expected_sha256 = str(config["archive_sha256"]).lower()
    if expected_bytes <= 0 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("archive_bytes and archive_sha256 must identify the audited archive")
    return expected_bytes, expected_sha256


def verify_archive(path: str | Path, expected_bytes: int, expected_sha256: str) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"archive must be a regular local file: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_bytes:
        raise ValueError(f"archive size mismatch: {actual_size} != {expected_bytes}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_sha256:
        raise ValueError(f"archive SHA-256 mismatch: {actual_hash}")
    return actual_hash


def download_archive(config: dict[str, Any], destination: str | Path) -> Path:
    """Download by public HTTPS with resumable temporary storage and integrity checks."""
    destination = Path(destination)
    expected_bytes, expected_hash = _archive_expectation(config)
    if destination.exists():
        verify_archive(destination, expected_bytes, expected_hash)
        return destination
    url = str(config.get("dataset_url", ""))
    if not url.startswith("https://"):
        raise ValueError("dataset_url must be a public HTTPS URL")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.is_symlink():
        raise ValueError("refusing a symlink at the temporary download path")
    existing = partial.stat().st_size if partial.exists() else 0
    if existing > expected_bytes:
        raise ValueError("partial archive exceeds expected size; inspect it before retrying")
    if existing < expected_bytes:
        headers = {"User-Agent": "MalImgCourseProject/1.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=90) as response:
            status = getattr(response, "status", 200)
            if existing and status == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {existing}-"):
                    raise ValueError("download server returned an unexpected resume range")
                mode = "ab"
            elif status == 200:
                mode = "wb"  # Public hosts may ignore Range; restart our temporary file.
                existing = 0
            else:
                raise ValueError(f"unexpected download response: HTTP {status}")
            with partial.open(mode) as output:
                downloaded = existing
                while True:
                    block = response.read(CHUNK_BYTES)
                    if not block:
                        break
                    downloaded += len(block)
                    if downloaded > expected_bytes:
                        raise ValueError("download is larger than the audited archive")
                    output.write(block)
    verify_archive(partial, expected_bytes, expected_hash)
    os.replace(partial, destination)
    return destination


def _safe_archive_path(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename.replace("\\", "/")
    path = PurePosixPath(name)
    if (
        not name or path.is_absolute() or any(part in {".", ".."} for part in name.split("/"))
        or any(any(character in '<>:"|?*' or ord(character) < 32 for character in part) or part.endswith((".", " ")) for part in path.parts)
        or any(re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part) for part in path.parts)
    ):
        raise ValueError(f"unsafe archive path: {info.filename!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
        raise ValueError(f"non-regular archive entry: {info.filename!r}")
    return path


def _image_entries(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str, str]]:
    entries = []
    archive_names: set[str] = set()
    output_names: set[str] = set()
    family_names: dict[str, str] = {}
    for info in archive.infolist():
        path = _safe_archive_path(info)  # Check ignored files as well as selected images.
        name_key = str(path).casefold()
        if name_key in archive_names:
            raise ValueError(f"repeated archive path: {info.filename!r}")
        archive_names.add(name_key)
        if info.is_dir() or path.suffix.lower() != ".png":
            continue
        if len(path.parts) < 2:
            raise ValueError(f"PNG has no family directory: {info.filename!r}")
        label = path.parent.name
        previous_family = family_names.setdefault(label.casefold(), label)
        if previous_family != label:
            raise ValueError(f"family names differ only in case: {previous_family} and {label}")
        relative = f"{label}/{path.name}"
        if relative.casefold() in output_names:
            raise ValueError(f"PNG output path collision: {relative}")
        output_names.add(relative.casefold())
        entries.append((info, relative, label))
    if not entries:
        raise ValueError("archive contains no family PNG images")
    return sorted(entries, key=lambda item: item[1])


def _decode_inventory(raw: bytes, relative: str, label: str, archive_path: str, allowed_modes: list[str]) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG":
                raise ValueError(f"file has .png extension but is not PNG: {relative}")
            image.load()
            if image.mode not in allowed_modes:
                raise ValueError(f"unexpected image mode {image.mode} at {relative}; allowed {allowed_modes}")
            width, height = image.size
            if width < 1 or height < 1:
                raise ValueError(f"empty image: {relative}")
            mode = image.mode
            pixel_hash = hashlib.sha256(f"{mode}:{width}x{height}:".encode("utf-8") + image.tobytes()).hexdigest()
    return {
        "sample_id": "sample_" + hashlib.sha256(relative.encode("utf-8")).hexdigest(),
        "relative_path": relative, "archive_path": archive_path, "label": label,
        "width": width, "height": height, "mode": mode, "file_bytes": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(), "pixel_sha256": pixel_hash,
    }


def _check_pixel_conflicts(rows: list[dict[str, Any]]) -> None:
    labels: dict[str, str] = {}
    for row in rows:
        previous = labels.setdefault(row["pixel_sha256"], row["label"])
        if previous != row["label"]:
            raise ValueError(f"cross-label pixel conflict: {previous} and {row['label']}")


def _verify_existing_images(data_root: Path, rows: list[dict[str, Any]]) -> None:
    expected = {row["relative_path"]: row for row in rows}
    actual: dict[str, Path] = {}
    for path in data_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink found in existing image directory: {path}")
        if path.is_file():
            actual[path.relative_to(data_root).as_posix()] = path
    if set(actual) != set(expected):
        raise ValueError("existing images do not match the verified source inventory")
    for relative, path in actual.items():
        row = expected[relative]
        if path.stat().st_size != row["file_bytes"] or sha256_file(path) != row["file_sha256"]:
            raise ValueError(f"existing image integrity mismatch: {relative}")


def _extract_inventory(archive_path: Path, data_root: Path, archive_hash: str, allowed_modes: list[str]) -> list[dict[str, Any]]:
    data_directory = data_root.parent
    source_manifest = data_directory / "image_inventory.json"
    provenance_path = data_directory / "extraction_provenance.json"
    identity = {"archive_sha256": archive_hash, "allowed_image_modes": sorted(allowed_modes), "pixel_hash_scheme": PIXEL_HASH_SCHEME, "schema": PREPARATION_SCHEMA}
    if data_root.exists():
        if data_root.is_symlink() or not data_root.is_dir() or source_manifest.is_symlink() or provenance_path.is_symlink() or not source_manifest.is_file() or not provenance_path.is_file():
            raise ValueError("existing image directory has no valid extraction provenance; inspect it before reuse")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("identity") != identity or provenance.get("inventory_sha256") != sha256_file(source_manifest):
            raise ValueError("existing extraction provenance or source inventory does not match")
        rows = json.loads(source_manifest.read_text(encoding="utf-8"))
        _verify_existing_images(data_root, rows)
        _check_pixel_conflicts(rows)
        return rows
    if source_manifest.exists() or provenance_path.exists():
        raise ValueError("incomplete extraction artifacts exist without images; inspect them before retrying")

    stage = data_directory / f".images-{uuid.uuid4().hex}.tmp"
    stage.mkdir(parents=True)
    rows = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info, relative, label in _image_entries(archive):
                raw = archive.read(info)  # The ZIP reader verifies CRC on each selected image.
                row = _decode_inventory(raw, relative, label, info.filename, allowed_modes)
                target = stage.joinpath(*PurePosixPath(relative).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                rows.append(row)
        _check_pixel_conflicts(rows)
        data_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, data_root)
        _atomic_write(source_manifest, _canonical_json(rows))
        _atomic_write(provenance_path, _canonical_json({"identity": identity, "inventory_sha256": sha256_file(source_manifest)}))
    finally:
        # Only the unique staging folder created above can be removed here.
        if stage.exists() and stage.resolve().parent == data_directory.resolve() and stage.name.startswith(".images-"):
            shutil.rmtree(stage)
    return rows


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        serialized = {key: row[key] for key in MANIFEST_FIELDS}
        for key in BOOLEAN_FIELDS:
            serialized[key] = "true" if row[key] else "false"
        writer.writerow(serialized)
    return output.getvalue().encode("utf-8")


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read typed manifest rows without silently accepting invalid booleans or paths."""
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not set(MANIFEST_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("manifest is missing required data fields")
        rows: list[dict[str, Any]] = []
        for record in reader:
            row: dict[str, Any] = dict(record)
            for key in INTEGER_FIELDS:
                row[key] = int(row[key])
            for key in BOOLEAN_FIELDS:
                value = str(row[key]).lower()
                if value not in {"true", "false"}:
                    raise ValueError(f"invalid boolean in {key}: {row[key]!r}")
                row[key] = value == "true"
            relative = str(row["relative_path"])
            path_object = PurePosixPath(relative)
            if "\\" in relative or path_object.is_absolute() or ".." in path_object.parts or ":" in relative:
                raise ValueError(f"unsafe manifest relative_path: {relative}")
            if row["width"] < 1 or row["height"] < 1 or row["file_bytes"] < 1:
                raise ValueError("manifest dimensions and file size must be positive")
            if row["included"] and (row["label_id"] < 0 or row["split"] not in {"train", "val", "test"}):
                raise ValueError("included row has an invalid label_id or split")
            if not row["included"] and (row["split"] != "excluded" or row["label_id"] != -1):
                raise ValueError("excluded row must have split=excluded and label_id=-1")
            rows.append(row)
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("manifest must contain unique, nonempty sample IDs")
    if any(not row["sample_id"] for row in rows):
        raise ValueError("sample IDs cannot be empty")
    return rows


def prepare_dataset(root: Path, config: dict[str, Any], archive_path: Path | None = None) -> dict[str, Any]:
    """Prepare a versioned, deterministic dataset and verify it again on reuse.

    Existing artifacts are immutable for a manifest version. Change the version
    after a reviewed config change; tampered files cause an explicit failure.
    The result describes generated engineering artifacts, not instructor approval.
    """
    root = Path(root).resolve()
    expected_bytes, expected_hash = _archive_expectation(config)
    targets = validate_targets(config.get("split_target", DEFAULT_TARGETS))
    version = str(config.get("manifest_version", "v1"))
    if not re.fullmatch(r"v[0-9]+", version):
        raise ValueError("manifest_version must be a version such as v1 or v2")
    excluded = sorted(set(map(str, config.get("excluded_families", ["Yuner.A"]))))
    allowed_modes = sorted(set(map(str, config.get("allowed_image_modes", ["L"]))))
    if not allowed_modes:
        raise ValueError("allowed_image_modes cannot be empty")
    configured_data = Path(str(config.get("data_dir", "data/images")))
    data_root = configured_data if configured_data.is_absolute() else root / configured_data
    if data_root.is_symlink() or data_root.parent.is_symlink():
        raise ValueError("data_dir must not be a symlink")
    data_root = data_root.resolve()
    data_directory = data_root.parent
    actual_archive = Path(archive_path).resolve() if archive_path is not None else root / "data" / "MalImg_original_dataset.zip"
    if archive_path is None:
        download_archive(config, actual_archive)
    archive_hash = verify_archive(actual_archive, expected_bytes, expected_hash)
    source_rows = _extract_inventory(actual_archive, data_root, archive_hash, allowed_modes)
    labels = sorted({row["label"] for row in source_rows} - set(excluded))
    label_map = {label: index for index, label in enumerate(labels)}
    representatives: dict[str, str] = {}
    for row in source_rows:
        pixel_hash = row["pixel_sha256"]
        representatives[pixel_hash] = min(representatives.get(pixel_hash, row["relative_path"]), row["relative_path"])
    dataset_rows = []
    for source in source_rows:
        row = dict(source)
        included = row["label"] not in excluded
        row.update({
            "included": included, "label_id": label_map.get(row["label"], -1),
            "exclusion_reason": "excluded_family_by_project_config" if not included else "",
            "split": "excluded" if not included else "",
            "eval_representative": row["relative_path"] == representatives[row["pixel_sha256"]],
        })
        dataset_rows.append(row)
    split_rows = assign_group_splits(dataset_rows, seed=int(config.get("split_seed", 42)), targets=targets)
    assignments = {row["sample_id"]: row for row in split_rows}
    dataset_rows = [assignments.get(row["sample_id"], row) for row in dataset_rows]
    paths = {
        "dataset_manifest": root / "metadata" / f"dataset_manifest_{version}.csv",
        "manifest_path": root / "metadata" / f"split_manifest_{version}.csv",
        "label_map_path": root / "metadata" / f"label_map_{version}.json",
        "summary_path": root / "metadata" / f"split_summary_{version}.json",
    }
    provenance_path = root / "metadata" / f"preparation_provenance_{version}.json"
    preparation_identity = {
        "schema": PREPARATION_SCHEMA, "manifest_version": version, "archive_sha256": archive_hash,
        "archive_bytes": expected_bytes, "excluded_families": excluded, "allowed_image_modes": allowed_modes,
        "split_seed": int(config.get("split_seed", 42)), "split_target": targets, "algorithm": SPLIT_ALGORITHM,
        "source_inventory_sha256": sha256_file(data_directory / "image_inventory.json"),
    }
    fingerprint = _fingerprint(preparation_identity)
    summary = summarize_split(split_rows, targets)
    pixel_counts = Counter(row["pixel_sha256"] for row in source_rows)
    summary.update({
        "manifest_version": version, "split_seed": preparation_identity["split_seed"],
        "archive_sha256": archive_hash, "archive_bytes": expected_bytes,
        "dataset_url": config.get("dataset_url", ""), "pixel_hash_scheme": PIXEL_HASH_SCHEME,
        "source_files": len(source_rows), "source_families": len({row["label"] for row in source_rows}),
        "excluded_files": len(source_rows) - len(split_rows), "excluded_families": excluded,
        "source_pixel_groups": len(pixel_counts), "source_pixel_duplicate_groups": sum(count > 1 for count in pixel_counts.values()),
        "source_duplicate_extra_files": sum(count - 1 for count in pixel_counts.values()),
        "preparation_fingerprint": fingerprint,
        "approval_status": config.get("approval_status", "instructor approval and team split review pending"),
    })
    content = {
        "dataset_manifest": _csv_bytes(dataset_rows), "manifest_path": _csv_bytes(split_rows),
        "label_map_path": _canonical_json(label_map),
    }
    manifest_hash = hashlib.sha256(content["manifest_path"]).hexdigest()
    summary["manifest_sha256"] = manifest_hash
    summary["dataset_manifest_sha256"] = hashlib.sha256(content["dataset_manifest"]).hexdigest()
    summary["label_map_sha256"] = hashlib.sha256(content["label_map_path"]).hexdigest()
    content["summary_path"] = _canonical_json(summary)

    existing = [path.exists() for path in paths.values()]
    if any(existing) or provenance_path.exists():
        if not all(existing) or not provenance_path.is_file():
            raise ValueError("incomplete preparation artifacts; inspect them before regenerating")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_symlink() or provenance.get("preparation_fingerprint") != fingerprint:
            raise ValueError("preparation config changed for an existing manifest version; review and increment manifest_version")
        for key, path in paths.items():
            expected = hashlib.sha256(content[key]).hexdigest()
            if path.is_symlink() or sha256_file(path) != expected or provenance.get("artifacts", {}).get(path.name) != expected:
                raise ValueError(f"preparation artifact integrity mismatch: {path.name}")
    else:
        for key, path in paths.items():
            _atomic_write(path, content[key])
        _atomic_write(provenance_path, _canonical_json({
            "preparation_fingerprint": fingerprint, "identity": preparation_identity,
            "artifacts": {path.name: hashlib.sha256(content[key]).hexdigest() for key, path in paths.items()},
        }))
    return {
        "archive_path": str(actual_archive), "data_root": str(data_root),
        **{key: str(path) for key, path in paths.items()},
        "manifest_sha256": manifest_hash, "manifest_version": version,
        "provenance_path": str(provenance_path), "summary": summary,
    }
