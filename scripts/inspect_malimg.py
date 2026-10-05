"""Inventory a MalImg ZIP without executing files or extracting archive paths."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "data/MalImg_original_dataset.zip"
WORK = ROOT / "metadata"
MANIFEST = WORK / "malimg_manifest.csv"
SUMMARY = WORK / "malimg_summary.json"


def quantiles(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[round(0.95 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    other_entries: list[str] = []
    failures: list[dict[str, str]] = []
    classes = Counter()
    modes = Counter()
    widths: list[int] = []
    heights: list[int] = []
    byte_hashes: dict[str, list[int]] = defaultdict(list)
    pixel_hashes: dict[str, list[int]] = defaultdict(list)

    with zipfile.ZipFile(ARCHIVE) as archive:
        entries = archive.infolist()
        for info in entries:
            if info.is_dir():
                continue
            if not info.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                other_entries.append(info.filename)
                continue
            try:
                raw = archive.read(info)  # ZIP CRC is verified here.
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    width, height = image.size
                    mode = image.mode
                    pixel_hash = hashlib.sha256(
                        f"{mode}:{width}x{height}:".encode() + image.tobytes()
                    ).hexdigest()
                path = Path(info.filename.replace("\\", "/"))
                label = path.parent.name
                row = {
                    "sample_id": path.stem,
                    "archive_path": info.filename,
                    "label": label,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "compressed_bytes": info.compress_size,
                    "file_bytes": len(raw),
                    "file_sha256": hashlib.sha256(raw).hexdigest(),
                    "pixel_sha256": pixel_hash,
                }
                row_id = len(rows)
                rows.append(row)
                classes[label] += 1
                modes[mode] += 1
                widths.append(width)
                heights.append(height)
                byte_hashes[row["file_sha256"]].append(row_id)
                pixel_hashes[pixel_hash].append(row_id)
                if len(rows) % 1000 == 0:
                    print(f"validated={len(rows)}", flush=True)
            except (OSError, ValueError, UnidentifiedImageError, RuntimeError) as error:
                failures.append({"path": info.filename, "error": str(error)})

    duplicate_groups = [ids for ids in pixel_hashes.values() if len(ids) > 1]
    conflicting_groups = [
        {"paths": [rows[i]["archive_path"] for i in ids],
         "labels": sorted({str(rows[i]["label"]) for i in ids})}
        for ids in duplicate_groups
        if len({rows[i]["label"] for i in ids}) > 1
    ]
    summary = {
        "archive": str(ARCHIVE),
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "zip_entries": len(entries),
        "valid_images": len(rows),
        "class_count": len(classes),
        "class_counts": dict(sorted(classes.items())),
        "image_modes": dict(sorted(modes.items())),
        "width": quantiles(widths),
        "height": quantiles(heights),
        "unreadable_images": failures,
        "other_files": other_entries[:100],
        "other_file_count": len(other_entries),
        "file_duplicate_groups": sum(len(ids) > 1 for ids in byte_hashes.values()),
        "file_duplicate_extra_images": sum(len(ids) - 1 for ids in byte_hashes.values()),
        "pixel_duplicate_groups": len(duplicate_groups),
        "pixel_duplicate_extra_images": sum(len(ids) - 1 for ids in duplicate_groups),
        "cross_label_pixel_duplicate_groups": conflicting_groups,
    }
    with MANIFEST.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("class_counts", "unreadable_images", "other_files", "cross_label_pixel_duplicate_groups")}, indent=2), flush=True)
    print(f"summary={SUMMARY} manifest={MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
