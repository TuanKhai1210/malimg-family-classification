"""Descriptive dataset quality figures and tables, independent of model scores."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image


def _distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "min": ordered[0], "median": statistics.median(ordered),
        "p95": ordered[round(0.95 * (len(ordered) - 1))], "max": ordered[-1],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _safe_image(data_root: Path, relative: str) -> Path:
    path_object = PurePosixPath(relative)
    if path_object.is_absolute() or ".." in path_object.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"unsafe EDA image path: {relative}")
    path = data_root.joinpath(*path_object.parts)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(data_root.resolve()):
        raise ValueError(f"EDA image is missing or outside data_root: {relative}")
    return path


def generate_eda(rows: list[dict[str, Any]], data_root: Path, output_dir: Path) -> dict[str, Any]:
    """Write counts, dimensions, duplicate profiles and labeled example figures.

    Supply the dataset manifest to include excluded families in the quality
    audit, or the split manifest to describe the included benchmark. These are
    descriptive checks; test performance is never used to choose a model.
    """
    if not rows:
        raise ValueError("EDA requires nonempty manifest rows")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_root, output_dir = Path(data_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["width"]) < 1 or int(row["height"]) < 1:
            raise ValueError("EDA received invalid image dimensions")
        classes[str(row["label"])].append(row)
    counts = []
    duplicate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        duplicate_groups[str(row["pixel_sha256"])].append(row)
    for pixel_hash, members in duplicate_groups.items():
        if len({row["label"] for row in members}) > 1:
            raise ValueError(f"cross-label pixel conflict in EDA: {pixel_hash}")
        if len({row["split"] for row in members}) > 1:
            raise ValueError(f"pixel group crosses partitions in EDA: {pixel_hash}")
    for label in sorted(classes):
        members = classes[label]
        groups = len({row["pixel_sha256"] for row in members})
        counts.append({
            "label": label, "label_id": int(members[0]["label_id"]),
            "included": bool(members[0].get("included", True)), "files": len(members),
            "pixel_groups": groups, "duplicate_extra_files": len(members) - groups,
            "duplicate_extra_fraction": (len(members) - groups) / len(members),
        })
    split_counts = []
    for label in sorted(classes):
        for split in sorted({str(row["split"]) for row in classes[label]}):
            partition = [row for row in classes[label] if row["split"] == split]
            split_counts.append({
                "label": label, "split": split, "files": len(partition),
                "pixel_groups": len({row["pixel_sha256"] for row in partition}),
                "representatives": sum(bool(row["eval_representative"]) for row in partition),
            })
    duplicate_rows = []
    for pixel_hash, members in sorted(duplicate_groups.items()):
        if len(members) <= 1:
            continue
        duplicate_rows.append({
            "pixel_sha256": pixel_hash, "label": members[0]["label"], "files": len(members),
            "split": members[0]["split"], "representative_path": min(row["relative_path"] for row in members),
        })
    widths = [int(row["width"]) for row in rows]
    heights = [int(row["height"]) for row in rows]
    aspects = [height / width for width, height in zip(widths, heights)]
    summary = {
        "files": len(rows), "classes": len(classes), "pixel_groups": len(duplicate_groups),
        "included_files": sum(bool(row.get("included", True)) for row in rows),
        "excluded_files": sum(not row.get("included", True) for row in rows),
        "image_modes": dict(sorted(Counter(str(row["mode"]) for row in rows).items())),
        "width": _distribution(widths), "height": _distribution(heights), "height_width_ratio": _distribution(aspects),
        "pixel_duplicate_groups": len(duplicate_rows), "duplicate_extra_files": len(rows) - len(duplicate_groups),
        "pixel_hash_definition": "mode + dimensions + decoded pixel bytes; exact duplicates only",
        "purpose": "dataset quality and descriptive audit; model choices use train/validation only",
        "example_policy": "lexicographically smallest included train image per family when available",
        "statistical_independence_claimed": False,
    }
    paths = {
        "summary_path": output_dir / "eda_summary.json",
        "class_counts_path": output_dir / "class_counts.csv",
        "split_counts_path": output_dir / "split_counts.csv",
        "dimensions_path": output_dir / "image_dimensions.csv",
        "duplicate_groups_path": output_dir / "pixel_duplicates.csv",
        "class_distribution_figure": output_dir / "class_distribution.png",
        "dimensions_figure": output_dir / "image_dimensions.png",
        "examples_figure": output_dir / "family_examples.png",
        "duplicate_profile_figure": output_dir / "duplicate_profile.png",
    }
    paths["summary_path"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(paths["class_counts_path"], counts, list(counts[0]))
    _write_csv(paths["split_counts_path"], split_counts, list(split_counts[0]))
    _write_csv(paths["duplicate_groups_path"], duplicate_rows, ["pixel_sha256", "label", "files", "split", "representative_path"])
    dimension_rows = [{"sample_id": row["sample_id"], "label": row["label"], "split": row["split"], "width": row["width"], "height": row["height"], "mode": row["mode"], "height_width_ratio": int(row["height"]) / int(row["width"])} for row in rows]
    _write_csv(paths["dimensions_path"], dimension_rows, list(dimension_rows[0]))

    figure, axis = plt.subplots(figsize=(max(10, len(counts) * 0.42), 5.5))
    positions = list(range(len(counts)))
    axis.bar([position - 0.2 for position in positions], [item["files"] for item in counts], width=0.4, label="Files")
    axis.bar([position + 0.2 for position in positions], [item["pixel_groups"] for item in counts], width=0.4, label="Exact pixel groups")
    axis.set_xticks(positions, [item["label"] + (" *" if not item["included"] else "") for item in counts], rotation=65, ha="right")
    axis.set_ylabel("Count")
    axis.set_title("Family distribution (* excluded family)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["class_distribution_figure"], dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, values, title in zip(axes, (widths, heights, aspects), ("Width (px)", "Height (px)", "Height / width")):
        axis.hist(values, bins=min(30, max(1, len(set(values)))), color="#386cb0", edgecolor="white")
        axis.set_xlabel(title)
        axis.set_ylabel("Files")
    figure.suptitle("Image dimension distributions (before preprocessing)")
    figure.tight_layout()
    figure.savefig(paths["dimensions_figure"], dpi=170)
    plt.close(figure)

    columns = min(5, len(classes))
    figure, axes = plt.subplots(math.ceil(len(classes) / columns), columns, figsize=(columns * 2.6, math.ceil(len(classes) / columns) * 2.8), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, label in zip(axes.flat, sorted(classes)):
        train = [row for row in classes[label] if row["split"] == "train"]
        candidate = min(train or classes[label], key=lambda row: str(row["relative_path"]))
        image_path = _safe_image(data_root, str(candidate["relative_path"]))
        with Image.open(image_path) as image:
            image.load()
            axis.imshow(image, cmap="gray", vmin=0, vmax=255, aspect="auto")
        axis.set_title(f"{label}\n{candidate['width']} × {candidate['height']} · {candidate['split']}", fontsize=9)
    figure.suptitle("Family examples (display scaled; original dimensions shown)")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(paths["examples_figure"], dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(max(10, len(counts) * 0.42), 4.5))
    axis.bar([item["label"] for item in counts], [100 * item["duplicate_extra_fraction"] for item in counts], color="#d95f02")
    axis.set_ylabel("Extra duplicate files (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Exact pixel duplication by family")
    axis.tick_params(axis="x", rotation=65)
    figure.tight_layout()
    figure.savefig(paths["duplicate_profile_figure"], dpi=170)
    plt.close(figure)
    return {**{key: str(path) for key, path in paths.items()}, "summary": summary}
