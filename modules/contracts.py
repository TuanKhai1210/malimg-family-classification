"""Shared data contracts; these checks must be reused by A, B and C."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence


SPLITS = ("train", "val", "test")
REQUIRED_MANIFEST_FIELDS = (
    "sample_id", "relative_path", "label", "label_id", "pixel_sha256", "split"
)


def validate_sample_ids(sample_ids: Sequence[str]) -> None:
    """Reject IDs that cannot join features, labels and predictions unambiguously."""
    if not sample_ids or any(not str(value).strip() for value in sample_ids):
        raise ValueError("sample_ids must be nonempty strings")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")


def validate_split_manifest(rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
    """Check the minimum split contract after A creates its versioned manifest.

    This catches exact-pixel leakage but cannot detect near-duplicates or confirm
    that the instructor has approved the dataset.
    """
    counts = {split: 0 for split in SPLITS}
    seen_ids: set[str] = set()
    pixel_splits: dict[str, set[str]] = defaultdict(set)
    label_splits: dict[str, set[str]] = defaultdict(set)
    labels_to_ids: dict[str, str] = {}
    total = 0
    for row in rows:
        missing = [key for key in REQUIRED_MANIFEST_FIELDS if key not in row]
        if missing:
            raise ValueError(f"manifest row missing fields: {missing}")
        sample_id = str(row["sample_id"])
        label = str(row["label"])
        label_id = str(row["label_id"])
        pixel_hash = str(row["pixel_sha256"])
        split = str(row["split"])
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"missing or repeated sample_id: {sample_id!r}")
        if not label or not pixel_hash or split not in SPLITS:
            raise ValueError(f"invalid label, pixel hash or split for {sample_id}")
        if label in labels_to_ids and labels_to_ids[label] != label_id:
            raise ValueError(f"label_id changed for {label}")
        seen_ids.add(sample_id)
        labels_to_ids[label] = label_id
        pixel_splits[pixel_hash].add(split)
        label_splits[label].add(split)
        counts[split] += 1
        total += 1
    if total == 0:
        raise ValueError("manifest is empty")
    leaked = [pixel_hash for pixel_hash, splits in pixel_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"pixel hash crosses partitions: {leaked[0]}")
    incomplete = [label for label, splits in label_splits.items() if splits != set(SPLITS)]
    if incomplete:
        raise ValueError(f"class missing from a partition: {incomplete[0]}")
    return counts
