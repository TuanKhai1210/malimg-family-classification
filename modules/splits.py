"""Deterministic, class-stratified partitions of exact decoded-pixel groups.

File proportions are targets, not guarantees: a duplicate group is indivisible.
The algorithm makes no use of image content beyond grouping and no model scores.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import SPLITS, validate_split_manifest


SPLIT_ALGORITHM = "class-group-file-target-greedy-refine-v1"
DEFAULT_TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}


def validate_targets(targets: Mapping[str, float]) -> dict[str, float]:
    """Require positive proportions for exactly train, validation and test."""
    if set(targets) != set(SPLITS):
        raise ValueError("split_target must contain exactly train, val and test")
    result = {split: float(targets[split]) for split in SPLITS}
    if any(not 0 < value < 1 for value in result.values()):
        raise ValueError("split proportions must be finite values between zero and one")
    if abs(sum(result.values()) - 1.0) > 1e-9:
        raise ValueError("split proportions must sum to one")
    return result


def _tie_key(seed: int, label: str, pixel_hash: str) -> str:
    return hashlib.sha256(f"{seed}:{label}:{pixel_hash}".encode("utf-8")).hexdigest()


def _objective(counts: Mapping[str, int], desired: Mapping[str, float]) -> float:
    # Scaling by desired files gives comparable pressure for the three targets.
    return sum((counts[split] - desired[split]) ** 2 / desired[split] for split in SPLITS)


def assign_group_splits(
    rows: Sequence[Mapping[str, Any]],
    seed: int = 42,
    targets: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Return included rows with locked group partitions and representatives.

    Within each class, process larger groups first with a seed-derived tie order.
    Choose the partition that best approaches the file targets while reserving
    enough groups to satisfy coverage. Improve by deterministic single-group
    moves. If at least five groups exist, reserve at least two groups in each
    held-out partition and one in training; otherwise reserve one per split.
    """
    proportions = validate_targets(DEFAULT_TARGETS if targets is None else targets)
    classes: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    pixel_labels: dict[str, str] = {}
    seen_ids: set[str] = set()
    for original in rows:
        row = dict(original)
        pixel_hash, label = str(row["pixel_sha256"]), str(row["label"])
        previous = pixel_labels.setdefault(pixel_hash, label)
        if previous != label:
            raise ValueError(f"cross-label pixel conflict: {previous} and {label}")
        if not row.get("included", True):
            continue
        sample_id = str(row["sample_id"])
        if sample_id in seen_ids:
            raise ValueError(f"repeated sample_id: {sample_id}")
        seen_ids.add(sample_id)
        classes[label][pixel_hash].append(row)
    if not classes:
        raise ValueError("no included images remain after exclusions")

    result: list[dict[str, Any]] = []
    for label in sorted(classes):
        groups = classes[label]
        if len(groups) < 3:
            raise ValueError(f"class {label} has only {len(groups)} pixel groups; three partitions require at least 3")
        minimum = {"train": 1, "val": 2, "test": 2} if len(groups) >= 5 else {split: 1 for split in SPLITS}
        order = sorted(groups, key=lambda key: (-len(groups[key]), _tie_key(int(seed), label, key)))
        desired = {split: sum(map(len, groups.values())) * proportions[split] for split in SPLITS}
        files = {split: 0 for split in SPLITS}
        group_counts = {split: 0 for split in SPLITS}
        assignments: dict[str, str] = {}
        for index, key in enumerate(order):
            weight = len(groups[key])
            remaining = len(order) - index - 1
            candidates: list[tuple[float, int, str]] = []
            for split_index, split in enumerate(SPLITS):
                trial_groups = dict(group_counts)
                trial_groups[split] += 1
                deficits = sum(max(0, minimum[name] - trial_groups[name]) for name in SPLITS)
                if deficits > remaining:
                    continue
                trial_files = dict(files)
                trial_files[split] += weight
                candidates.append((_objective(trial_files, desired), split_index, split))
            if not candidates:
                raise ValueError(f"unable to satisfy partition coverage for class {label}")
            split = min(candidates)[2]
            assignments[key] = split
            group_counts[split] += 1
            files[split] += weight

        # This cannot split groups or reduce the minimum held-out coverage.
        for _ in range(20):
            changed = False
            for key in order:
                source, weight = assignments[key], len(groups[key])
                if group_counts[source] <= minimum[source]:
                    continue
                baseline = _objective(files, desired)
                candidates = []
                for split_index, destination in enumerate(SPLITS):
                    if destination == source:
                        continue
                    trial = dict(files)
                    trial[source] -= weight
                    trial[destination] += weight
                    cost = _objective(trial, desired)
                    if cost < baseline - 1e-12:
                        candidates.append((cost, split_index, destination))
                if candidates:
                    destination = min(candidates)[2]
                    assignments[key] = destination
                    group_counts[source] -= 1
                    group_counts[destination] += 1
                    files[source] -= weight
                    files[destination] += weight
                    changed = True
            if not changed:
                break

        for key, members in groups.items():
            representative = min(str(row["relative_path"]) for row in members)
            for row in members:
                row["split"] = assignments[key]
                row["eval_representative"] = str(row["relative_path"]) == representative
                result.append(row)
    result.sort(key=lambda row: str(row["relative_path"]))
    validate_split_manifest(result)
    return result


def summarize_split(rows: Sequence[Mapping[str, Any]], targets: Mapping[str, float]) -> dict[str, Any]:
    """Describe achieved file/group proportions and class coverage explicitly."""
    proportions = validate_targets(targets)
    included = [row for row in rows if row.get("included", True)]
    counts = validate_split_manifest(included)
    classes = sorted({str(row["label"]) for row in included})
    by_class: dict[str, dict[str, Any]] = {}
    for label in classes:
        members = [row for row in included if row["label"] == label]
        class_counts: dict[str, Any] = {}
        minimum_heldout = 2 if len({row["pixel_sha256"] for row in members}) >= 5 else 1
        for split in SPLITS:
            partition = [row for row in members if row["split"] == split]
            groups = len({row["pixel_sha256"] for row in partition})
            if sum(bool(row["eval_representative"]) for row in partition) != groups:
                raise ValueError(f"representative count differs from pixel groups: {label}/{split}")
            class_counts[split] = {
                "files": len(partition), "pixel_groups": groups,
                "file_fraction": len(partition) / len(members),
                "target_fraction": proportions[split],
                "fraction_deviation": len(partition) / len(members) - proportions[split],
            }
        if any(class_counts[split]["pixel_groups"] < minimum_heldout for split in ("val", "test")):
            raise ValueError(f"held-out group coverage is insufficient: {label}")
        by_class[label] = {"minimum_heldout_groups": minimum_heldout, "splits": class_counts}
    return {
        "algorithm": SPLIT_ALGORITHM,
        "file_target": proportions,
        "included_files": len(included),
        "included_pixel_groups": len({row["pixel_sha256"] for row in included}),
        "class_count": len(classes),
        "split_counts": counts,
        "split_pixel_groups": {split: len({row["pixel_sha256"] for row in included if row["split"] == split}) for split in SPLITS},
        "split_file_fractions": {split: counts[split] / len(included) for split in SPLITS},
        "by_class": by_class,
        "cross_partition_pixel_overlap": False,
        "representative_policy": "lexicographically smallest relative_path per exact pixel group in every split",
        "statistical_independence_claimed": False,
    }
