"""One consistent evaluation entry point for the planned 24-class benchmark."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int],
    representative: Sequence[bool] | None = None,
) -> dict[str, object]:
    """Return metrics on the selected unit of evaluation.

    Pass a fixed representative flag from A's manifest to give each exact pixel
    group one vote. Pass None to score every file. The same labels must be used
    across all experiments, including classes with zero support in one slice.
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0 or len(labels) == 0:
        raise ValueError("truth/prediction lengths and class list must be valid")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")
    if representative is not None and len(representative) != len(y_true):
        raise ValueError("representative flags must align with predictions")
    indices = range(len(y_true)) if representative is None else [i for i, flag in enumerate(representative) if flag]
    if not indices:
        raise ValueError("no evaluation samples selected")
    actual = [y_true[i] for i in indices]
    predicted = [y_pred[i] for i in indices]
    if set(actual + predicted) - set(labels):
        raise ValueError("truth or predictions contain labels outside class list")
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=list(labels), zero_division=0
    )
    return {
        "unit": "file" if representative is None else "unique_pixel_representative",
        "n_samples": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=list(labels), average="macro", zero_division=0)),
        "per_class": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
    }
