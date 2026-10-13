"""Configured traditional classifiers; fit only on the train partition."""

from __future__ import annotations

import math

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_classifier(
    kind: str,
    *,
    c: float = 1.0,
    seed: int = 42,
    max_iter: int = 5000,
    class_weight: str | dict | None = "balanced",
    pca_components: int | float | None = None,
):
    """Build a train-only pipeline; callers must never fit it on validation/test.

    PCA, when enabled, is learned after scaling in this same fitted pipeline.
    Random state is explicit even for solvers whose current path is deterministic.
    """
    if isinstance(c, bool) or not math.isfinite(c) or c <= 0:
        raise ValueError("C must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if pca_components is not None:
        if isinstance(pca_components, bool) or pca_components <= 0:
            raise ValueError("pca_components must be a positive integer or fraction")
        if isinstance(pca_components, float) and pca_components >= 1:
            raise ValueError("fractional pca_components must be less than one")
    if kind == "logistic_regression":
        model = LogisticRegression(
            C=c, max_iter=max_iter, class_weight=class_weight, random_state=seed
        )
    elif kind == "linear_svm":
        model = LinearSVC(
            C=c, class_weight=class_weight, max_iter=max_iter,
            random_state=seed, dual="auto",
        )
    else:
        raise ValueError(f"unknown classifier: {kind}")
    steps = [StandardScaler()]
    if pca_components is not None:
        steps.append(PCA(n_components=pca_components, random_state=seed))
    return make_pipeline(*steps, model)
