"""Configured traditional classifiers; fit only on the train partition."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_classifier(kind: str, *, c: float = 1.0):
    if c <= 0:
        raise ValueError("C must be positive")
    if kind == "logistic_regression":
        model = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
    elif kind == "linear_svm":
        model = LinearSVC(C=c, class_weight="balanced", max_iter=5000)
    else:
        raise ValueError(f"unknown classifier: {kind}")
    return make_pipeline(StandardScaler(), model)
