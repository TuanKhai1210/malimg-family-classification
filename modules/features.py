"""Feature extraction interface owned by B.

Implement ResNet50 and ConvNeXt-Tiny adapters here. Each adapter should return
one finite vector per sample, record weights and transforms, run in eval mode,
and avoid updating pretrained weights for the frozen core experiments.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class FeatureExtractor(Protocol):
    name: str
    weights_id: str

    def extract(self, image_paths: list[str]) -> np.ndarray:
        """Return [n_images, n_features] in the exact input order."""
