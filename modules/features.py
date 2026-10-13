"""Frozen visual features and optional handcrafted baselines.

No model is fitted here. Test images may be encoded by a frozen representation,
but model selection and test scoring belong to the separate experiment stage.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageOps

from .contracts import SPLITS, validate_sample_ids
from .feature_io import atomic_json, file_sha256, load_feature_set, save_feature_set


class FeatureExtractor(Protocol):
    name: str
    weights_id: str
    provenance: dict[str, Any]

    def extract(self, image_paths: list[str]) -> np.ndarray:
        """Return finite [n_images, n_features] vectors in exact input order."""


def _versions(*packages: str) -> dict[str, str]:
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _ImageTransform:
    """Picklable preprocessing for DataLoader workers on Windows and Linux."""

    def __init__(self, size: int, policy: str, mean: list[float], std: list[float], interpolation: Any):
        from torchvision import transforms

        self.size, self.policy, self.interpolation = size, policy, interpolation
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image: Image.Image):
        from torchvision.transforms import functional as tf

        image = image.convert("RGB")
        if self.policy == "direct":
            image = tf.resize(image, [self.size, self.size], interpolation=self.interpolation, antialias=True)
        else:
            width, height = image.size
            scale = self.size / max(width, height)
            resized = [max(1, round(height * scale)), max(1, round(width * scale))]
            image = tf.resize(image, resized, interpolation=self.interpolation, antialias=True)
            canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
            canvas.paste(image, ((self.size - image.width) // 2, (self.size - image.height) // 2))
            image = canvas
        return self.normalize(self.to_tensor(image))


class _ImageDataset:
    def __init__(self, paths: list[str], transform: _ImageTransform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.transform(image)


class FrozenTorchvisionExtractor:
    """Lazy torchvision backbone; initialization itself never downloads weights."""

    def __init__(self, name: str, settings: dict[str, Any]):
        try:
            import torch
            from torchvision import models
        except (ImportError, RuntimeError) as error:
            raise RuntimeError("Frozen features require compatible torch and torchvision; install requirements.txt") from error
        self.name = name
        size = int(settings.get("image_size", 224))
        policy = settings.get("resize_policy", "direct")
        if size < 32 or policy not in {"direct", "pad"}:
            raise ValueError("features.image_size must be >= 32 and resize_policy must be direct or pad")
        self.batch_size = int(settings.get("batch_size", 32))
        self.num_workers = int(settings.get("num_workers", 0))
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("features.batch_size must be positive and num_workers nonnegative")
        defaults = {"resnet50": "IMAGENET1K_V2", "convnext_tiny": "IMAGENET1K_V1"}
        configured = settings.get("weights", {})
        if not isinstance(configured, dict):
            raise ValueError("features.weights must map extractor names to explicit weight IDs")
        weights_name = configured.get(name, defaults[name])
        if weights_name == "DEFAULT":
            raise ValueError("Use an explicit torchvision weight ID; DEFAULT can change between releases")
        enum = models.ResNet50_Weights if name == "resnet50" else models.ConvNeXt_Tiny_Weights
        try:
            self.weights = enum[weights_name]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Unsupported weights for {name}: {weights_name}") from error
        preset = self.weights.transforms()
        self.weights_id = f"{enum.__name__}.{self.weights.name}"
        requested_device = str(settings.get("device", "auto"))
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("features.device requests CUDA but torch cannot access a CUDA device")
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("features.device requests MPS but torch cannot access an MPS device")
        self.transform = _ImageTransform(size, policy, list(preset.mean), list(preset.std), preset.interpolation)
        self.dimension = 2048 if name == "resnet50" else 768
        self.provenance = {
            "extractor": name,
            "weights": self.weights_id,
            "weights_url": self.weights.url,
            "frozen": True,
            "feature_dimension": self.dimension,
            "feature_layer": "global-average-pool + flatten" if name == "resnet50" else "global-average-pool + classifier layernorm + flatten",
            "transform": {
                "image_size": [size, size],
                "resize_policy": policy,
                "grayscale_to_rgb": "repeat grayscale into three RGB channels",
                "interpolation": preset.interpolation.value,
                "antialias": True,
                "padding": {"color_rgb": [0, 0, 0], "position": "center", "rounding": "Python round, minimum 1 pixel"} if policy == "pad" else None,
                "mean": list(preset.mean),
                "std": list(preset.std),
                "weights_reference_transform": {
                    "resize_size": list(preset.resize_size),
                    "crop_size": list(preset.crop_size),
                    "note": "Project resize policy above replaces the weights preset resize/crop; its normalization and interpolation are retained.",
                },
            },
            "software": _versions("numpy", "Pillow", "torch", "torchvision"),
            "execution": {
                "device": str(self.device),
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else self.device.type,
            },
        }
        self._model = None

    def _load_model(self):
        import torch
        from torchvision import models

        if self._model is None:
            if self.name == "resnet50":
                model = models.resnet50(weights=self.weights)
                model.fc = torch.nn.Identity()
            else:
                model = models.convnext_tiny(weights=self.weights)
                # Keep classifier LayerNorm and Flatten; remove only ImageNet logits.
                model.classifier[-1] = torch.nn.Identity()
            model.requires_grad_(False)
            model.eval()
            self._model = model.to(self.device)
        return self._model

    def extract(self, image_paths: list[str]) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader

        model = self._load_model()
        model.eval()
        loader = DataLoader(
            _ImageDataset(image_paths, self.transform),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        chunks = []
        with torch.inference_mode():
            for inputs in loader:
                vectors = model(inputs.to(self.device, non_blocking=self.device.type == "cuda"))
                chunks.append(vectors.detach().cpu().numpy().astype(np.float32, copy=False))
        result = np.concatenate(chunks, axis=0) if chunks else np.empty((0, self.dimension), dtype=np.float32)
        if result.shape != (len(image_paths), self.dimension) or not np.isfinite(result).all():
            raise ValueError(f"{self.name} returned an invalid feature matrix: {result.shape}")
        return result


class ImageSizeExtractor:
    """E0 audit features measured before resizing: width, height, ratio, pixels."""

    name, weights_id = "image_size", "none"

    def __init__(self):
        self.provenance = {
            "extractor": self.name,
            "weights": self.weights_id,
            "frozen": True,
            "feature_dimension": 4,
            "feature_names": ["width", "height", "aspect_ratio", "pixel_count"],
            "transform": {"resize": False, "source": "original decoded image dimensions"},
            "software": _versions("numpy", "Pillow"),
        }

    def extract(self, image_paths: list[str]) -> np.ndarray:
        result = []
        for path in image_paths:
            with Image.open(path) as image:
                width, height = image.size
                result.append([width, height, width / height, width * height])
        return np.asarray(result, dtype=np.float32).reshape(-1, 4)


class HOGExtractor:
    """Optional E1 grayscale HOG baseline, with explicit resize provenance."""

    name, weights_id = "hog", "none"

    def __init__(self, settings: dict[str, Any]):
        try:
            from skimage.feature import hog
        except ImportError as error:
            raise RuntimeError("The optional HOG baseline requires scikit-image") from error
        self._hog = hog
        self.size = int(settings.get("image_size", 224))
        self.policy = settings.get("resize_policy", "direct")
        if self.size < 32 or self.policy not in {"direct", "pad"}:
            raise ValueError("HOG image_size must be >= 32 and resize_policy must be direct or pad")
        options = settings.get("hog", {})
        self.options = {
            "orientations": int(options.get("orientations", 9)),
            "pixels_per_cell": tuple(options.get("pixels_per_cell", [8, 8])),
            "cells_per_block": tuple(options.get("cells_per_block", [2, 2])),
            "block_norm": "L2-Hys",
            "transform_sqrt": bool(options.get("transform_sqrt", True)),
            "feature_vector": True,
        }
        self.provenance = {
            "extractor": self.name,
            "weights": self.weights_id,
            "frozen": True,
            "transform": {"mode": "L", "image_size": [self.size, self.size], "resize_policy": self.policy, "interpolation": "bilinear", "padding_value": 0, "scale": "pixel / 255.0", "hog": self.options},
            "software": _versions("numpy", "Pillow", "scikit-image"),
        }

    def extract(self, image_paths: list[str]) -> np.ndarray:
        result = []
        for path in image_paths:
            with Image.open(path) as source:
                image = source.convert("L")
                if self.policy == "direct":
                    image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
                else:
                    image = ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.BILINEAR, color=0, centering=(0.5, 0.5))
                result.append(self._hog(np.asarray(image, dtype=np.float32) / 255.0, **self.options))
        return np.asarray(result, dtype=np.float32)


def create_extractor(name: str, settings: dict[str, Any]) -> FeatureExtractor:
    """Construct an extractor; pretrained weights download on first extraction."""
    if name in {"resnet50", "convnext_tiny"}:
        return FrozenTorchvisionExtractor(name, settings)
    if name == "image_size":
        return ImageSizeExtractor()
    if name == "hog":
        return HOGExtractor(settings)
    raise ValueError(f"Unsupported extractor: {name}")


def _split_rows(rows: list[dict[str, Any]], data_root: Path) -> dict[str, list[dict[str, Any]]]:
    selected = {split: [] for split in SPLITS}
    all_ids = []
    for row in rows:
        if row.get("included", True) in (False, "False", "false", "0", 0):
            continue
        split = row.get("split")
        if split not in selected:
            raise ValueError(f"Included sample has invalid split: {row.get('sample_id')}")
        label = row.get("label_id")
        if isinstance(label, bool) or not isinstance(label, (int, np.integer)) or label < 0:
            raise ValueError("Feature extraction requires nonnegative integer label_id values")
        relative = Path(str(row.get("relative_path", "")))
        path = (data_root / relative).resolve()
        if not str(row.get("relative_path", "")) or relative.is_absolute() or not path.is_relative_to(data_root.resolve()):
            raise ValueError("Manifest image paths must remain inside data_root")
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {relative}")
        expected_hash = row.get("file_sha256")
        if expected_hash and file_sha256(path) != expected_hash:
            raise ValueError(f"Manifest image checksum differs: {relative}; regenerate and review the data manifest")
        selected[split].append(row)
        all_ids.append(row.get("sample_id"))
    if any(not isinstance(value, str) for value in all_ids):
        raise ValueError("Feature sample IDs must be strings")
    validate_sample_ids(all_ids)
    if any(not selected[split] for split in SPLITS):
        raise ValueError("Feature extraction requires nonempty train, val and test splits")
    return selected


def _cache_valid(directory: Path, signature: dict[str, Any], selected: dict[str, list[dict[str, Any]]]) -> bool:
    complete = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    if not isinstance(complete, dict):
        raise ValueError("Feature cache completion marker must be an object")
    signature_hash = _canonical_hash(signature)
    if complete.get("extraction_signature_sha256") != signature_hash or complete.get("extraction_signature") != signature:
        raise ValueError("Feature cache extraction signature differs")
    if complete.get("metadata_sha256") != file_sha256(directory / "metadata.json"):
        raise ValueError("Feature cache common metadata checksum differs")
    common = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(common, dict):
        raise ValueError("Feature cache common metadata must be an object")
    if common.get("extraction_signature") != signature or common.get("extraction_signature_sha256") != signature_hash:
        raise ValueError("Feature cache common metadata differs")
    expected_common = dict(
        signature["extractor_provenance"],
        manifest_sha256=signature["manifest_sha256"],
        dataset_archive_sha256=signature["dataset_archive_sha256"],
    )
    if any(common.get(key) != value for key, value in expected_common.items()):
        raise ValueError("Feature cache common provenance contradicts its signature")
    split_metadata_hashes = complete.get("split_metadata_sha256")
    if not isinstance(split_metadata_hashes, dict) or set(split_metadata_hashes) != set(SPLITS):
        raise ValueError("Feature cache split metadata checksums are missing")
    dimensions = set()
    for split in SPLITS:
        if split_metadata_hashes[split] != file_sha256(directory / f"metadata_{split}.json"):
            raise ValueError(f"Feature cache metadata checksum differs for {split}")
        x, y, ids, metadata = load_feature_set(directory, split)
        expected = selected[split]
        if ids != [row["sample_id"] for row in expected] or not np.array_equal(y, [row["label_id"] for row in expected]):
            raise ValueError(f"Feature cache sample ordering or labels differ for {split}")
        if metadata.get("extraction_signature") != signature or metadata.get("extraction_signature_sha256") != signature_hash:
            raise ValueError(f"Feature cache provenance differs for {split}")
        if any(metadata.get(key) != value for key, value in expected_common.items()):
            raise ValueError(f"Feature cache split provenance contradicts its signature for {split}")
        dimensions.add(x.shape[1])
    if len(dimensions) != 1:
        raise ValueError("Feature dimension differs between splits")
    return True


def _remove_generated_directory(path: Path, parent: Path) -> None:
    if path.resolve().parent != parent.resolve():
        raise ValueError("Refusing to remove a directory outside the feature cache parent")
    shutil.rmtree(path)


def extract_features(
    root: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    data_root: Path,
    manifest_sha256: str,
    extractor_name: str,
) -> Path:
    """Extract/save all partitions, or reuse a fully verified matching cache.

    Cache policy: an incomplete or inconsistent cache is never used. It is
    rebuilt in a staging directory, and replaces the old directory only after
    every split has passed read-back validation. A changed extraction signature
    creates a separate directory. No classifier fitting or scoring occurs here.
    """
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    selected = _split_rows(rows, data_root)
    settings = config.get("features", {})
    extractor = create_extractor(extractor_name, settings)
    provenance = json.loads(json.dumps(extractor.provenance, allow_nan=False))
    row_signature = {
        split: [
            {key: row.get(key) for key in ("sample_id", "relative_path", "label_id", "eval_representative", "pixel_sha256", "file_sha256")}
            for row in selected[split]
        ]
        for split in SPLITS
    }
    archive_hash = config.get("archive_sha256", config.get("dataset", {}).get("archive_sha256"))
    signature = {
        "schema_version": 2,
        "implementation_sha256": file_sha256(Path(__file__)),
        "manifest_sha256": manifest_sha256,
        "dataset_archive_sha256": archive_hash,
        "rows_sha256": _canonical_hash(row_signature),
        "extractor_provenance": provenance,
    }
    signature_hash = _canonical_hash(signature)
    feature_root = Path(config.get("features_dir", "features"))
    if not feature_root.is_absolute():
        feature_root = root / feature_root
    parent = feature_root / extractor_name
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / signature_hash[:16]
    if settings.get("cache", True) and directory.exists():
        try:
            if _cache_valid(directory, signature, selected):
                print(f"Feature cache verified: {extractor_name} ({signature_hash[:16]})")
                return directory
        except (OSError, ValueError, TypeError, KeyError, EOFError) as error:
            warnings.warn(f"Rebuilding invalid {extractor_name} feature cache: {error}", RuntimeWarning)
    staging = Path(tempfile.mkdtemp(prefix=f".{signature_hash[:16]}.", suffix=".tmp", dir=parent))
    backup = None
    started = time.perf_counter()
    common = dict(
        provenance,
        manifest_sha256=manifest_sha256,
        dataset_archive_sha256=archive_hash,
        dataset_name=config.get("dataset_name", config.get("dataset", {}).get("name", "MalImg")),
        extraction_signature=signature,
        extraction_signature_sha256=signature_hash,
    )
    try:
        for split in SPLITS:
            entries = selected[split]
            paths = [str(data_root / row["relative_path"]) for row in entries]
            split_started = time.perf_counter()
            x = np.asarray(extractor.extract(paths), dtype=np.float32)
            y = np.asarray([row["label_id"] for row in entries], dtype=np.int64)
            ids = [row["sample_id"] for row in entries]
            save_feature_set(staging, split, x, y, ids, dict(common, extraction_seconds=time.perf_counter() - split_started))
            print(f"Extracted {extractor_name}/{split}: {x.shape[0]} samples x {x.shape[1]} features")
        atomic_json(staging / "metadata.json", dict(common, total_extraction_seconds=time.perf_counter() - started))
        complete = {
            "extraction_signature": signature,
            "extraction_signature_sha256": signature_hash,
            "metadata_sha256": file_sha256(staging / "metadata.json"),
            "split_metadata_sha256": {split: file_sha256(staging / f"metadata_{split}.json") for split in SPLITS},
        }
        atomic_json(staging / "complete.json", complete)
        _cache_valid(staging, signature, selected)
        if directory.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{signature_hash[:16]}.stale.", dir=parent))
            backup.rmdir()
            os.replace(directory, backup)
        try:
            os.replace(staging, directory)
        except OSError:
            if backup is not None:
                os.replace(backup, directory)
                backup = None
            raise
        if backup is not None:
            _remove_generated_directory(backup, parent)
        return directory
    finally:
        if staging.exists():
            _remove_generated_directory(staging, parent)
