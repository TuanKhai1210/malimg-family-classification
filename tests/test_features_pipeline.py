import json
import tempfile
import unittest
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from modules.feature_io import file_sha256, load_feature_set, save_feature_set
from modules.features import ImageSizeExtractor, extract_features


class FeaturePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.rows = []
        for split in ("train", "val", "test"):
            for index in (1, 0):
                relative = f"{split}-{index}.png"
                Image.new("L", (10 + index, 20 + index), 50 + index).save(self.images / relative)
                self.rows.append({"sample_id": relative, "relative_path": relative, "label_id": index, "split": split, "included": True, "eval_representative": True})
        self.config = {"archive_sha256": "archive-verified", "features": {"cache": True}}

    def tearDown(self):
        self.temporary.cleanup()

    def run_extraction(self, config=None, rows=None, manifest="manifest-verified"):
        return extract_features(self.root, config or self.config, rows or self.rows, self.images, manifest, "image_size")

    def test_handcrafted_extraction_preserves_manifest_order_and_provenance(self):
        directory = self.run_extraction()
        x, y, ids, metadata = load_feature_set(directory, "train")
        self.assertEqual(ids, ["train-1.png", "train-0.png"])
        np.testing.assert_array_equal(y, [1, 0])
        np.testing.assert_array_equal(x[:, :2], [[11, 21], [10, 20]])
        self.assertEqual(metadata["dataset_archive_sha256"], "archive-verified")
        self.assertEqual(metadata["manifest_sha256"], "manifest-verified")
        self.assertEqual(metadata["file_sha256"]["X"], file_sha256(directory / "X_train.npy"))
        self.assertTrue((directory / "complete.json").exists())
        self.assertTrue((directory / "X_test.npy").exists())

    def test_matching_cache_reuses_without_extracting(self):
        directory = self.run_extraction()
        with patch.object(ImageSizeExtractor, "extract", side_effect=AssertionError("should use cache")):
            self.assertEqual(self.run_extraction(), directory)

    def test_cache_signature_changes_for_manifest_row_order_and_transform(self):
        directory = self.run_extraction()
        self.assertNotEqual(directory, self.run_extraction(manifest="new-manifest"))
        self.assertNotEqual(directory, self.run_extraction(rows=list(reversed(self.rows))))
        extractor = ImageSizeExtractor()
        extractor.provenance["transform"]["extra"] = "new-transform"
        with patch("modules.features.create_extractor", return_value=extractor):
            self.assertNotEqual(directory, self.run_extraction())

    def test_cache_signature_changes_for_weights_and_library_version(self):
        directory = self.run_extraction()
        for section, key, value in ((None, "weights", "new-weights"), ("software", "numpy", "new-version")):
            extractor = ImageSizeExtractor()
            if section:
                extractor.provenance[section][key] = value
            else:
                extractor.provenance[key] = value
            with patch("modules.features.create_extractor", return_value=extractor):
                self.assertNotEqual(directory, self.run_extraction())

    def test_modified_feature_file_is_rejected_then_rebuilt(self):
        directory = self.run_extraction()
        np.save(directory / "X_train.npy", np.zeros((2, 4), dtype=np.float32), allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_feature_set(directory, "train")
        with self.assertWarnsRegex(RuntimeWarning, "Rebuilding invalid"):
            self.assertEqual(directory, self.run_extraction())
        x, _, _, _ = load_feature_set(directory, "train")
        self.assertEqual(x[0, 0], 11)

    def test_missing_completion_marker_is_never_reused(self):
        directory = self.run_extraction()
        (directory / "complete.json").unlink()
        with self.assertWarnsRegex(RuntimeWarning, "Rebuilding invalid"):
            self.assertEqual(directory, self.run_extraction())
        self.assertTrue((directory / "complete.json").exists())

    def test_metadata_or_ids_change_invalidates_cache(self):
        directory = self.run_extraction()
        (directory / "sample_ids_train.json").write_text(json.dumps(["train-0.png", "train-1.png"]), encoding="utf-8")
        with self.assertWarnsRegex(RuntimeWarning, "Rebuilding invalid"):
            self.run_extraction()
        metadata_path = directory / "metadata_val.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["weights"] = "changed"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertWarnsRegex(RuntimeWarning, "Rebuilding invalid"):
            self.run_extraction()

    def test_interrupted_rebuild_preserves_previous_complete_cache(self):
        directory = self.run_extraction()
        config = deepcopy(self.config)
        config["features"]["cache"] = False
        extractor = ImageSizeExtractor()
        original = extractor.extract
        calls = []

        def fail_on_val(paths):
            calls.append(paths)
            if len(calls) == 2:
                raise RuntimeError("synthetic extraction interrupted")
            return original(paths)

        extractor.extract = fail_on_val
        with patch("modules.features.create_extractor", return_value=extractor):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_extraction(config=config)
        self.assertTrue((directory / "complete.json").exists())
        self.assertEqual(len(list(directory.parent.glob("*.tmp"))), 0)
        self.assertEqual(self.run_extraction(), directory)

    def test_manifest_paths_cannot_escape_data_root(self):
        rows = deepcopy(self.rows)
        rows[0]["relative_path"] = "../outside.png"
        with self.assertRaisesRegex(ValueError, "inside data_root"):
            self.run_extraction(rows=rows)

    def test_source_image_change_is_rejected_before_cache_reuse(self):
        rows = deepcopy(self.rows)
        for row in rows:
            row["file_sha256"] = file_sha256(self.images / row["relative_path"])
        self.run_extraction(rows=rows)
        Image.new("L", (11, 21), 250).save(self.images / rows[0]["relative_path"])
        with self.assertRaisesRegex(ValueError, "image checksum differs"):
            self.run_extraction(rows=rows)

    def test_feature_store_rejects_nonfinite_or_noninteger_labels(self):
        metadata = {"manifest_sha256": "abc", "extractor": "toy", "weights": "none", "transform": "none"}
        with self.assertRaisesRegex(ValueError, "finite"):
            save_feature_set(self.root / "features", "train", np.array([[np.nan]]), np.array([0]), ["a"], metadata)
        with self.assertRaisesRegex(ValueError, "integers"):
            save_feature_set(self.root / "features", "train", np.array([[1]]), np.array([0.5]), ["a"], metadata)

    @unittest.skipUnless(find_spec("skimage"), "optional HOG baseline requires scikit-image")
    def test_hog_baseline_persists_finite_vectors(self):
        config = deepcopy(self.config)
        config["features"]["image_size"] = 32
        config["features"]["resize_policy"] = "pad"
        directory = extract_features(self.root, config, self.rows, self.images, "manifest-verified", "hog")
        x, _, ids, metadata = load_feature_set(directory, "val")
        self.assertEqual(x.shape, (2, 324))
        self.assertEqual(ids, ["val-1.png", "val-0.png"])
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(metadata["weights"], "none")


@unittest.skipUnless(find_spec("torch") and find_spec("torchvision"), "install torch and torchvision for frozen model tests")
class FrozenModelTests(unittest.TestCase):
    def test_actual_torchvision_architectures_have_expected_feature_dimensions(self):
        import torch
        from torchvision import models
        from modules.features import FrozenTorchvisionExtractor

        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "geometry.png"
                Image.new("L", (45, 70), 90).save(path)
                for name, dimension in (("resnet50", 2048), ("convnext_tiny", 768)):
                    builder = getattr(models, name)
                    with patch(f"torchvision.models.{name}", side_effect=lambda weights, original=builder: original(weights=None)):
                        extractor = FrozenTorchvisionExtractor(name, {"device": "cpu", "image_size": 32})
                        x = extractor.extract([str(path)])
                        self.assertEqual(x.shape, (1, dimension))
                        self.assertFalse(extractor._model.training)
                        self.assertTrue(all(not parameter.requires_grad for parameter in extractor._model.parameters()))
                        self.assertTrue(np.isfinite(x).all())
        finally:
            torch.set_num_threads(threads)

    def test_models_remove_only_logits_and_run_frozen_in_manifest_order(self):
        import torch
        from modules.features import FrozenTorchvisionExtractor

        class TinyResNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(1))
                self.fc = torch.nn.Linear(2048, 1000)

            def forward(self, x):
                values = x.mean(dim=(1, 2, 3)).unsqueeze(1).repeat(1, 2048) * self.scale
                return self.fc(values)

        class TinyConvNeXt(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(1))
                self.classifier = torch.nn.Sequential(torch.nn.LayerNorm(768), torch.nn.Flatten(1), torch.nn.Linear(768, 1000))

            def forward(self, x):
                values = x.mean(dim=(1, 2, 3)).unsqueeze(1).repeat(1, 768) * self.scale
                return self.classifier(values)

        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index, value in enumerate((0, 255)):
                path = Path(temporary) / f"{index}.png"
                Image.new("L", (15, 35), value).save(path)
                paths.append(str(path))
            for name, builder, toy, dimension in (
                ("resnet50", "resnet50", TinyResNet, 2048),
                ("convnext_tiny", "convnext_tiny", TinyConvNeXt, 768),
            ):
                with patch(f"torchvision.models.{builder}", side_effect=lambda weights, cls=toy: cls()) as mocked:
                    extractor = FrozenTorchvisionExtractor(name, {"device": "cpu", "image_size": 32, "batch_size": 1, "resize_policy": "pad"})
                    mocked.assert_not_called()  # Cache lookup does not download pretrained weights.
                    x = extractor.extract(paths)
                    self.assertEqual(x.shape, (2, dimension))
                    self.assertFalse(extractor._model.training)
                    self.assertTrue(all(not parameter.requires_grad for parameter in extractor._model.parameters()))
                    self.assertIsInstance(extractor._model.fc if name == "resnet50" else extractor._model.classifier[-1], torch.nn.Identity)
                    if name == "convnext_tiny":
                        self.assertIsInstance(extractor._model.classifier[0], torch.nn.LayerNorm)
                    else:
                        self.assertLess(x[0, 0], x[1, 0])
                    self.assertEqual(extractor.transform(Image.new("L", (15, 35))).shape, (3, 32, 32))
                    self.assertTrue(np.isfinite(x).all())

    def test_weight_normalization_and_default_alias_rejection(self):
        from modules.features import FrozenTorchvisionExtractor

        extractor = FrozenTorchvisionExtractor("resnet50", {"device": "cpu"})
        self.assertEqual(extractor.provenance["transform"]["mean"], [0.485, 0.456, 0.406])
        self.assertEqual(extractor.provenance["transform"]["std"], [0.229, 0.224, 0.225])
        with self.assertRaisesRegex(ValueError, "explicit"):
            FrozenTorchvisionExtractor("resnet50", {"weights": {"resnet50": "DEFAULT"}})


if __name__ == "__main__":
    unittest.main()
