"""Synthetic integration checks: no MalImg download, training, or model weights."""

import json
import hashlib
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np

from modules.feature_io import save_feature_set


@unittest.skipUnless(find_spec("sklearn"), "install requirements.txt to test experiments")
class ExperimentTests(unittest.TestCase):
    def setUp(self):
        from modules.experiments import run_validation, run_final_test
        self.run_validation = run_validation
        self.run_final_test = run_final_test
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.feature_dir = self.root / "features" / "resnet50"
        self.manifest_sha = "1" * 64
        self.config = {
            "archive_sha256": "2" * 64,
            "results_dir": "results", "data_dir": "data/images",
            "training": {"seed": 42, "max_iter": 1000, "class_weight": "balanced",
                         "train_unit": "file", "pca_components": None},
            "experiments": [{"id": "E2", "extractor": "resnet50",
                             "classifier": "logistic_regression", "c_values": [1.0, 0.1]}],
            "evaluation": {"figures": False},
        }
        self.arrays = {
            "train": (np.array([[-4, 0], [-2, 2], [2, 0], [4, 2]], np.float32),
                      np.array([0, 0, 1, 1], np.int64)),
            "val": (np.array([[-3, 1], [-3, 1], [3, 1]], np.float32),
                    np.array([0, 0, 1], np.int64)),
            "test": (np.array([[-5, 2], [5, 2]], np.float32), np.array([0, 1], np.int64)),
        }
        self.rows = []
        for split, (_, y) in self.arrays.items():
            for index, label in enumerate(y):
                is_duplicate = split == "val" and index == 1
                group_index = 0 if is_duplicate else index
                self.rows.append({
                    "sample_id": f"{split}-{index}", "relative_path": f"family-{label}/{split}-{index}.png",
                    "label": f"family-{label}", "label_id": int(label),
                    "pixel_sha256": f"pixel-{split}-{group_index}", "split": split,
                    "included": True, "eval_representative": not is_duplicate,
                })
        self.metadata = {"manifest_sha256": self.manifest_sha, "extractor": "resnet50",
                         "weights": "synthetic-local-features", "transform": {"mode": "synthetic"},
                         "dataset_archive_sha256": self.config["archive_sha256"]}
        for split, (x, y) in self.arrays.items():
            save_feature_set(self.feature_dir, split, x, y,
                             [row["sample_id"] for row in self.rows if row["split"] == split], self.metadata)

    def validate(self):
        return self.run_validation(self.root, self.config, self.rows, self.manifest_sha,
                                   {"resnet50": self.feature_dir})

    def test_validation_never_opens_test_arrays_and_fits_scaler_only_on_train(self):
        import joblib
        from modules.feature_io import load_feature_set
        accessed = []

        def guarded_loader(directory, split):
            accessed.append(split)
            self.assertNotEqual(split, "test", "validation must not deserialize test features")
            return load_feature_set(directory, split)

        with patch("modules.experiments.load_feature_set", side_effect=guarded_loader):
            result = self.validate()
        self.assertEqual(accessed, ["train", "val"])
        selection = json.loads(Path(result["selection_path"]).read_text(encoding="utf-8"))
        best = selection["experiments"][0]
        model = joblib.load(self.root / best["model_path"])
        np.testing.assert_allclose(model.named_steps["standardscaler"].mean_,
                                   self.arrays["train"][0].mean(axis=0))
        self.assertEqual(best["selected_c"], 0.1, "ties must choose smaller C")
        self.assertEqual(best["validation_metrics"]["primary"]["n_samples"], 2)
        self.assertEqual(best["validation_metrics"]["file_secondary"]["n_samples"], 3)
        trials = json.loads((self.root / best["trials_path"]).read_text(encoding="utf-8"))
        self.assertEqual(len(trials["trials"]), 2)
        self.assertTrue(all(trial["status"] == "completed" for trial in trials["trials"]))
        self.assertTrue(all("warnings" in trial for trial in trials["trials"]))
        comparison = (self.root / "results" / "comparison.csv").read_text(encoding="utf-8")
        self.assertIn("test_macro_f1_representative", comparison)
        self.assertFalse((self.root / "results" / "final_test.lock.json").exists())

    def test_validation_rejects_changed_sample_id_even_when_files_are_consistent(self):
        x, y = self.arrays["val"]
        save_feature_set(self.feature_dir, "val", x, y, ["wrong-id", "val-1", "val-2"], self.metadata)
        with self.assertRaisesRegex(ValueError, "sample IDs/order"):
            self.validate()
        registry = json.loads((self.root / "results" / "registry.json").read_text(encoding="utf-8"))
        self.assertEqual(registry["runs"][-1]["status"], "failed")

    def test_final_test_uses_locked_models_without_fit_and_reuses_completed_result(self):
        result = self.validate()
        with patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("final test must never fit")):
            final = self.run_final_test(self.root, self.config, self.rows, self.manifest_sha,
                                       Path(result["selection_path"]))
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["experiments"][0]["metrics"]["primary"]["macro_f1"], 1.0)
        lock = json.loads((self.root / "results" / "final_test.lock.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["status"], "completed")
        artifact = self.root / "results" / "final_test" / "E2"
        for name in ("metrics.json", "predictions.csv", "per_class.csv", "confusion_matrix.csv",
                     "confusion_matrix_files.csv", "errors.csv", "error_analysis.json"):
            self.assertTrue((artifact / name).exists(), name)
        with patch("modules.experiments.load_feature_set", side_effect=AssertionError("no second test scoring")):
            reused = self.run_final_test(self.root, self.config, self.rows, self.manifest_sha,
                                        Path(result["selection_path"]))
        self.assertTrue(reused["reused"])
        with self.assertRaisesRegex(ValueError, "final test is locked"):
            self.validate()

    def test_final_test_rejects_changed_config_and_feature_files_before_lock(self):
        result = self.validate()
        changed = {**self.config, "training": {**self.config["training"], "seed": 43}}
        with self.assertRaisesRegex(ValueError, "config changed"):
            self.run_final_test(self.root, changed, self.rows, self.manifest_sha, Path(result["selection_path"]))
        self.assertFalse((self.root / "results" / "final_test.lock.json").exists())
        # Updating both array and metadata is still a new artifact and invalidates selection.
        x, y = self.arrays["test"]
        save_feature_set(self.feature_dir, "test", x + 1, y, ["test-0", "test-1"], self.metadata)
        with self.assertRaisesRegex(ValueError, "feature files changed"):
            self.run_final_test(self.root, self.config, self.rows, self.manifest_sha, Path(result["selection_path"]))
        self.assertFalse((self.root / "results" / "final_test.lock.json").exists())

    def test_final_test_rejects_mutated_manifest_and_model(self):
        result = self.validate()
        altered = [dict(row) for row in self.rows]
        altered[0]["relative_path"] = "changed.png"
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.run_final_test(self.root, self.config, altered, self.manifest_sha, Path(result["selection_path"]))
        selected = result["experiments"][0]
        with (self.root / selected["model_path"]).open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "selected model"):
            self.run_final_test(self.root, self.config, self.rows, self.manifest_sha, Path(result["selection_path"]))
        self.assertFalse((self.root / "results" / "final_test.lock.json").exists())

    def test_optional_pca_is_learned_inside_train_pipeline(self):
        import joblib
        self.config["training"]["pca_components"] = 1
        result = self.validate()
        selected = result["experiments"][0]
        model = joblib.load(self.root / selected["model_path"])
        self.assertEqual(model.named_steps["pca"].n_components_, 1)
        np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, [0, 1])

    def test_two_extractors_and_linear_svm_create_comparable_figures(self):
        second = self.root / "features" / "convnext_tiny"
        for split, (x, y) in self.arrays.items():
            save_feature_set(second, split, x, y,
                             [row["sample_id"] for row in self.rows if row["split"] == split],
                             {**self.metadata, "extractor": "convnext_tiny"})
        self.config["experiments"] = [
            {"id": "E3", "extractor": "resnet50", "classifier": "linear_svm", "c_values": [1]},
            {"id": "E4", "extractor": "convnext_tiny", "classifier": "linear_svm", "c_values": [1]},
        ]
        self.config["evaluation"]["figures"] = True
        result = self.run_validation(self.root, self.config, self.rows, self.manifest_sha,
                                    {"resnet50": self.feature_dir, "convnext_tiny": second})
        final = self.run_final_test(self.root, self.config, self.rows, self.manifest_sha,
                                   Path(result["selection_path"]))
        self.assertEqual(len(final["experiments"]), 2)
        for experiment in final["experiments"]:
            self.assertEqual(experiment["metrics"]["primary"]["macro_f1"], 1.0)
            figures = self.root / experiment["output_dir"] / "figures"
            self.assertTrue((figures / "confusion_matrix.png").is_file())
            self.assertTrue((figures / "per_class_f1.png").is_file())

    def test_real_convergence_warnings_and_failed_trials_remain_visible(self):
        from modules.classifiers import build_classifier
        self.config["training"]["max_iter"] = 1

        def one_failed_candidate(kind, *, c, **kwargs):
            if c == 0.1:
                raise ValueError("synthetic failed candidate")
            return build_classifier(kind, c=c, **kwargs)

        with patch("modules.experiments.build_classifier", side_effect=one_failed_candidate):
            result = self.validate()
        best = result["experiments"][0]
        trials = json.loads((self.root / best["trials_path"]).read_text(encoding="utf-8"))["trials"]
        self.assertEqual(trials[0]["status"], "failed")
        self.assertIn("synthetic failed candidate", trials[0]["error"])
        self.assertFalse(trials[1]["converged"])
        self.assertTrue(any(warning["category"] == "ConvergenceWarning" for warning in trials[1]["warnings"]))

    def test_summary_rejects_tampered_final_metrics_and_missing_completion_lock(self):
        from modules.experiments import summarize_results
        result = self.validate()
        final = self.run_final_test(self.root, self.config, self.rows, self.manifest_sha,
                                   Path(result["selection_path"]))
        path = Path(final["results_path"])
        original = path.read_text(encoding="utf-8")
        altered = json.loads(original)
        altered["experiments"][0]["metrics"]["primary"]["macro_f1"] = 0.123
        path.write_text(json.dumps(altered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "completed lock"):
            summarize_results(self.root / "results")
        path.write_text(original, encoding="utf-8")
        (self.root / "results" / "final_test.lock.json").unlink()
        with self.assertRaisesRegex(ValueError, "no completion lock"):
            summarize_results(self.root / "results")

    def test_orphan_final_outputs_block_validation_before_any_fit(self):
        (self.root / "results" / "final_test").mkdir(parents=True)
        with patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("must fail before fitting")):
            with self.assertRaisesRegex(ValueError, "final test is locked"):
                self.validate()

    def test_signed_feature_store_rejects_current_config_resize_and_weights_changes(self):
        from modules.feature_io import atomic_json, file_sha256

        def digest(value):
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                 allow_nan=False).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        declared = {"extractor": "resnet50", "weights": "ResNet50_Weights.IMAGENET1K_V2",
                    "transform": {"image_size": [224, 224], "resize_policy": "direct"},
                    "software": {"source": "synthetic fixture"}}
        row_signature = {split: [{key: row.get(key) for key in (
            "sample_id", "relative_path", "label_id", "eval_representative", "pixel_sha256", "file_sha256")}
            for row in self.rows if row["split"] == split] for split in self.arrays}
        signature = {"schema_version": 2, "implementation_sha256": "synthetic code",
                     "manifest_sha256": self.manifest_sha,
                     "dataset_archive_sha256": self.config["archive_sha256"],
                     "rows_sha256": digest(row_signature), "extractor_provenance": declared}
        metadata = {**declared, "manifest_sha256": self.manifest_sha,
                    "dataset_archive_sha256": self.config["archive_sha256"],
                    "extraction_signature": signature, "extraction_signature_sha256": digest(signature)}
        for split, (x, y) in self.arrays.items():
            save_feature_set(self.feature_dir, split, x, y,
                             [row["sample_id"] for row in self.rows if row["split"] == split], metadata)
        atomic_json(self.feature_dir / "metadata.json", metadata)
        atomic_json(self.feature_dir / "complete.json", {
            "extraction_signature": signature, "extraction_signature_sha256": digest(signature),
            "metadata_sha256": file_sha256(self.feature_dir / "metadata.json"),
            "split_metadata_sha256": {split: file_sha256(self.feature_dir / f"metadata_{split}.json")
                                      for split in self.arrays},
        })
        self.config["features"] = {"image_size": 224, "resize_policy": "direct",
                                   "weights": {"resnet50": "IMAGENET1K_V2"}}
        self.validate()  # The correctly signed fixture works without any pretrained download.
        for field, value in (("image_size", 256), ("resize_policy", "pad"),
                             ("weights", {"resnet50": "IMAGENET1K_V1"})):
            original = self.config["features"][field]
            self.config["features"][field] = value
            with self.assertRaisesRegex(ValueError, f"feature {field} differ"):
                self.validate()
            self.config["features"][field] = original


if __name__ == "__main__":
    unittest.main()
