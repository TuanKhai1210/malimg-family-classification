import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from modules.contracts import validate_split_manifest
from modules.feature_io import load_feature_set, save_feature_set


class ContractTests(unittest.TestCase):
    def test_split_rejects_pixel_leakage(self):
        rows = [
            {"sample_id": "a", "relative_path": "a.png", "label": "L", "label_id": 0, "pixel_sha256": "same", "split": "train"},
            {"sample_id": "b", "relative_path": "b.png", "label": "L", "label_id": 0, "pixel_sha256": "same", "split": "val"},
            {"sample_id": "c", "relative_path": "c.png", "label": "L", "label_id": 0, "pixel_sha256": "other", "split": "test"},
        ]
        with self.assertRaisesRegex(ValueError, "crosses partitions"):
            validate_split_manifest(rows)

    def test_features_roundtrip_preserves_row_order(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent / "results") as directory:
            x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            y = np.array([1, 0], dtype=np.int64)
            ids = ["sample-b", "sample-a"]
            meta = {"manifest_sha256": "abc", "extractor": "toy", "weights": "toy", "transform": "none"}
            save_feature_set(Path(directory), "train", x, y, ids, meta)
            restored_x, restored_y, restored_ids, restored_meta = load_feature_set(directory, "train")
            np.testing.assert_array_equal(restored_x, x)
            np.testing.assert_array_equal(restored_y, y)
            self.assertEqual(restored_ids, ids)
            self.assertEqual(restored_meta["manifest_sha256"], "abc")

    @unittest.skipUnless(find_spec("sklearn"), "install requirements.txt to test evaluation")
    def test_representative_metric_ignores_duplicate_votes(self):
        from modules.evaluation import classification_metrics

        all_files = classification_metrics([0, 0, 1], [1, 1, 1], labels=[0, 1])
        representatives = classification_metrics(
            [0, 0, 1], [1, 1, 1], labels=[0, 1], representative=[True, False, True]
        )
        self.assertEqual(all_files["n_samples"], 3)
        self.assertEqual(representatives["n_samples"], 2)
        self.assertNotEqual(all_files["accuracy"], representatives["accuracy"])


if __name__ == "__main__":
    unittest.main()
