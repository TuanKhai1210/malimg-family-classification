"""Resume equivalence and leakage checks on tiny synthetic feature arrays."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from modules.epoch_training import (DEFAULT_SETTINGS, _StableSGDClassifier, _stable_log_probabilities,
                                    run_epoch_final_test, run_epoch_training)
from modules.feature_io import save_feature_set


class EpochTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "workspace"
        self.root.mkdir()
        self.directory = self.root / "features/resnet50/synthetic"
        self.config = {
            "archive_sha256": "a" * 64, "results_dir": "results", "data_dir": "data/images",
            "training": {"seed": 42, "class_weight": "balanced", "train_unit": "file", "pca_components": None},
            "evaluation": {"figures": False},
        }
        (self.root / "configs").mkdir()
        (self.root / "configs/project.json").write_text(json.dumps(self.config), encoding="utf-8")
        self.settings = dict(DEFAULT_SETTINGS, epochs=6, patience=99, batch_size=4)
        self.rows = []
        rng = np.random.default_rng(4)
        self.arrays = {}
        for split, count in (("train", 18), ("val", 9), ("test", 6)):
            y = np.arange(count) % 3
            x = np.eye(3)[y] * 3 + rng.normal(0, .2, (count, 3))
            self.arrays[split] = x, y
            ids = []
            for i, label in enumerate(y):
                sid = f"{split}-{i}"
                ids.append(sid)
                self.rows.append({"sample_id": sid, "relative_path": f"family-{label}/{sid}.png",
                    "label": f"family-{label}", "label_id": int(label), "pixel_sha256": sid,
                    "split": split, "included": True, "eval_representative": True})
            save_feature_set(self.directory, split, x.astype(np.float32), y.astype(np.int64), ids,
                {"manifest_sha256": "b" * 64, "extractor": "resnet50", "weights": "synthetic",
                 "transform": {"mode": "synthetic"}, "dataset_archive_sha256": "a" * 64})

    def train(self, **kwargs):
        return run_epoch_training(self.root, self.config, self.rows, "b" * 64,
                                  {"resnet50": self.directory}, self.settings, **kwargs)

    def state(self, result):
        directory = Path(result["run_dir"]) / "epochs" / f"epoch_{result['epoch']:06d}"
        return joblib.load(directory / "state.joblib")

    def assert_same_optimizer(self, left, right):
        for attribute in ("coef_", "intercept_", "t_", "classes_"):
            np.testing.assert_array_equal(getattr(left["model"][-1], attribute), getattr(right["model"][-1], attribute))
        self.assertEqual(left["rng_state"], right["rng_state"])
        self.assertEqual(left["best_epoch"], right["best_epoch"])
        self.assertEqual(left["stale_epochs"], right["stale_epochs"])
        for key in ("train_loss_unweighted", "val_macro_f1", "val_accuracy"):
            self.assertEqual([r[key] for r in left["history"]], [r[key] for r in right["history"]])

    def test_segmented_resume_matches_uninterrupted_optimizer_and_does_not_open_test(self):
        from modules import experiments as ex
        loader = ex._load_aligned
        def guarded(*args, **kwargs):
            self.assertNotEqual(args[1], "test")
            return loader(*args, **kwargs)
        with patch.object(ex, "_load_aligned", side_effect=guarded):
            full = self.train()
            part = self.train(epochs_this_session=2)
            self.assertEqual(part["status"], "paused")
            resumed = self.train(resume_run=part["run_id"])
        self.assert_same_optimizer(self.state(full), self.state(resumed))
        np.testing.assert_allclose(self.state(resumed)["model"][0].mean_, self.arrays["train"][0].astype(np.float32).mean(axis=0), atol=1e-6)
        with patch.object(SGDClassifier, "partial_fit", side_effect=AssertionError("completed run must not fit")):
            again = self.train(resume_run=part["run_id"])
        self.assertEqual(again["epoch"], 6)

    def test_mid_epoch_interrupt_replays_only_uncommitted_epoch(self):
        full = self.train()
        calls = 0
        original = SGDClassifier.partial_fit
        def interrupt(model, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 7:  # five batches in epoch one, then interrupt epoch two
                raise KeyboardInterrupt()
            return original(model, *args, **kwargs)
        with patch.object(SGDClassifier, "partial_fit", new=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.train()
        runs = list((self.root / "results/epoch_runs").iterdir())
        interrupted = next(p for p in runs if p.name != full["run_id"])
        self.assertEqual(len(list((interrupted / "epochs").glob("epoch_*"))), 1)
        resumed = self.train(resume_run=interrupted.name)
        self.assert_same_optimizer(self.state(full), self.state(resumed))

    def test_upload_failure_preserves_epoch_and_retry_does_not_repeat_it(self):
        def fail(label, directory):
            if label == "epoch_000001":
                raise OSError("offline")
        with self.assertRaisesRegex(OSError, "offline"):
            self.train(checkpoint_callback=fail)
        directory = next((self.root / "results/epoch_runs").iterdir())
        before = (directory / "epochs/epoch_000001/state.joblib").read_bytes()
        events = []
        result = self.train(resume_run=directory.name, checkpoint_callback=lambda label, _: events.append(label))
        self.assertEqual((directory / "epochs/epoch_000001/state.joblib").read_bytes(), before)
        self.assertNotIn("epoch_000001", events)
        self.assertEqual(events[0], "epoch_session_started")
        self.assertEqual(result["status"], "completed")

    def test_changed_identity_and_corruption_reject_before_model_load(self):
        result = self.train(epochs_this_session=1)
        for option in ("alpha", "loss"):
            changed = dict(self.settings)
            changed[option] = .001 if option == "alpha" else "hinge"
            with patch("modules.epoch_training.joblib.load", side_effect=AssertionError("unsafe load")):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    run_epoch_training(self.root, self.config, self.rows, "b" * 64,
                        {"resnet50": self.directory}, changed, resume_run=result["run_id"])
        path = Path(result["run_dir"]) / "epochs/epoch_000001/state.joblib"
        path.write_bytes(path.read_bytes() + b"corruption")
        with patch("modules.epoch_training.joblib.load", side_effect=AssertionError("unsafe load")):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                self.train(resume_run=result["run_id"])

    def test_backup_restore_continues_in_new_root(self):
        from modules.checkpoints import save_backup, restore_backup
        full = self.train()
        part = self.train(epochs_this_session=2)
        backup = save_backup(self.root, Path(self.temporary.name) / "backups")
        restored = restore_backup(backup, Path(self.temporary.name) / "restored")
        result = run_epoch_training(restored, self.config, self.rows, "b" * 64,
            {"resnet50": restored / "features/resnet50/synthetic"}, self.settings, resume_run=part["run_id"])
        self.assert_same_optimizer(self.state(full), self.state(result))

    def test_hinge_pca_balanced_weights_and_best_epoch_are_saved(self):
        self.settings["loss"] = "hinge"
        self.settings["patience"] = 2
        self.config["training"]["pca_components"] = 2
        result = self.train()
        state = self.state(result)
        self.assertIn("pca", state["model"].named_steps)
        self.assertIsInstance(state["model"][-1].class_weight, dict)
        self.assertEqual(state["best_epoch"], max(state["history"], key=lambda r:r["val_macro_f1"])["epoch"])

    def test_final_requires_freeze_completeness_and_locks_training(self):
        result = self.train(epochs_this_session=1)
        args = (self.root, self.config, self.rows, "b" * 64, {"resnet50": self.directory}, self.settings)
        with self.assertRaisesRegex(ValueError, "FINAL_TEST"):
            run_epoch_final_test(*args, run_id=result["run_id"])
        with self.assertRaisesRegex(ValueError, "Complete"):
            run_epoch_final_test(*args, run_id=result["run_id"], confirm_frozen=True)
        result = self.train(resume_run=result["run_id"])
        with patch.object(SGDClassifier, "partial_fit", side_effect=AssertionError("no test fitting")):
            final = run_epoch_final_test(*args, run_id=result["run_id"], confirm_frozen=True)
        self.assertEqual(final["status"], "completed")
        with patch("modules.epoch_training.joblib.load", side_effect=AssertionError("reuse results only")):
            again = run_epoch_final_test(*args, run_id=result["run_id"], confirm_frozen=True)
        self.assertTrue(again["reused"])
        with self.assertRaisesRegex(ValueError, "Final test is locked"):
            self.train(resume_run=result["run_id"])

    def test_partial_staging_is_ignored_and_run_paths_are_restricted(self):
        part = self.train(epochs_this_session=1)
        (Path(part["run_dir"]) / "epochs/.staging-dead").mkdir()
        self.train(resume_run=part["run_id"])
        for value in (True, "../escape", "not-present"):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError, FileNotFoundError)):
                self.train(resume_run=value)

    def test_extreme_negative_scores_complete_epochs_export_and_final(self):
        original = SGDClassifier.partial_fit
        def extreme(model, *args, **kwargs):
            result = original(model, *args, **kwargs)
            model.coef_.fill(0)
            model.intercept_[:] = -10000.0 - np.arange(len(model.classes_))
            return result
        with patch.object(SGDClassifier, "partial_fit", new=extreme):
            result = self.train()
        self.assertEqual(result["status"], "completed")
        state = self.state(result)
        self.assertTrue(all(np.isfinite(row["train_loss_unweighted"]) for row in state["history"]))
        proba = state["model"].predict_proba(self.arrays["val"][0])
        self.assertTrue(np.isfinite(proba).all())
        np.testing.assert_allclose(proba.sum(axis=1), 1)
        self.assertTrue((Path(result["run_dir"]) / "validation/predictions.csv").is_file())
        final = run_epoch_final_test(self.root, self.config, self.rows, "b" * 64,
            {"resnet50": self.directory}, self.settings, run_id=result["run_id"], confirm_frozen=True)
        self.assertEqual(final["status"], "completed")

    def test_zero_epoch_run_with_changed_source_keeps_identity_guard(self):
        def stop_before_fit(label, directory):
            raise OSError("stop after registration")
        with self.assertRaises(OSError):
            self.train(checkpoint_callback=stop_before_fit)
        directory = next((self.root / "results/epoch_runs").iterdir())
        identity_path = directory / "identity.json"
        identity = json.loads(identity_path.read_text())
        identity["implementation"]["epoch_training.py"] = "previous-source"
        identity_path.write_text(json.dumps(identity))
        before = identity_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.train(resume_run=directory.name)
        self.assertEqual(identity_path.read_bytes(), before)
        new = self.train(epochs_this_session=1)
        self.assertNotEqual(new["run_id"], directory.name)

    def test_nonfinite_scores_do_not_commit_a_completed_epoch(self):
        original = SGDClassifier.partial_fit
        def invalid(model, *args, **kwargs):
            result = original(model, *args, **kwargs)
            model.intercept_[:] = np.inf
            return result
        with patch.object(SGDClassifier, "partial_fit", new=invalid):
            with self.assertRaises((FloatingPointError, ValueError)):
                self.train()
        directory = next((self.root / "results/epoch_runs").iterdir())
        self.assertEqual(list((directory / "epochs").glob("epoch_*")), [])
        self.assertFalse((directory / "selection.json").exists())


class StableProbabilityTests(unittest.TestCase):
    def model(self, scores, stable):
        scores = np.asarray(scores, dtype=np.float64)
        model = (_StableSGDClassifier if stable else SGDClassifier)(loss="log_loss")
        model.classes_ = np.array([3, 7]) if scores.ndim == 1 else np.arange(scores.shape[1]) * 2 + 3
        if scores.ndim == 1:
            model.coef_ = np.array([[1.0]])
            model.intercept_ = np.zeros(1)
        else:
            model.coef_ = np.eye(scores.shape[1])
            model.intercept_ = np.zeros(scores.shape[1])
        return model

    def test_matches_sklearn_probabilities_and_labels_on_regular_scores(self):
        for scores in (np.array([-20., -1., 0., 2., 20.]),
                       np.array([[-3., 0., 2.], [4., -2., 1.]])):
            x = scores[:, None] if scores.ndim == 1 else scores
            base, stable = self.model(scores, False), self.model(scores, True)
            np.testing.assert_allclose(stable.predict_proba(x), base.predict_proba(x), rtol=1e-12, atol=1e-15)
            np.testing.assert_array_equal(stable.predict(x), base.predict(x))
            np.testing.assert_allclose(np.exp(stable.predict_log_proba(x)), stable.predict_proba(x))

    def test_all_zero_sigmoids_normalize_without_nan_or_uniform_replacement(self):
        scores = np.array([[-10000., -10001., -10002.], [-1e300, -1e300, -1e300]])
        with np.errstate(all="raise"):
            # Ignore benign tiny exponential underflow; no invalid division is allowed.
            with np.errstate(under="ignore"):
                probabilities = np.exp(_stable_log_probabilities(scores))
        expected = np.exp([0., -1., -2.]); expected /= expected.sum()
        np.testing.assert_allclose(probabilities[0], expected)
        np.testing.assert_allclose(probabilities[1], np.ones(3) / 3)
        self.assertTrue(np.isfinite(log_loss([0, 1], probabilities, labels=[0, 1, 2])))

    def test_nonfinite_scores_fail_and_hinge_does_not_advertise_probabilities(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaisesRegex(FloatingPointError, "Non-finite"):
                _stable_log_probabilities(np.array([[value, 0., 1.]]))
        model = _StableSGDClassifier(loss="hinge")
        self.assertFalse(hasattr(model, "predict_proba"))
        self.assertFalse(hasattr(model, "predict_log_proba"))

    def test_stable_subclass_has_identical_partial_fit_optimizer_updates(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(15, 4))
        y = np.arange(15) % 3
        options = dict(loss="log_loss", shuffle=False, random_state=42, alpha=.0001)
        base, stable = SGDClassifier(**options), _StableSGDClassifier(**options)
        for start in (0, 5, 10):
            for model in (base, stable):
                model.partial_fit(x[start:start + 5], y[start:start + 5], classes=np.arange(3))
        np.testing.assert_array_equal(base.coef_, stable.coef_)
        np.testing.assert_array_equal(base.intercept_, stable.intercept_)
        self.assertEqual(base.t_, stable.t_)
