"""Remote transport fixtures: private-only, readback, outages and recovery."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from modules.kaggle_backup import KaggleDatasetBackup
from modules.checkpoints import restore_backup


class MissingDataset(Exception):
    status = 404


class FakeKaggle:
    """A byte-preserving provider fixture, not evidence of live Kaggle I/O."""
    def __init__(self):
        self.exists = False
        self.private = True
        self.files = {}
        self.versions = []
        self.corrupt_download = False
        self.processing = False

    def dataset_metadata(self, reference, folder):
        if not self.exists:
            raise MissingDataset()
        path = Path(folder) / "dataset-metadata.json"
        path.write_text(json.dumps({"id": reference, "isPrivate": self.private}), encoding="utf-8")
        return str(path)

    def dataset_status(self, reference):
        return "processing" if self.processing else "ready"

    def _publish(self, folder):
        self.files = {p.name: p.read_bytes() for p in Path(folder).iterdir() if p.is_file()}
        self.versions.append(dict(self.files))
        self.exists = True
        return SimpleNamespace(status="ok", error=None)

    def dataset_create_new(self, folder, *, public, **kwargs):
        assert public is False
        return self._publish(folder)

    def dataset_create_version(self, folder, *, delete_old_versions, **kwargs):
        assert delete_old_versions is False
        return self._publish(folder)

    def dataset_download_file(self, reference, name, *, path, **kwargs):
        data = self.files[name]
        if self.corrupt_download and name == "recovery.zip":
            data += b"corruption"
        (Path(path) / name).write_bytes(data)
        return True


class KaggleBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "workspace"
        (self.root / "configs").mkdir(parents=True)
        (self.root / "configs/project.json").write_text(json.dumps({"results_dir": "results", "features_dir": "features"}), encoding="utf-8")
        (self.root / "features/resnet50").mkdir(parents=True)
        (self.root / "features/resnet50/fixture.npy").write_bytes(b"synthetic fixed feature bytes")
        (self.root / "results").mkdir()
        (self.root / "results/progress.json").write_text('{"epoch": 1}', encoding="utf-8")
        self.api = FakeKaggle()
        self.destination = Path(self.temp.name) / "backups"

    def backup(self, **kwargs):
        return KaggleDatasetBackup(self.root, self.destination, "fixture-owner/malimg-checkpoints", api=self.api,
                                   sleep=lambda _:None, poll_timeout=2, poll_interval=1, **kwargs)

    def test_private_publish_readback_and_recovery_keep_previous_versions(self):
        backup = self.backup()
        one = backup("epoch_000001", Path("sgd_fixture"))
        (self.root / "results/progress.json").write_text('{"epoch": 2}', encoding="utf-8")
        two = backup("epoch_000002", Path("sgd_fixture"))
        self.assertTrue(two["remote_verified"])
        self.assertNotEqual(one["checkpoint_id"], two["checkpoint_id"])
        self.assertEqual(len(self.api.versions), 2)
        snapshot = backup.download(Path(self.temp.name) / "download")
        restored = restore_backup(snapshot, Path(self.temp.name) / "restored")
        self.assertEqual(json.loads((restored / "results/progress.json").read_text())["epoch"], 2)
        self.assertEqual((restored / "features/resnet50/fixture.npy").read_bytes(), b"synthetic fixed feature bytes")
        self.assertEqual(json.loads(self.api.versions[0]["latest.json"])["label"], "epoch_000001")

    def test_public_or_unknown_privacy_is_rejected_before_upload(self):
        self.api.exists = True
        for value in (False, None):
            self.api.private = value
            with self.assertRaisesRegex(ValueError, "privacy|public"):
                self.backup()
        self.assertEqual(self.api.versions, [])

    def test_failed_readback_keeps_prior_receipt_and_all_local_snapshots(self):
        backup = self.backup()
        backup("epoch_000001")
        receipt = (self.destination / "remote_receipt.json").read_bytes()
        self.api.corrupt_download = True
        with self.assertRaisesRegex(RuntimeError, "NOT verified"):
            backup("epoch_000002")
        self.assertEqual((self.destination / "remote_receipt.json").read_bytes(), receipt)
        self.assertEqual(len(list((self.destination / "snapshots").glob("*.zip"))), 2)

    def test_version_processing_timeout_does_not_claim_success(self):
        backup = self.backup()
        original = self.api.dataset_create_new
        def processing(*args, **kwargs):
            result = original(*args, **kwargs)
            self.api.processing = True
            return result
        self.api.dataset_create_new = processing
        with self.assertRaisesRegex(RuntimeError, "NOT verified"):
            backup("epoch_000001")
        self.assertFalse((self.destination / "remote_receipt.json").exists())

    def test_errors_are_sanitized_and_no_credentials_enter_upload(self):
        backup = self.backup()
        def error(*args, **kwargs):
            raise OSError("PRIVATE_TOKEN_SHOULD_NOT_APPEAR")
        self.api.dataset_create_new = error
        with self.assertRaises(RuntimeError) as captured:
            backup("epoch_000001")
        self.assertNotIn("PRIVATE_TOKEN", str(captured.exception))
        self.assertTrue(list((self.destination / "snapshots").glob("*.zip")))

    def test_conflicting_remote_writer_is_detected(self):
        backup = self.backup()
        original = self.api.dataset_download_file
        def conflicting(ref, name, **kwargs):
            result = original(ref, name, **kwargs)
            if name == "latest.json":
                path = Path(kwargs["path"]) / name
                record = json.loads(path.read_text())
                record["checkpoint_id"] = "another-writer"
                path.write_text(json.dumps(record))
            return result
        self.api.dataset_download_file = conflicting
        with self.assertRaisesRegex(RuntimeError, "NOT verified"):
            backup("epoch_000001")
