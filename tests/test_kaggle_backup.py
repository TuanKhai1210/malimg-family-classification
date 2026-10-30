"""Remote transport fixtures: private-only, readback, outages and recovery."""

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from modules.kaggle_backup import KaggleDatasetBackup, RECOVERY_FILE, RECOVERY_MAGIC, unpack_recovery
from modules.checkpoints import restore_backup


class MissingDataset(Exception):
    status = 404


class FakeKaggle:
    """Model archive expansion as well as transport; not live Kaggle evidence."""
    def __init__(self):
        self.exists = False
        self.private = True
        self.files = {}
        self.versions = []
        self.corrupt_download = False
        self.processing = False
        self.downloads = []

    def dataset_metadata(self, reference, folder):
        if not self.exists:
            raise MissingDataset()
        path = Path(folder) / "dataset-metadata.json"
        path.write_text(json.dumps({"id": reference, "isPrivate": self.private}), encoding="utf-8")
        return str(path)

    def dataset_status(self, reference):
        return "processing" if self.processing else "ready"

    def _publish(self, folder):
        self.files = {}
        def add(name, data):
            if name.endswith(".zip") or data.startswith(b"PK\x03\x04"):
                prefix = name[:-4] if name.endswith(".zip") else name
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    for child in archive.namelist():
                        add(prefix + "/" + child, archive.read(child))
            else:
                self.files[name] = data
        for path in Path(folder).iterdir():
            if path.is_file() and path.name != "dataset-metadata.json":
                add(path.name, path.read_bytes())
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
        self.downloads.append(name)
        data = self.files[name]
        if self.corrupt_download and name in {"recovery.zip", RECOVERY_FILE}:
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

    def test_provider_expands_legacy_zip_but_framed_payload_survives(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            with zipfile.ZipFile(folder / "recovery.zip", "w") as archive:
                archive.writestr("example.txt", "provider expands this archive")
            self.api._publish(folder)
        self.assertNotIn("recovery.zip", self.api.files)
        self.assertIn("recovery/example.txt", self.api.files)
        receipt = self.backup()("epoch_000001")
        self.assertEqual(receipt["schema"], 2)
        self.assertEqual(receipt["recovery_file"], RECOVERY_FILE)
        self.assertEqual(set(self.api.files), {"latest.json", RECOVERY_FILE})
        self.assertTrue(self.api.files[RECOVERY_FILE].startswith(RECOVERY_MAGIC))
        self.assertTrue(receipt["remote_verified"])

    def test_legacy_unexpanded_zip_receipt_can_still_restore(self):
        backup = self.backup()
        backup("epoch_000001")
        with zipfile.ZipFile(io.BytesIO(self.api.files[RECOVERY_FILE])) as framed:
            legacy = io.BytesIO()
            with zipfile.ZipFile(legacy, "w", zipfile.ZIP_STORED) as archive:
                for name in framed.namelist():
                    archive.writestr(name, framed.read(name))
        data = legacy.getvalue()
        receipt = json.loads(self.api.files["latest.json"])
        receipt["schema"] = 1
        receipt.pop("recovery_file")
        receipt["recovery_sha256"] = hashlib.sha256(data).hexdigest()
        self.api.files = {"latest.json": json.dumps(receipt).encode(), "recovery.zip": data}
        snapshot = backup.download(Path(self.temp.name) / "legacy-download")
        restored = restore_backup(snapshot, Path(self.temp.name) / "legacy-restored")
        self.assertEqual(json.loads((restored / "results/progress.json").read_text())["epoch"], 1)

    def test_unknown_receipt_schema_or_filename_stops_before_payload_download(self):
        backup = self.backup()
        backup("epoch_000001")
        original = json.loads(self.api.files["latest.json"])
        for patch in ({"schema": 3}, {"schema": True}, {"recovery_file": "../private.bin"}):
            receipt = dict(original, **patch)
            self.api.files["latest.json"] = json.dumps(receipt).encode()
            self.api.downloads.clear()
            with self.assertRaisesRegex(ValueError, "schema|filename"):
                backup.download(Path(self.temp.name) / ("invalid-" + str(len(list(Path(self.temp.name).iterdir())))))
            self.assertEqual(self.api.downloads, ["latest.json"])

    def test_framed_payload_requires_header_even_with_matching_checksum(self):
        self.backup()("epoch_000001")
        receipt = json.loads(self.api.files["latest.json"])
        data = b"X" * len(RECOVERY_MAGIC) + self.api.files[RECOVERY_FILE][len(RECOVERY_MAGIC):]
        payload = Path(self.temp.name) / RECOVERY_FILE
        payload.write_bytes(data)
        receipt["recovery_sha256"] = hashlib.sha256(data).hexdigest()
        destination = Path(self.temp.name) / "bad-header"
        with self.assertRaisesRegex(ValueError, "header"):
            unpack_recovery(payload, destination, receipt=receipt)
        self.assertFalse(destination.exists())
