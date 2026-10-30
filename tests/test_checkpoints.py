"""Backups are verified and restorable without executing saved models."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from modules.checkpoints import restore_backup, save_backup


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "source"
        for name in ("modules/example.py", "features/resnet/cache/X_train.npy",
                     "results/runs/run1/E2/trial_00/model.joblib", "metadata/split.csv",
                     "data/images/raw.png", ".env", "features/resnet/.unfinished.tmp/partial.npy"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        self.backup = self.base / "drive"

    def test_versions_share_features_and_restore_exact_files_without_raw_data(self):
        first = save_backup(self.root, self.backup, label="trial_00")
        (self.root / "results/runs/run1/E2/trial_00/model.joblib").write_bytes(b"next version")
        second = save_backup(self.root, self.backup, label="trial_01")
        self.assertNotEqual(first, second)
        self.assertEqual(len(list((self.backup / "features").glob("*.zip"))), 1)
        for archive_path, expected in ((first, b"results/runs/run1/E2/trial_00/model.joblib"),
                                       (second, b"next version")):
            restored = self.base / archive_path.stem
            restore_backup(archive_path, restored)
            self.assertEqual((restored / "results/runs/run1/E2/trial_00/model.joblib").read_bytes(), expected)
            self.assertTrue((restored / "features/resnet/cache/X_train.npy").is_file())
            self.assertFalse((restored / "data").exists())
            self.assertFalse((restored / ".env").exists())
            self.assertFalse((restored / "features/resnet/.unfinished.tmp").exists())
        with self.assertRaisesRegex(ValueError, "new directory"):
            restore_backup(second, restored)

    def test_corrupt_feature_bundle_is_rejected_before_restore_creates_workspace(self):
        snapshot = save_backup(self.root, self.backup)
        bundle = next((self.backup / "features").glob("*.zip"))
        with bundle.open("ab") as stream:
            stream.write(b"corruption")
        target = self.base / "restored"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            restore_backup(snapshot, target)
        self.assertFalse(target.exists())

    def test_modified_state_and_unsafe_zip_members_are_rejected(self):
        snapshot = save_backup(self.root, self.backup)
        # Rewrite a listed source file without updating the embedded checksum.
        rewritten = self.base / "rewritten.zip"
        with zipfile.ZipFile(snapshot) as source, zipfile.ZipFile(rewritten, "w") as output:
            for member in source.infolist():
                content = b"modified" if member.filename == "modules/example.py" else source.read(member)
                output.writestr(member, content)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            restore_backup(rewritten, self.base / "changed")
        unsafe = self.base / "unsafe.zip"
        with zipfile.ZipFile(snapshot) as source, zipfile.ZipFile(unsafe, "w") as output:
            for member in source.infolist():
                output.writestr(member, source.read(member))
            output.writestr("../escape.txt", "no")
        with self.assertRaisesRegex(ValueError, "Unsafe backup member"):
            restore_backup(unsafe, self.base / "unsafe-target")
        self.assertFalse((self.base / "escape.txt").exists())

    def test_backups_reject_recursive_destination_and_unpersisted_output_paths(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            save_backup(self.root, self.root / "backup")
        with self.assertRaisesRegex(ValueError, "Persist results_dir"):
            save_backup(self.root, self.backup, config={"results_dir": "other"})
        with self.assertRaisesRegex(ValueError, "inside"):
            save_backup(self.root, self.backup, config={"features_dir": str(self.base / "outside")})

    def test_failed_copy_does_not_replace_existing_backup_or_leave_partial_file(self):
        from modules.checkpoints import _publish
        source, target = self.base / "source.zip", self.base / "saved.zip"
        source.write_bytes(b"new complete backup")
        target.write_bytes(b"previous verified backup")

        def fail_copy(source_path, target_path):
            Path(target_path).write_bytes(b"partial")
            raise OSError("synthetic full disk")

        with patch("modules.checkpoints.shutil.copyfile", side_effect=fail_copy):
            with self.assertRaisesRegex(OSError, "full disk"):
                _publish(source, target)
        self.assertEqual(target.read_bytes(), b"previous verified backup")
        self.assertEqual(list(self.base.glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
