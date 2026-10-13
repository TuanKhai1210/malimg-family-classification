"""Synthetic archive checks for integrity, leakage and reproducible preparation."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import tempfile
import unittest
import zipfile
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from PIL import Image, PngImagePlugin

from modules.data import download_archive, prepare_dataset, read_manifest, sha256_file
from modules.splits import assign_group_splits


def _png(value: int, annotation: str = "") -> bytes:
    buffer = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    info.add_text("note", annotation)
    Image.new("L", (5, 7), color=value).save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


def _write_archive(path: Path, *, groups: int = 20, conflict: bool = False, unsafe: str | None = None, symlink: bool = False) -> dict:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for label_index, label in enumerate(("Alpha", "Beta")):
            for index in range(groups):
                value = index + 1 + label_index * 80
                if conflict and label == "Beta" and index == 0:
                    value = 1
                archive.writestr(f"source/{label}/{index:03}.png", _png(value))
            # Same decoded image, different encoded PNG bytes; lexicographic
            # representative must be the original 000.png regardless of split.
            archive.writestr(f"source/{label}/duplicate.png", _png(1 + label_index * 80, "different PNG metadata"))
        archive.writestr("source/Yuner.A/a.png", _png(240))
        archive.writestr("source/Yuner.A/b.png", _png(240, "copy"))
        archive.writestr("source/README.txt", "Not an image; do not extract.")
        if unsafe:
            archive.writestr(unsafe, "unsafe ignored file")
        if symlink:
            info = zipfile.ZipInfo("source/link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "../../other")
    return {
        "dataset_name": "synthetic", "dataset_url": "https://example.invalid/dataset.zip",
        "archive_bytes": path.stat().st_size, "archive_sha256": sha256_file(path),
        "excluded_families": ["Yuner.A"], "split_seed": 42,
        "split_target": {"train": 0.70, "val": 0.15, "test": 0.15},
    }


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.archive = self.directory / "source.zip"
        self.config = _write_archive(self.archive)
        self.root = self.directory / "project"

    def prepare(self, **changes):
        return prepare_dataset(self.root, dict(self.config, **changes), self.archive)

    def test_inventory_split_exclusion_and_representatives(self):
        result = self.prepare()
        dataset = read_manifest(result["dataset_manifest"])
        split = read_manifest(result["manifest_path"])
        self.assertEqual(len(dataset), 44)
        self.assertEqual(len(split), 42)
        self.assertEqual(json.loads(Path(result["label_map_path"]).read_text()), {"Alpha": 0, "Beta": 1})
        excluded = [row for row in dataset if not row["included"]]
        self.assertEqual(len(excluded), 2)
        self.assertTrue(all(row["split"] == "excluded" and row["label_id"] == -1 for row in excluded))
        self.assertTrue(all((Path(result["data_root"]) / row["relative_path"]).is_file() for row in dataset))
        self.assertFalse(any(Path(result["data_root"]).rglob("*.txt")))
        self.assertEqual(result["manifest_sha256"], sha256_file(result["manifest_path"]))
        groups = {}
        for row in split:
            groups.setdefault(row["pixel_sha256"], []).append(row)
            self.assertIsInstance(row["label_id"], int)
            self.assertIsInstance(row["eval_representative"], bool)
        for members in groups.values():
            self.assertEqual(len({row["split"] for row in members}), 1)
            representatives = [row for row in members if row["eval_representative"]]
            self.assertEqual(len(representatives), 1)
            self.assertEqual(representatives[0]["relative_path"], min(row["relative_path"] for row in members))
        for label in ("Alpha", "Beta"):
            for partition in ("train", "val", "test"):
                selected = [row for row in split if row["label"] == label and row["split"] == partition]
                self.assertGreaterEqual(len({row["pixel_sha256"] for row in selected}), 1 if partition == "train" else 2)

    def test_exact_pixel_hash_matches_prior_audit_scheme(self):
        result = self.prepare()
        rows = read_manifest(result["manifest_path"])
        original = next(row for row in rows if row["relative_path"] == "Alpha/000.png")
        duplicate = next(row for row in rows if row["relative_path"] == "Alpha/duplicate.png")
        self.assertNotEqual(original["file_sha256"], duplicate["file_sha256"])
        self.assertEqual(original["pixel_sha256"], duplicate["pixel_sha256"])
        self.assertEqual(original["pixel_sha256"], hashlib.sha256(b"L:5x7:" + bytes([1] * 35)).hexdigest())

    def test_preparation_is_reproducible_across_locations_and_reuse(self):
        first = self.prepare()
        first_content = Path(first["manifest_path"]).read_bytes()
        second = prepare_dataset(self.directory / "other", self.config, self.archive)
        self.assertEqual(first_content, Path(second["manifest_path"]).read_bytes())
        reused = self.prepare()
        self.assertEqual(first["manifest_sha256"], reused["manifest_sha256"])

    def test_archive_size_and_hash_are_required(self):
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            self.prepare(archive_bytes=self.config["archive_bytes"] + 1)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.prepare(archive_sha256="0" * 64)
        self.assertFalse((self.root / "data/images").exists())

    def test_traversal_even_in_ignored_entry_is_rejected(self):
        for unsafe in ("../escape.txt", "/absolute.txt", "root/../../escape.txt", "C:/drive.txt", "root\\..\\escape.txt"):
            with self.subTest(path=unsafe):
                config = _write_archive(self.archive, unsafe=unsafe)
                with self.assertRaisesRegex(ValueError, "unsafe archive path"):
                    prepare_dataset(self.root, config, self.archive)
                self.assertFalse((self.root / "data/images").exists())

    def test_zip_symlink_is_rejected(self):
        config = _write_archive(self.archive, symlink=True)
        with self.assertRaisesRegex(ValueError, "non-regular archive entry"):
            prepare_dataset(self.root, config, self.archive)

    def test_unreadable_png_is_rejected_without_publishing_images(self):
        with zipfile.ZipFile(self.archive, "a") as archive:
            archive.writestr("source/Alpha/broken.png", b"not an image")
        config = dict(self.config, archive_bytes=self.archive.stat().st_size, archive_sha256=sha256_file(self.archive))
        with self.assertRaises((OSError, ValueError)):
            prepare_dataset(self.root, config, self.archive)
        self.assertFalse((self.root / "data/images").exists())

    def test_case_alias_family_directories_are_rejected(self):
        with zipfile.ZipFile(self.archive, "a") as archive:
            archive.writestr("source/alpha/extra.png", _png(250))
        config = dict(self.config, archive_bytes=self.archive.stat().st_size, archive_sha256=sha256_file(self.archive))
        with self.assertRaisesRegex(ValueError, "family names differ only in case"):
            prepare_dataset(self.root, config, self.archive)

    def test_cross_label_pixel_conflict_is_rejected(self):
        config = _write_archive(self.archive, conflict=True)
        with self.assertRaisesRegex(ValueError, "cross-label pixel conflict"):
            prepare_dataset(self.root, config, self.archive)

    def test_class_with_insufficient_groups_is_rejected(self):
        config = _write_archive(self.archive, groups=2)
        with self.assertRaisesRegex(ValueError, "require at least 3"):
            prepare_dataset(self.root, config, self.archive)
        self.assertFalse((self.root / "metadata/split_manifest_v1.csv").exists())

    def test_existing_image_tampering_is_rejected(self):
        result = self.prepare()
        image = Path(result["data_root"]) / "Alpha/000.png"
        image.write_bytes(_png(254))
        with self.assertRaisesRegex(ValueError, "existing image integrity mismatch"):
            self.prepare()

    def test_existing_manifest_tampering_is_rejected(self):
        result = self.prepare()
        manifest = Path(result["manifest_path"])
        manifest.write_bytes(manifest.read_bytes().replace(b"Alpha/000.png", b"Alpha/tampered.png"))
        with self.assertRaisesRegex(ValueError, "artifact integrity mismatch"):
            self.prepare()

    def test_source_inventory_tampering_is_rejected(self):
        result = self.prepare()
        inventory = Path(result["data_root"]).parent / "image_inventory.json"
        inventory.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source inventory does not match"):
            self.prepare()

    def test_missing_image_is_rejected_on_reuse(self):
        result = self.prepare()
        (Path(result["data_root"]) / "Alpha/000.png").unlink()
        with self.assertRaisesRegex(ValueError, "do not match the verified source inventory"):
            self.prepare()

    def test_config_change_requires_a_new_manifest_version(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "increment manifest_version"):
            self.prepare(split_seed=17)
        result = self.prepare(split_seed=17, manifest_version="v2")
        self.assertTrue(Path(result["manifest_path"]).name.endswith("v2.csv"))

    def test_configured_data_directory_is_respected(self):
        result = self.prepare(data_dir="custom_data")
        self.assertEqual(Path(result["data_root"]), self.root / "custom_data")

    def test_three_groups_fall_back_to_one_group_per_partition(self):
        rows = [{"sample_id": str(index), "relative_path": f"family/{index}.png", "label": "family", "label_id": 0, "pixel_sha256": str(index), "included": True} for index in range(3)]
        split = assign_group_splits(rows)
        self.assertEqual({row["split"] for row in split}, {"train", "val", "test"})
        self.assertTrue(all(row["eval_representative"] for row in split))

    def test_download_publishes_only_verified_archive(self):
        source = self.archive.read_bytes()

        class Response(io.BytesIO):
            status = 200
            headers = {}

        destination = self.directory / "download.zip"
        with patch("modules.data.urllib.request.urlopen", return_value=Response(source)):
            self.assertEqual(download_archive(self.config, destination), destination)
        self.assertEqual(destination.read_bytes(), source)
        self.assertFalse(destination.with_suffix(".zip.part").exists())
        destination.unlink()
        with patch("modules.data.urllib.request.urlopen", return_value=Response(source[:-1])):
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                download_archive(self.config, destination)
        self.assertFalse(destination.exists())

    @unittest.skipUnless(find_spec("matplotlib"), "install requirements.txt to render EDA")
    def test_eda_writes_descriptive_tables_and_figures(self):
        from modules.eda import generate_eda

        result = self.prepare()
        rows = read_manifest(result["dataset_manifest"])
        eda = generate_eda(rows, Path(result["data_root"]), self.root / "results/eda")
        self.assertEqual(eda["summary"]["files"], 44)
        self.assertEqual(eda["summary"]["excluded_files"], 2)
        for key, value in eda.items():
            if key != "summary":
                self.assertTrue(Path(value).is_file(), key)


if __name__ == "__main__":
    unittest.main()
