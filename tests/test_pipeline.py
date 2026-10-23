"""Exercise the complete base with synthetic images, never the real benchmark."""

import ast
import copy
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path

from PIL import Image

from modules.config import load_config, validate_config
from modules.pipeline import run_pipeline
from modules.submission import build_submission

ROOT = Path(__file__).resolve().parent.parent


class PipelineTests(unittest.TestCase):
    def config(self):
        return copy.deepcopy(load_config(ROOT / "configs/project.json"))

    def test_rejects_invalid_search_and_external_output_paths(self):
        for key, value in (("c_values", [float("nan")]), ("c_values", [-1]), ("id", "../../bad")):
            config = self.config()
            config["experiments"][0][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)
        config = self.config()
        config["results_dir"] = "../outside"
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_notebook_code_is_valid_python_and_has_no_fake_outputs(self):
        notebook = json.loads((ROOT / "notebooks/main_colab.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = cell["source"]
                ast.parse(source if isinstance(source, str) else "".join(source))
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_prepare_extract_train_final_and_draft_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "synthetic.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for label_index, label in enumerate(("Alpha", "Beta")):
                    for index in range(12):
                        image = Image.new("L", (32 + label_index * 32 + index, 48 + index),
                                          color=label_index * 100 + index)
                        stream = io.BytesIO()
                        image.save(stream, format="PNG")
                        archive.writestr(f"images/{label}/{index:02d}.png", stream.getvalue())
            config = self.config()
            config.update(dataset_name="Synthetic integration fixture", excluded_families=[],
                          instructor_approved=True,
                          archive_bytes=archive_path.stat().st_size,
                          archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest())
            config["experiments"] = [{"id": "E0", "extractor": "image_size",
                                       "classifier": "logistic_regression", "c_values": [0.1, 1]}]
            prepared = run_pipeline(root, config, stage="prepare", archive_path=archive_path)
            self.assertFalse((root / "results/selection.json").exists())
            self.assertTrue(Path(prepared["eda"]["class_distribution_figure"]).is_file())
            trained = run_pipeline(root, config, stage="all", archive_path=archive_path)
            self.assertEqual(trained["test_status"].split(";")[0], "held out")
            self.assertFalse((root / "results/final_test.lock.json").exists())
            run_pipeline(root, config, stage="final", archive_path=archive_path)
            lock = json.loads((root / "results/final_test.lock.json").read_text(encoding="utf-8"))
            self.assertEqual(lock["status"], "completed")
            with patch("modules.features.extract_features") as extraction:
                with self.assertRaisesRegex(ValueError, "Final test artifacts"):
                    run_pipeline(root, config, stage="train", archive_path=archive_path)
                extraction.assert_not_called()
            (root / "configs").mkdir()
            (root / "configs/project.json").write_text(json.dumps(config), encoding="utf-8")
            (root / "notebooks").mkdir()
            (root / "notebooks/main_colab.ipynb").write_bytes((ROOT / "notebooks/main_colab.ipynb").read_bytes())
            (root / "reports").mkdir()
            package = build_submission(root, root / "draft.zip")
            with zipfile.ZipFile(package) as archive:
                names = archive.namelist()
                self.assertIn("features/", names)
                self.assertFalse(any(name.startswith("data/") for name in names))
                self.assertTrue(any(name.endswith(".joblib") for name in names))
                self.assertTrue(json.loads(archive.read("PACKAGE.json"))["draft"])
            unapproved = dict(config, instructor_approved=False)
            (root / "configs/project.json").write_text(json.dumps(unapproved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "approval"):
                build_submission(root, root / "final.zip", draft=False)
            (root / "configs/project.json").write_text(json.dumps(config), encoding="utf-8")
            (root / "reports/final_report.pdf").write_bytes(b"%PDF- synthetic packaging test only")
            build_submission(root, root / "final.zip", draft=False)
            test_array = next((root / "features").rglob("X_test.npy"))
            with test_array.open("ab") as stream:
                stream.write(b"modified")
            with self.assertRaisesRegex(ValueError, "feature artifact has changed"):
                build_submission(root, root / "altered.zip", draft=False)

    def test_draft_package_includes_custom_result_folder_and_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs/project.json").write_text('{"results_dir":"custom_results"}', encoding="utf-8")
            (root / "custom_results").mkdir()
            (root / "custom_results/registry.json").write_text('{}', encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests/test_example.py").write_text('pass', encoding="utf-8")
            package = build_submission(root, root / "draft.zip")
            with zipfile.ZipFile(package) as archive:
                self.assertIn("custom_results/registry.json", archive.namelist())
                self.assertIn("tests/test_example.py", archive.namelist())

    def test_notebook_saves_then_restores_and_resumes_in_a_fresh_python_process(self):
        """Execute every code cell; fake only the interactive Drive mount."""
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "workspace"
            root.mkdir()
            for name in ("modules", "scripts", "notebooks", "configs"):
                shutil.copytree(ROOT / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copyfile(ROOT / "requirements.txt", root / "requirements.txt")
            archive_path = base / "synthetic.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for class_id in range(2):
                    for index in range(12):
                        image = Image.new("L", (32 + index, 40 + class_id * 20), index + class_id * 80)
                        stream = io.BytesIO()
                        image.save(stream, format="PNG")
                        archive.writestr(f"images/Class{class_id}/{index}.png", stream.getvalue())
            config = self.config()
            config.update(excluded_families=[], archive_bytes=archive_path.stat().st_size,
                          archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest())
            config["experiments"] = [{"id": "E0", "extractor": "image_size",
                                      "classifier": "logistic_regression", "c_values": [0.1, 1]}]
            (root / "configs/project.json").write_text(json.dumps(config), encoding="utf-8")
            worker = base / "execute_notebook.py"
            worker.write_text('''import json, pathlib, sys, types
from unittest.mock import patch
options = json.loads(pathlib.Path(sys.argv[1]).read_text())
colab = types.ModuleType('google.colab')
colab.drive = types.SimpleNamespace(mount=lambda path: print('Synthetic mount:', path))
sys.modules['google.colab'] = colab
notebook = json.loads(pathlib.Path(options['notebook']).read_text(encoding='utf-8'))
scope = {}
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])
    if cell['id'] == 'pipeline-run' and options.get('RESUME_RUN'):
        with patch('modules.experiments.build_classifier', side_effect=AssertionError('must resume')):
            exec(compile(source, cell['id'], 'exec'), scope)
    else:
        exec(compile(source, cell['id'], 'exec'), scope)
    if cell['id'] == 'controls':
        scope.update({key: value for key, value in options.items() if key != 'notebook'})
''', encoding="utf-8")
            options = {"notebook": str(root / "notebooks/main_colab.ipynb"),
                       "STAGE": "all", "INSTALL_DEPENDENCIES": False,
                       "ARCHIVE_PATH": str(archive_path), "USE_DRIVE_BACKUP": True,
                       "BACKUP_DIR": str(base / "drive")}
            def execute(settings):
                payload = base / "options.json"
                payload.write_text(json.dumps(settings), encoding="utf-8")
                result = subprocess.run([sys.executable, "-X", "utf8", str(worker), str(payload)],
                                        cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
                return result.stdout
            execute(options)
            run_id = next((root / "results/runs").glob("validation_*")).name
            snapshot = sorted((base / "drive/snapshots").glob("*.zip"))[-1]
            restored = base / "restored"
            output = execute({**options, "STAGE": "train", "RESTORE_SNAPSHOT": str(snapshot),
                              "RESTORE_TO": str(restored), "RESUME_RUN": run_id})
            self.assertIn("Reused verified checkpoint", output)
            self.assertTrue((restored / "results/selection.json").is_file())
            self.assertFalse((restored / "results/final_test.lock.json").exists())


if __name__ == "__main__":
    unittest.main()
