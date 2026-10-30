"""Execute all Kaggle cells on synthetic images, then restore in a new process."""

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
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks/main_kaggle.ipynb"


class KaggleNotebookTests(unittest.TestCase):
    def cells(self):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        return {c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}

    def test_notebook_has_executable_clean_cells_and_complete_path_controls(self):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                ast.parse("".join(c["source"]))
                self.assertEqual(c["outputs"], [])
                self.assertIsNone(c["execution_count"])
        cells = self.cells()
        for name in ("WORKSPACE", "WORKING_ROOT", "CONFIG_FILE", "BACKUP_DIR", "RESTORE_SNAPSHOT",
                     "RESTORE_PAYLOAD", "RESTORE_TO", "ARCHIVE_PATH", "RUN_ID"):
            for value in (True, False):
                with self.subTest(name=name, value=value):
                    scope = {}
                    exec(cells["controls"], scope)
                    scope[name] = value
                    with patch("subprocess.run") as run:
                        with self.assertRaisesRegex(TypeError, name):
                            exec(cells["setup"], scope)
                        run.assert_not_called()

    def test_full_notebook_and_fresh_process_restore_resume_without_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            workspace = base / "workspace"
            workspace.mkdir()
            for folder in ("modules", "scripts", "notebooks"):
                shutil.copytree(ROOT / folder, workspace / folder, ignore=shutil.ignore_patterns("__pycache__"))
            (workspace / "configs").mkdir()
            archive_path = base / "images.zip"
            with zipfile.ZipFile(archive_path, "w") as z:
                for label_index, label in enumerate(("Alpha", "Beta")):
                    for index in range(12):
                        image = Image.new("L", (32 + label_index * 32 + index, 48 + index), label_index * 100 + index)
                        stream = io.BytesIO()
                        image.save(stream, format="PNG")
                        z.writestr(f"images/{label}/{index:02d}.png", stream.getvalue())
            config = json.loads((ROOT / "configs/project.json").read_text())
            config.update(dataset_name="Synthetic Kaggle notebook fixture", excluded_families=[],
                          archive_bytes=archive_path.stat().st_size,
                          archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest())
            config["experiments"] = [{"id": "E0", "extractor": "image_size", "classifier": "logistic_regression", "c_values": [1.0]}]
            config["evaluation"] = {"figures": False}
            (workspace / "configs/project.json").write_text(json.dumps(config), encoding="utf-8")
            controls = {"WORKING_ROOT": str(workspace), "INSTALL_DEPENDENCIES": False,
                        "INSTALL_KAGGLE_API": False, "AUTO_BACKUP": False, "SHOW_PLOTS": False,
                        "BACKUP_DIR": str(base / "backups"), "ARCHIVE_PATH": str(archive_path),
                        "MAX_SESSION_MINUTES": None,
                        "EPOCH_SETTINGS": {"extractor": "image_size", "loss": "log_loss", "alpha": .0001,
                            "epochs": 3, "batch_size": 4, "patience": 5, "learning_rate": "optimal", "eta0": .01}}
            program = '''
import json, sys
from pathlib import Path
notebook = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
scope = {}
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    exec(''.join(cell['source']), scope)
    if cell['id'] == 'controls':
        scope.update(json.loads(sys.argv[2]))
assert scope['training_result']['status'] == 'completed'
assert not (Path(scope['repo_root']) / 'results/final_test.lock.json').exists()
from modules.checkpoints import save_backup, restore_backup
snapshot = save_backup(scope['repo_root'], sys.argv[3])
restored = restore_backup(snapshot, sys.argv[4])
Path(sys.argv[5]).write_text(json.dumps({'run':scope['RUN_ID'],'root':str(restored)}))
'''
            metadata = base / "smoke-result.json"
            first = subprocess.run([sys.executable, "-c", program, str(NOTEBOOK), json.dumps(controls),
                str(base / "portable-backup"), str(base / "restored"), str(metadata)], capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stdout[-3000:] + first.stderr[-5000:])
            record = json.loads(metadata.read_text())
            controls.update(WORKING_ROOT=record["root"], RUN_ID=record["run"], ARCHIVE_PATH=None,
                            PIN_RESTORE_LIBRARIES=False, EXPORT_REVIEW_ZIP=False)
            resumed = '''
import json, sys
from pathlib import Path
from unittest.mock import patch
notebook = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
scope = {}
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    if cell['id'] in {'data','train','final'}:
        with patch('modules.data.prepare_dataset',side_effect=AssertionError('no raw download')), \\
             patch('modules.features.extract_features',side_effect=AssertionError('no extraction')), \\
             patch('sklearn.linear_model.SGDClassifier.partial_fit',side_effect=AssertionError('no refit')):
            exec(''.join(cell['source']),scope)
    else:
        exec(''.join(cell['source']),scope)
    if cell['id']=='controls':
        scope.update(json.loads(sys.argv[2]))
assert scope['training_result']['status']=='completed'
assert scope['training_result']['epoch']==3
'''
            second = subprocess.run([sys.executable, "-c", resumed, str(NOTEBOOK), json.dumps(controls)], capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stdout[-3000:] + second.stderr[-5000:])
            controls.update(ACTION="final", FINAL_TEST=True)
            final_program = resumed.replace("assert scope['training_result']['status']=='completed'", "assert scope['final_result']['status']=='completed'").replace("assert scope['training_result']['epoch']==3", "assert scope['final_result']['engine']=='sgd_epoch'")
            final = subprocess.run([sys.executable, "-c", final_program, str(NOTEBOOK), json.dumps(controls)], capture_output=True, text=True)
            self.assertEqual(final.returncode, 0, final.stdout[-3000:] + final.stderr[-5000:])
