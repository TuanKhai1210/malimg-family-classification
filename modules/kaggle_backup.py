"""Private Kaggle Dataset checkpoint versions with downloaded-byte readback.

Local /kaggle/working is session storage. Only a verified remote receipt means
this module has read back the published recovery payload. No credentials enter
the payload, and an upload/readback failure stops the caller before another epoch.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

from .checkpoints import _safe_name, _sha, save_backup


RECOVERY_FILE = "recovery.bin"
RECOVERY_MAGIC = b"MALIMG_RECOVERY_V1\n"


def _recovery_file(receipt):
    """Allow only our two transport formats, never an arbitrary receipt path."""
    schema = receipt.get("schema")
    if type(schema) is not int or schema not in {1, 2}:
        raise ValueError("Unsupported remote recovery receipt schema")
    expected = "recovery.zip" if schema == 1 else RECOVERY_FILE
    if receipt.get("recovery_file", expected if schema == 1 else None) != expected:
        raise ValueError("Invalid remote recovery filename")
    return expected


def validate_dataset_ref(reference):
    if not isinstance(reference, str) or not re.fullmatch(r"[A-Za-z0-9_-]+/[a-z0-9][a-z0-9-]{5,49}", reference):
        raise ValueError("BACKUP_DATASET must be your Kaggle username/private-dataset-slug")
    if reference.split("/")[0].lower() in {"yourname", "your-username", "username"}:
        raise ValueError("Replace the BACKUP_DATASET example with your own Kaggle username")
    return reference


def authenticated_api():
    """Read Kaggle Secrets/environment; never print or write keys to a workspace."""
    import os
    if not (os.environ.get("KAGGLE_API_TOKEN") or
            (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))):
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
            os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")
        except Exception:
            raise RuntimeError("Add and enable Kaggle Secrets KAGGLE_USERNAME and KAGGLE_KEY; never paste keys into cells") from None
    try:
        # Kaggle's top-level import can authenticate automatically; suppress its output.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
        return api
    except Exception:
        raise RuntimeError("Kaggle authentication failed; inspect Secrets and the installed kaggle package") from None


class KaggleDatasetBackup:
    """One writer per dedicated private dataset; versions never delete old versions.

    Each version contains recovery.bin (framed ZIP bytes) and latest.json. Kaggle
    expands uploaded archives, so neither the suffix nor the prefix is ZIP-like.
    The bundle is uploaded/read back each time: account for network,
    version/storage limits and keep epochs meaningful rather than tiny batches.
    """

    def __init__(self, root, destination, dataset_ref, *, config=None, api=None,
                 poll_timeout=300, poll_interval=5, sleep=time.sleep):
        self.root = Path(root).resolve()
        self.destination = Path(destination).resolve()
        self.reference = validate_dataset_ref(dataset_ref)
        self.config = config
        self.api = api if api is not None else authenticated_api()
        self.poll_timeout, self.poll_interval, self.sleep = poll_timeout, poll_interval, sleep
        self.preflight()

    def _metadata(self):
        with tempfile.TemporaryDirectory(prefix="malimg-kaggle-metadata-") as folder:
            with contextlib.redirect_stdout(io.StringIO()):
                path = self.api.dataset_metadata(self.reference, folder)
            metadata = json.loads(Path(path).read_text(encoding="utf-8"))
        info = metadata.get("info", metadata)
        private = info.get("isPrivate", info.get("is_private"))
        if private is not True:
            raise ValueError("Backup dataset is public or privacy could not be verified; use a dedicated private dataset")
        return info

    def preflight(self):
        try:
            self._metadata()
            return True
        except Exception as error:
            status = getattr(error, "status", None)
            if status is None:
                status = getattr(getattr(error, "response", None), "status_code", None)
            if status == 404:
                return False  # first checkpoint creates a private dataset explicitly
            if isinstance(error, ValueError):
                raise
            raise RuntimeError("Cannot verify private backup access; check Internet, dataset ref, and credentials") from None

    def _ready(self):
        steps = max(1, int(self.poll_timeout / max(self.poll_interval, 0.01)))
        for _ in range(steps):
            with contextlib.redirect_stdout(io.StringIO()):
                status = self.api.dataset_status(self.reference)
            if str(status).lower() == "ready":
                return
            if str(status).lower() in {"error", "failed"}:
                raise RuntimeError("Kaggle dataset version failed processing")
            self.sleep(self.poll_interval)
        raise TimeoutError("Kaggle version is not ready; local checkpoint remains saved")

    def __call__(self, label, run_dir=None):
        """Publish, wait, download, verify, then and only then mark remote success."""
        exists = self.preflight()
        if exists:
            self._ready()
        snapshot = save_backup(self.root, self.destination, label=label, config=self.config)
        with zipfile.ZipFile(snapshot) as archive:
            record = json.loads(archive.read("BACKUP.json"))
        features = self.destination / _safe_name(record["feature_bundle"])
        receipt = {"schema": 2, "recovery_file": RECOVERY_FILE, "checkpoint_id": uuid.uuid4().hex,
                   "dataset": self.reference, "label": label,
                   "run_id": run_dir.name if run_dir else None,
                   "snapshot": "snapshots/" + snapshot.name,
                   "snapshot_sha256": _sha(snapshot),
                   "feature_bundle": record["feature_bundle"],
                   "feature_bundle_sha256": record["feature_bundle_sha256"]}
        try:
            with tempfile.TemporaryDirectory(prefix="malimg-kaggle-upload-") as folder:
                folder = Path(folder)
                payload = folder / RECOVERY_FILE
                # The prefix prevents archive sniffing as well as suffix-based
                # expansion. ZipFile supports a leading preamble and still
                # verifies the same nested snapshot/feature archive bytes.
                with payload.open("wb") as stream:
                    stream.write(RECOVERY_MAGIC)
                    with zipfile.ZipFile(stream, "w", zipfile.ZIP_STORED) as archive:
                        archive.write(snapshot, receipt["snapshot"])
                        archive.write(features, receipt["feature_bundle"])
                receipt["recovery_sha256"] = _sha(payload)
                (folder / "latest.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                (folder / "dataset-metadata.json").write_text(json.dumps({
                    "id": self.reference, "title": "MalImg private experiment checkpoints",
                    "licenses": [{"name": "other"}],
                    "description": "Private own-group recovery artifacts. No raw malware binaries or images. "
                                   "Original dataset and pretrained-model terms remain applicable; no public redistribution grant.",
                }), encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    if exists:
                        response = self.api.dataset_create_version(str(folder), version_notes=f"{label} {receipt['checkpoint_id']}",
                            quiet=True, convert_to_csv=False, delete_old_versions=False)
                    else:
                        response = self.api.dataset_create_new(str(folder), public=False, quiet=True, convert_to_csv=False)
                if response is None or getattr(response, "error", None) or str(getattr(response, "status", "")).lower() != "ok":
                    raise RuntimeError("Kaggle did not accept the backup version")
                self._ready()
                self._metadata()  # verify private state after publication, too
                checked = folder / "readback"
                checked.mkdir()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.api.dataset_download_file(self.reference, "latest.json", path=str(checked), force=True, quiet=True, unzip=True)
                    self.api.dataset_download_file(self.reference, RECOVERY_FILE, path=str(checked), force=True, quiet=True, unzip=True)
                if json.loads((checked / "latest.json").read_text(encoding="utf-8")) != receipt:
                    raise ValueError("Remote checkpoint changed or another writer replaced this dataset version")
                if _sha(checked / RECOVERY_FILE) != receipt["recovery_sha256"]:
                    raise ValueError("Remote recovery payload checksum mismatch")
            receipt["remote_verified"] = True
            from .experiments import _write_json
            self.destination.mkdir(parents=True, exist_ok=True)
            _write_json(self.destination / "remote_receipt.json", receipt)
            print(f"REMOTE VERIFIED: {self.reference}; {label}; run={receipt['run_id']}", flush=True)
            return receipt
        except Exception as error:
            # Do not include API response bodies/headers/credentials in public output.
            raise RuntimeError(f"Remote checkpoint was NOT verified ({type(error).__name__}). "
                               f"Local snapshot is saved at {snapshot}; training stops before the next epoch") from None

    def download(self, destination):
        """Read the latest private recovery version; do not start a training job."""
        if not self.preflight():
            raise FileNotFoundError("Private backup dataset does not exist yet")
        self._ready()
        destination = Path(destination).resolve()
        destination.mkdir(parents=True, exist_ok=False)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.api.dataset_download_file(self.reference, "latest.json", path=str(destination), force=True, quiet=True, unzip=True)
        receipt = json.loads((destination / "latest.json").read_text(encoding="utf-8"))
        if receipt.get("dataset") != self.reference:
            raise ValueError("Downloaded private backup dataset identity mismatch")
        filename = _recovery_file(receipt)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.api.dataset_download_file(self.reference, filename, path=str(destination), force=True, quiet=True, unzip=True)
        if _sha(destination / filename) != receipt.get("recovery_sha256"):
            raise ValueError("Downloaded private backup identity/checksum mismatch")
        return unpack_recovery(destination / filename, destination / "verified", receipt=receipt)


def unpack_recovery(payload, destination, *, receipt=None):
    """Extract exactly the two backup ZIPs, not arbitrary uploaded file paths."""
    payload = Path(payload)
    with payload.open("rb") as stream:
        framed = stream.read(len(RECOVERY_MAGIC)) == RECOVERY_MAGIC
    if receipt is not None:
        filename = _recovery_file(receipt)
        if filename == RECOVERY_FILE and not framed:
            raise ValueError("Recovery payload is missing its transport header")
    if payload.suffix == ".bin" and not framed:
        raise ValueError("Recovery payload is missing its transport header")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(payload) as archive:
        names = archive.namelist()
        if len(names) != 2 or len(set(names)) != 2:
            raise ValueError("Recovery payload must contain one snapshot and one feature bundle")
        snapshots = [n for n in names if _safe_name(n).startswith("snapshots/") and n.endswith(".zip")]
        features = [n for n in names if _safe_name(n).startswith("features/") and n.endswith(".zip")]
        if len(snapshots) != 1 or len(features) != 1:
            raise ValueError("Recovery payload layout is invalid")
        for name in names:
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Recovery path escapes its destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
        if receipt:
            if snapshots[0] != receipt["snapshot"] or features[0] != receipt["feature_bundle"]:
                raise ValueError("Recovery members differ from receipt")
            for name, key in ((snapshots[0], "snapshot_sha256"), (features[0], "feature_bundle_sha256")):
                if _sha(destination / name) != receipt[key]:
                    raise ValueError("Recovery member checksum mismatch")
    return destination / snapshots[0]
