"""Restore a trusted workspace before importing its source into a notebook."""

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path)
    source.add_argument("--payload", type=Path)
    source.add_argument("--dataset")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--upgrade-from", type=Path)
    args = parser.parse_args()
    from modules.kaggle_backup import KaggleDatasetBackup, unpack_recovery
    from modules.kaggle_workspace import restore_workspace
    with tempfile.TemporaryDirectory(prefix="malimg-restore-") as temp:
        if args.dataset:
            backup = KaggleDatasetBackup(ROOT, args.backup_dir or Path(temp) / "backups", args.dataset)
            snapshot = backup.download(Path(temp) / "download")
        elif args.payload:
            snapshot = unpack_recovery(args.payload, Path(temp) / "unpacked")
        else:
            snapshot = args.snapshot
        restore_workspace(snapshot, args.destination, upgrade_from=args.upgrade_from)


if __name__ == "__main__":
    main()
