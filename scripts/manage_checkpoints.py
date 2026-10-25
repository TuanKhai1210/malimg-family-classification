"""Save or restore an own-group checkpoint on a mounted drive or local disk."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.checkpoints import restore_backup, save_backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser("save")
    save.add_argument("--root", type=Path, default=ROOT)
    save.add_argument("--destination", type=Path, required=True)
    save.add_argument("--label", default="manual")
    restore = commands.add_parser("restore")
    restore.add_argument("snapshot", type=Path)
    restore.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "save":
        save_backup(args.root, args.destination, label=args.label)
    else:
        restore_backup(args.snapshot, args.destination)


if __name__ == "__main__":
    main()
