"""Run the shared pipeline without notebook-specific paths or hidden state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.config import load_config
from modules.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "eda", "extract", "train", "final", "all"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/project.json")
    parser.add_argument("--archive", type=Path, help="use an existing archive; still verify its size and SHA-256")
    parser.add_argument("--final-test", action="store_true", help="with all: evaluate test after validation selection")
    parser.add_argument("--verbose", action="store_true", help="print full per-class summaries and artifact metadata")
    parser.add_argument("--resume-run", help="explicit validation run ID; reuse verified completed trials")
    parser.add_argument("--backup-dir", type=Path, help="external mounted directory for checkpoint backups")
    args = parser.parse_args()
    if args.final_test and args.stage != "all":
        parser.error("--final-test is only used with all; the final stage is already explicit")
    result = run_pipeline(ROOT, load_config(args.config), stage=args.stage,
                          archive_path=args.archive, final_test=args.final_test,
                          resume_run=args.resume_run, backup_dir=args.backup_dir)
    shown = result if args.verbose else {key: value for key, value in result.items() if key != "dataset"}
    if not args.verbose:
        shown["dataset"] = {key: value for key, value in result["dataset"].items() if key != "summary"}
        shown["split_counts"] = result["dataset"]["summary"]["split_counts"]
        if "eda" in shown:
            shown["eda"] = {key: value for key, value in shown["eda"].items() if key != "summary"}
    print(json.dumps(shown, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
