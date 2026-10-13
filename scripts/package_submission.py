"""Package code, notebook, report and generated artifacts without raw images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.submission import build_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/malimg_review_base.zip")
    parser.add_argument("--final", action="store_true", help="require actual final report, features and final-test artifacts")
    parser.add_argument("--without-features", action="store_true", help="create a lighter draft; cannot be used for final submission")
    args = parser.parse_args()
    print(build_submission(ROOT, args.output, draft=not args.final, include_features=not args.without_features))


if __name__ == "__main__":
    main()
