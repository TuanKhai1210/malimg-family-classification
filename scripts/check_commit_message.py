"""Validate new commit subjects and pull request titles.

This checks the documented project convention, not the whole Conventional
Commits grammar. Existing published history is intentionally not rewritten.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


SUBJECT = re.compile(
    r"^(feat|fix|docs|test|refactor|perf|build|ci|chore)"
    r"\([a-z][a-z0-9-]*\)!?: [a-z][^\r\n]*$"
)


def validate_message(message: str) -> list[str]:
    lines = message.rstrip("\r\n").splitlines()
    subject = lines[0] if lines else ""
    errors: list[str] = []
    if len(subject) > 72:
        errors.append("subject exceeds 72 characters")
    if not SUBJECT.fullmatch(subject):
        errors.append("subject must be type(scope): lowercase imperative summary")
    elif len(subject.split(": ", 1)[1]) < 10:
        errors.append("summary must contain at least 10 characters")
    if subject.endswith("."):
        errors.append("subject must not end with a period")
    if len(lines) > 1 and lines[1].strip():
        errors.append("separate the subject and body with a blank line")
    if any(line.rstrip() != line for line in lines):
        errors.append("remove trailing whitespace")
    return errors


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--message", help="message or pull request title")
    source.add_argument("--file", type=Path, help="commit message file, for local hooks")
    source.add_argument("--rev", help="Git revision whose full commit message is checked")
    source.add_argument("--range", dest="commit_range", help="Git base..head range to check")
    args = parser.parse_args()

    if args.message is not None:
        messages = [("input", args.message)]
    elif args.file is not None:
        messages = [(str(args.file), args.file.read_text(encoding="utf-8"))]
    elif args.rev is not None:
        messages = [(args.rev, git("show", "-s", "--format=%B", args.rev))]
    else:
        commits = git("rev-list", "--reverse", "--no-merges", args.commit_range).splitlines()
        messages = [(sha[:12], git("show", "-s", "--format=%B", sha)) for sha in commits]

    failures = 0
    for source_name, message in messages:
        errors = validate_message(message)
        if errors:
            failures += 1
            print(f"{source_name}: {message.splitlines()[0] if message else '<empty>'}", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
    if failures:
        return 1
    print(f"Validated {len(messages)} message(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
