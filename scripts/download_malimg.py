"""Download and verify the selected public MalImg archive for local inspection."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


URL = "https://www.dropbox.com/scl/fi/wdb6omeiu2lg796qvt9l7/malimg_dataset.zip?rlkey=63q2xqmtlm66gilf6idd2c9k7&dl=1"
EXPECTED_SIZE = 1_174_609_734
EXPECTED_SHA256 = "9766ae9f1daa520e367fb486ca94728fe1485c0f5cb8314c312d77089a1fe9ec"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "MalImg_original_dataset.zip"
CHUNK = 1024 * 1024


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    existing = DEST.stat().st_size if DEST.exists() else 0
    if existing > EXPECTED_SIZE:
        raise RuntimeError("Existing file is larger than expected; inspect it manually.")

    if existing < EXPECTED_SIZE:
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        request = urllib.request.Request(URL, headers=headers)
        with urllib.request.urlopen(request, timeout=90) as response:
            if existing and response.status != 206:
                raise RuntimeError("Server did not honor resume range.")
            with DEST.open("ab" if existing else "wb") as output:
                downloaded = existing
                next_report = ((downloaded // (128 * CHUNK)) + 1) * 128 * CHUNK
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    output.write(block)
                    downloaded += len(block)
                    if downloaded >= next_report:
                        print(f"downloaded={downloaded}/{EXPECTED_SIZE}", flush=True)
                        next_report += 128 * CHUNK

    actual_size = DEST.stat().st_size
    if actual_size != EXPECTED_SIZE:
        raise RuntimeError(f"Unexpected size: {actual_size} != {EXPECTED_SIZE}")

    sha256 = hashlib.sha256()
    with DEST.open("rb") as source:
        for block in iter(lambda: source.read(8 * CHUNK), b""):
            sha256.update(block)
    actual_sha256 = sha256.hexdigest()
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(f"Archive SHA-256 does not match the audited copy: {actual_sha256}")
    print(f"archive={DEST} size={actual_size} sha256={actual_sha256}", flush=True)


if __name__ == "__main__":
    main()
