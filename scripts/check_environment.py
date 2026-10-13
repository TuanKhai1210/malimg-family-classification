"""Report the environment before extraction or training."""

from __future__ import annotations

import importlib
import json
import platform
import sys


def main() -> int:
    versions = {}
    missing = []
    for package in ("numpy", "PIL", "sklearn", "matplotlib", "torch", "torchvision", "skimage", "joblib"):
        try:
            module = importlib.import_module(package)
            versions[package] = module.__version__
        except (ImportError, RuntimeError, OSError) as error:
            missing.append(f"{package}: {error}")
    output = {"python": sys.version, "platform": platform.platform(), "versions": versions, "errors": missing}
    if "torch" in versions:
        import torch
        output["cuda_available"] = torch.cuda.is_available()
        output["cuda_version"] = torch.version.cuda
        output["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(json.dumps(output, indent=2))
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
