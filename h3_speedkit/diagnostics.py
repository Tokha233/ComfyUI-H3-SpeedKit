"""Read-only, JSON environment report; no installation or model mutation."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from . import __version__


def collect_environment(probe_cuda=False):
    versions = {}
    for name in ("torch", "triton", "comfy-kitchen", "comfyui-frontend-package"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {
        "speedkit_version": __version__,
        "status": "source_release",
        "acceleration_enabled": False,
        "gpu_backend_shipped": True,
        "native_build_present": (Path(__file__).parent / "_native/build.json").is_file(),
        "python": platform.python_version(),
        "os": platform.system(),
        "machine": platform.machine(),
        "packages": versions,
        "devices": [],
        "cuda_probe": "not_requested",
        "note": "This command does not enable acceleration. See configs/compatibility.json and actual per-request backend reports.",
    }
    if probe_cuda:
        try:
            import torch
            report["torch_cuda_build"] = torch.version.cuda
            report["cuda_probe"] = "available" if torch.cuda.is_available() else "unavailable"
            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    device = torch.cuda.get_device_properties(index)
                    report["devices"].append({
                        "index": index,
                        "name": device.name,
                        "compute_capability": [device.major, device.minor],
                        "memory_bytes": device.total_memory,
                        "sm120_candidate": (device.major, device.minor) == (12, 0),
                        "validated_by_this_package": False,
                    })
        except Exception as exc:
            report["cuda_probe"] = "failed"
            # Do not expose filesystem paths, hostnames or raw runtime errors.
            report["cuda_probe_error_type"] = type(exc).__name__
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-cuda", action="store_true", help="Query devices through existing PyTorch")
    args = parser.parse_args()
    print(json.dumps(collect_environment(args.probe_cuda), ensure_ascii=False, indent=2))
