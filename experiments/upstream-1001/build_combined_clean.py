from pathlib import Path
import os
import subprocess

root = Path("/tmp/h3-kitchen-prs-0930")
repo = root / "combined"
env = os.environ.copy()
env.update(
    NVCC_PREPEND_FLAGS="--cudadevrt=none",
    LIBRARY_PATH="/legacy/toolchain/cuda/lib",
    PYTHONPATH=str(root / "build-deps"),
    PATH=str(root / "build-deps/bin") + ":/legacy/toolchain/cuda/bin:" + env["PATH"],
)
for name in ("cutlass", "flash-attention"):
    target = repo / "third_party" / name
    if target.is_dir() and not target.is_symlink() and not any(target.iterdir()):
        target.rmdir()
    if not target.exists():
        target.parent.mkdir(exist_ok=True)
        target.symlink_to(root / name, target_is_directory=True)
subprocess.run([
    "cmake", "-S", str(repo / "comfy_kitchen/backends/cuda"),
    "-B", str(root / "build-combined-clean"), "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release", "-DCOMFY_CUDA_ARCHS=120-real",
    "-DCMAKE_CUDA_COMPILER=/legacy/toolchain/cuda/bin/nvcc",
    "-DCUDAToolkit_ROOT=/legacy/toolchain/cuda",
    "-DPython_EXECUTABLE=/usr/local/bin/python",
], env=env, check=True)
subprocess.run([
    "cmake", "--build", str(root / "build-combined-clean"), "--parallel", "2"
], env=env, check=True)
