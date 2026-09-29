"""Build the source distribution's SM120 kernels without changing PyTorch."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cutlass", type=Path, required=True, help="CUTLASS checkout containing include/")
    parser.add_argument("--output", type=Path, default=ROOT / "h3_speedkit/_native")
    args = parser.parse_args()
    nvcc = shutil.which(args.nvcc)
    if not nvcc:
        parser.error("CUDA Toolkit nvcc is missing; install CUDA 13 and retry")
    cutlass = args.cutlass.resolve() / "include"
    if not (cutlass / "cutlass/cutlass.h").is_file():
        parser.error("--cutlass must point to the root of a CUTLASS source checkout")
    args.output.mkdir(parents=True, exist_ok=True)
    common = ["-std=c++20", "-O3", "-DNDEBUG", "--expt-relaxed-constexpr",
              "--expt-extended-lambda", "--shared", "--cudart=shared", "--cudadevrt=none",
              "-Xcompiler=-fPIC", "-Xptxas=-v"]
    specs = [
        ("dense", "dense/dense_ring.cu", ["--use_fast_math", "-lineinfo", "--maxrregcount=168",
         "-gencode=arch=compute_120a,code=sm_120a", "-DH3_VARIANT_ID=8102", "-lcuda"]),
        ("norm", "norm_quant/norm_quant.cu", ["--use_fast_math"]),
        ("residual", "residual_quant/residual_quant.cu", ["--use_fast_math"]),
        ("convrot", "convrot/convrot_register_v2.cu", ["--use_fast_math"]),
        ("qkv", "gemm/r84_kfourwarp.cu", ["-DCOMFY_HAVE_CUTLASS", "-DH3_EVT_STAGES=2"]),
        ("sample", "gemm/r82_sample_konly.cu", ["-DCOMFY_HAVE_CUTLASS", "-DH3_EVT_STAGES=2"]),
        ("anchor", "gemm/r79_sample_anchor.cu", ["--use_fast_math"]),
        ("fc1", "gemm/fc1.cu", ["-DCOMFY_HAVE_CUTLASS", "-DH3_EVT_STAGES=2"]),
        ("gate", "gemm/gate.cu", ["-DCOMFY_HAVE_CUTLASS", "-DH3_EVT_STAGES=1"]),
    ]
    manifest = {"schema": 1, "kitchen": "0.2.36", "status": "building",
                "nvcc": subprocess.check_output([nvcc, "--version"], text=True), "kernels": {}}
    for name, relative, flags in specs:
        source = ROOT / "kernels" / relative
        target = args.output.resolve() / (name + ".so")
        architecture = [] if name == "dense" else ["-gencode=arch=compute_120,code=sm_120"]
        command = [nvcc, *common, *architecture, *flags,
                   "-I" + str(cutlass), "-I" + str(source.parent),
                   "-I" + str(source.parent / "vendor"), str(source), "-o", str(target)]
        print("Building", name, flush=True)
        result = subprocess.run(command, text=True, capture_output=True)
        (args.output / (name + ".log")).write_text(result.stdout + result.stderr)
        record = {"source": relative, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "exit_code": result.returncode, "command": command}
        if result.returncode == 0:
            record["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest["kernels"][name] = record
        (args.output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if result.returncode:
            print(result.stderr[-5000:], file=sys.stderr)
            return result.returncode
    manifest["status"] = "compiled; numerical validation required"
    (args.output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
