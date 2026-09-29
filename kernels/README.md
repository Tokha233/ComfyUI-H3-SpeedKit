# SM120 source kernels

Build all nine libraries with `python tools/build_kernels.py --cutlass /path/to/cutlass`.
The tested CUDA Toolkit is 13.0, CUTLASS C++ v4.5.0, and GPU RTX 5090 D v2.

| Directory | Implementation |
|---|---|
| dense | INT8 QK/PV dense loop, three TMA slots, cached Q, BSHD output and live query window |
| gemm | INT8 QKV/RMS/RoPE/quant fusion; K-only sample; FC1 raster; outproj/FC2 gate epilogue |
| norm_quant | Native PyTorch norm reduction → modulation → ConvRot256 quantization |
| convrot | Register-shuffle ConvRot256 with exact rounding barriers |
| residual_quant | Historical residual/norm fusion source; inactive in the simplified default chain |

Every build writes compiler output, flags, source SHA and binary SHA. Python uses the
current CUDA stream, retains tensor owners and verifies the binary manifest. The
source release does not download binaries automatically. Do not apply SM120 launch
choices to other architectures without numerical and resource validation.

[Provenance](../evidence/kernel-provenance.json) · [Migration](../docs/MIGRATION-036.md)
