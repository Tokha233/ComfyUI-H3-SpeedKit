# Architecture

The implementation is model-scoped and separates compute, video output and service scheduling.

| Module | Responsibility |
|---|---|
| `__init__.py`, `h3_speedkit/nodes.py` | Five ComfyUI V3 nodes; CUDA imports on execution |
| `runtime.py` | Preflight, cloned model options, request context, exact INT8 block chain |
| `ops/` | CUDA/Triton tensor ownership, metadata checks and current-stream launch |
| `vendor/` | Attributed GPL H3 forward and VAE implementation |
| `video.py` | Old encoder / INT8 decoder loader, completed-blend RGB8, two-slot D2H |
| `export.py` | CPU PyAV MP4, atomic completion |
| `service.py` | Optional job/byte-bounded export queue, pending-job cancellation/drain/error handling |
| `tools/build_kernels.py` | Reproducible nine-library source build and SHA manifest |
| `benchmarks/run.py` | User-supplied public Ref2VA input, stock/optimized A/B, four signatures |

[Detailed data flow and invariants](MIGRATION-036.md) · [Installation](INSTALL.md)

CPU export overlap belongs to a service that owns GPU admission. Ordinary ComfyUI workflows return a completed MP4. They do not change queue scheduling globally. RGB8 uses a custom output type; workflows needing float IMAGE post-processing should use ordinary VAE Decode instead.
