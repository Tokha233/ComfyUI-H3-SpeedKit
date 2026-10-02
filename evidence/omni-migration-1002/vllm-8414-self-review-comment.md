Pre-submission self-check for `ce3039e4a48b6026302b6b9dec107dec8ac69fd9` (performed with Codex; no independent human review is claimed):

- Reviewed the complete three-file diff and the existing installer/dispatch contract. This adds only the SM120 entry, representative test shapes, and validation documentation. Existing numerical guards and fallback paths are preserved; no new quantization, pipeline, dependency or public API.
- Ran the repository's full precheck, including code quality, examples policy and simplification checks. No blocking findings or justified additional abstractions. All relevant local pre-commit hooks passed without allowlist changes; hosted pre-commit, Python 3.11/3.12 builds and DCO now pass too.
- Executed 69 relevant tests on RTX 5090 D v2. Official-weight full RGB8 decode and separate FP32 pre-conversion output are bit-exact, including three extra input/shape cases. The public reproduction script was independently rerun.
- Checked the performance scope and negative results: 6.202 -> 4.632 s is video VAE decode only; the small 195-row residual microbenchmark regresses. No serving-QPS, audio, full-generation or incremental gain over a deployed INT8 VAE is claimed.
- Confirmed merge compatibility with updated upstream `bbee488`; benchmark numbers retain their explicit `7b15d22` baseline.

[Reproduction and raw evidence](https://github.com/Tokha233/ComfyUI-H3-SpeedKit/tree/a3a62e9/evidence/omni-migration-1002).

@vllm-omni-review-bot please review the submitted diff.
