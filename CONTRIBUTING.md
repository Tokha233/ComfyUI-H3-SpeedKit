# Contributing

Start with [installation](docs/INSTALL.md), [migration invariants](docs/MIGRATION-036.md) and [licensing](docs/LICENSING.md).

For a kernel change, include exact source/build identities, target GPU, dimensions, warmups, repeated timings and output comparisons. Report stage latency separately from full request completion; do not add overlapping ablations. Preserve BF16 barriers, quantization scales, KV traversal and stream ownership. A CUDA error must abort, not silently rerun on modified state.

For new GPU or Torch support, contribute a reproducible benchmark and compatibility record. Do not remove guards solely to make a workflow run. Unknown shapes may use the first-use exact comparison; disclose its cost and result. Never include private prompts, reference media, model weights or access tokens in issues or pull requests.

Run `python -m unittest discover -s tests -v`, the metric verifiers and `python tools/check_repository.py`. These CPU checks do not substitute for GPU validation. Proposed code is contributed under the applicable file license.
