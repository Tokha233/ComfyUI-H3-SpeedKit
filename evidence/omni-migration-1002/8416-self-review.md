Self-review of `6925191e7f417a3c400b437039f95ab6d12cfcec`, with Codex assistance:

- Checked the entire diff and both consumers of `fill_timesteps` (request mode and batched packing). The four indices belong to an immutable request-local layout; all timestep values are still updated on every call.
- Checked edits, interleaved AV conditions, locked audio, empty indices, target-length validation and request/step-mode equality at 1/8/50 steps. 45 relevant tests pass.
- Ran all relevant local pre-commit gates, including mypy, markers, platform API, forbidden imports and SPDX. Hosted pre-commit, both wheel builds and DCO also pass.
- Independently ran the checked-in CUDA benchmark after the sampler finished. The separate profile observes `nonzero` 4→0 and stream synchronization 8→0; the isolated preparation time is ~0.229→0.029 ms for 32,320 packed rows. Full DiT or service throughput was not measured or claimed.
- Full precheck, simplification and examples-policy audit found no blocking issue. No new dependency, cross-request cache or model arithmetic change. Raw evidence and spread are linked in the description.

@vllm-omni-review-bot please review. This is agent-assisted author self-review, not independent human review.
