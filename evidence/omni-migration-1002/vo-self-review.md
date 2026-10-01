# vLLM-Omni full precheck — ce3039e

Base 7b15d22f48a4af2ba83f9a209944d0496348fc91; origin/main fetched again, unchanged.

- Title: [Hardware][NVIDIA] Enable exact H3 VAE operators on SM120.
- Scope: one flat dispatch entry, existing test parameter coverage, model-local README. No new pipeline, dependency, cache, endpoint, example or public API.
- Ownership: existing model-owned numerical contract and installer are retained, as requested in ops/README.md. No candidate simplification justified in this diff.
- Code quality: no added broad catches, Any, kwargs plumbing, copies in execution loops, synchronization, torch.cuda helpers, locks, or event-loop blocking.
- Accuracy: 69 relevant tests pass on actual SM120; full default RGB and separate preconversion FP32 output exact; three extra float-output shape/input cases exact. Actual numerical/performance evidence archived publicly.
- Performance: ABBA full decode, fixed model/input/autocast; 25.32% latency reduction. Parameterized public script independently exercised, 25.29%. Small residual-op regression disclosed. No E2E or concurrency-throughput claim.
- Lint: all relevant changed-file pre-commit gates pass, including markdownlint, mypy, SPDX, mark and forbidden imports. No SKIP or allowlist expansion.
- Examples: no new Python examples. Reproduction scripts and exact commands publicly archived in SpeedKit.
- Remote readiness: patch committed, public evidence present. vLLM-Omni fork/PR creation still requires restored authenticated GitHub access; no claim of upstream CI or human review.
