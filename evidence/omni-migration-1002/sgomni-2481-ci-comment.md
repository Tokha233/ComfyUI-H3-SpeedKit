The initial GitHub workflows are waiting for approval (`action_required`). Could a maintainer approve the external-contributor workflows and authorize the appropriate shared-memory/payload CI with `run-ci`?

The submitted commit `9897501` already passed 32 relevant tests (1 skipped) on the actual CUDA environment and full-repository pre-commit. The PR includes exact round-trip checks, backpressure/cancellation coverage, and ABBA transaction measurements; it does not claim model inference or serving throughput gains.
