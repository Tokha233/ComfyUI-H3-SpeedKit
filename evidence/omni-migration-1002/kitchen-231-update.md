Addressed the benchmark reproducibility finding in `98f7fb9`: the report now links the archived sampler/attention harness, exact source revisions, compile/link commands and input hashes, and distinguishes the small V-quant smoke command from reproduction of the full measurements. The integration build's common objects and rebuilt attention object are identified separately.

The latest automated review reports no actionable comments for this update. Build Wheels is still waiting for external-contributor approval (`action_required`): https://github.com/Comfy-Org/comfy-kitchen/actions/runs/36928829866 . Could a maintainer approve that workflow so the submitted code can complete upstream build validation?

[Archived scripts and build commands](https://github.com/Tokha233/ComfyUI-H3-SpeedKit/tree/9c9093d/experiments/v-quant-1001). AI assistance: Codex prepared this follow-up after checking the published evidence and current workflow state.
