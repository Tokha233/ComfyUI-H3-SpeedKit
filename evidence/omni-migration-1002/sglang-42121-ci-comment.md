The failing gate jobs stop at `Require run-ci label`; the downstream finish jobs propagate that gate result. Could a maintainer authorize the appropriate diffusion CI with `run-ci`?

Local validation on the submitted commit: 28 passed, 1 skipped, 3 xfailed; real H3 INT8 safetensors loading and full Larry8 sampling complete on RTX 5090 D v2. Changed-file pre-commit passes. This PR fixes checkpoint loading only; it does not claim cross-framework output equivalence or a kernel speedup. [Test evidence](https://github.com/Tokha233/ComfyUI-H3-SpeedKit/tree/d6d6da7/evidence/omni-migration-1002).
