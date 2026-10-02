The GPU workflows stopped at `Require run-ci label`: the job log reports `Missing required label 'run-ci'`. Could a maintainer authorize the relevant Diffusion CI?

Local verification on RTX 5090 D v2: 14 policy tests; 30 exact matrix pairs with stock Kitchen 0.2.36; a second run of the checked-in benchmark; and full Larry8 sampler ABBA with all video/audio latent hashes identical. Pre-commit passes. The PR explicitly reports the sampler timing variance and does not claim stable serving throughput improvement.
