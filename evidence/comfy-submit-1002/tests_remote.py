from pathlib import Path
import argparse
import sys

p = argparse.ArgumentParser()
p.add_argument('arm')
a = p.parse_args()
r = Path('/tmp/h3-comfy-submit-1002')
dr = Path('/tmp/h3-kitchen-prs-0930')
kitchen = 'gate' if a.arm == 'gate' else 'base'
sys.path[:0] = [str(r/a.arm), str(dr/kitchen), str(dr/'current-deps'), str(dr/'build-deps'), '/last/deps']
import pytest
import torch
torch.set_num_threads(4)
tests = ['tests-unit/comfy_test/test_minimax_h3_model.py']
if a.arm == 'gate':
    tests += ['tests-unit/comfy_quant/test_indexed_residual.py', 'tests-unit/comfy_quant/test_mixed_precision.py']
raise SystemExit(pytest.main(['-q', *[str(r/a.arm/t) for t in tests]]))
