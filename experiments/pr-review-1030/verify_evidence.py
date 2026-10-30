"""Verify archived H3 VAE ABBA numbers and original artifact identities."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'evidence/pr-review-1030'
identities = json.loads((EVIDENCE / 'artifact-sha256.json').read_text())
for name, expected in identities.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name

order = [(arm, i, repeat)
         for i, arm in enumerate(('baseline', 'candidate', 'candidate', 'baseline'))
         for repeat in (-1, 0, 1)]
summary = {}
reports = []
for mode in ('rgb8', 'fp32'):
    data = json.loads((EVIDENCE / f'decode-{mode}.json').read_text())
    reports.append(data)
    rows = data['records']
    assert data['complete'] and not data['source_mismatches']
    assert [(r['arm'], r['arm_index'], r['repeat']) for r in rows] == order
    assert all(r['warmup'] == (r['repeat'] == -1) for r in rows)
    assert all(r['exact'] and r['max_abs'] == 0 for r in rows)
    assert len({r['sha256'] for r in rows}) == 1
    assert all(r['shape'] == [1, 3, 124, 512, 768] for r in rows)
    expected_dtype = 'torch.uint8' if mode == 'rgb8' else 'torch.float32'
    assert all(r['dtype'] == expected_dtype for r in rows)
    medians = {}
    peaks = {}
    for arm in ('baseline', 'candidate'):
        timed = [r for r in rows if r['arm'] == arm and not r['warmup']]
        assert len(timed) == 4
        medians[arm] = statistics.median(r['ms'] for r in timed)
        peaks[arm] = statistics.median(r['peak_memory_allocated_bytes'] for r in timed)
    reduction = 100 * (1 - medians['candidate'] / medians['baseline'])
    assert medians == data['medians_ms']
    assert peaks == data['median_peak_bytes']
    assert reduction == data['latency_reduction_percent']
    summary[mode] = {'medians_ms': medians, 'latency_reduction_percent': reduction,
                     'outputs': len(rows), 'sha256': rows[0]['sha256']}

assert reports[0]['latent_sha256'] == reports[1]['latent_sha256']
assert reports[0]['vae_files'] == reports[1]['vae_files']
metadata = json.loads((EVIDENCE / 'official-vae-source-metadata.json').read_text())
weight_sha = next(f['lfs']['oid'] for f in metadata if f['path'].endswith('/model.safetensors'))
assert reports[0]['vae_files']['source/model.safetensors'] == weight_sha
source_audit = json.loads((EVIDENCE / 'source-audit.json').read_text())
assert source_audit == {'files': 4596, 'mismatches': []}
assert '69 passed' in (EVIDENCE / 'pytest.log').read_text()
print(json.dumps({'status': 'passed', 'results': summary}, indent=2))
