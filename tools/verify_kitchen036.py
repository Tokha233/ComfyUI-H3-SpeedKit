"""Recompute Kitchen .36 claims from anonymized formal-run records."""
import json
from pathlib import Path
from statistics import mean
root = Path(__file__).resolve().parents[1]
data = json.loads((root / 'evidence/kitchen036-comparison.json').read_text())
rows = data['records']; checks = 0
for arm in ('stock_fp16', 'stock_int8', 'speedkit_int8'):
    subset = [r for r in rows if r['arm'] == arm]
    assert len(subset) == 4
    for field in ('seconds', 'dit_seconds', 'video_seconds', 'audio_seconds', 'export_seconds'):
        assert abs(mean(r[field] for r in subset) - data['summary'][arm][field]) < 1e-9
        checks += 1
for case in (1, 8):
    subset = [r for r in rows if r['case'] == case]
    ref = next(r for r in subset if r['arm'] == 'stock_int8')['sha256']
    for r in subset:
        assert r['nfe'] == 8
        assert r['sha256'][:2] == ref[:2] and r['sha256'][3] == ref[3]
        if r['arm'] != 'stock_fp16':
            assert r['sha256'] == ref
        if r['arm'] == 'speedkit_int8':
            assert r['backend']['counts'] == {
                'qkv': 50, 'attention': 50, 'norm_quant': 100,
                'fc1': 50, 'gate': 100, 'last_block_window': 1,
            }
        checks += 1
s = data['summary']; b = s['stock_fp16']['seconds']; i = s['stock_int8']['seconds']; o = s['speedkit_int8']['seconds']
dit_baseline = s['stock_int8']['dit_seconds']
dit_optimized = s['speedkit_int8']['dit_seconds']
for k, value in [('latency_reduction_vs_fp16', 1-o/b), ('capacity_equivalent_vs_fp16', b/o-1),
                 ('latency_reduction_vs_int8', 1-o/i), ('dit_reduction', 1-dit_optimized/dit_baseline)]:
    assert abs(s[k] - value) < 1e-12
    checks += 1
print(json.dumps({'status': 'passed', 'checks': checks, 'latency_reduction_percent': (1-o/b)*100,
                  'dit_baseline_seconds': dit_baseline, 'dit_optimized_seconds': dit_optimized,
                  'dit_seconds_saved': dit_baseline-dit_optimized,
                  'dit_time_reduction_percent': (1-dit_optimized/dit_baseline)*100}))

for name in ('public-input-small.json', 'public-input-medium.json'):
    public = json.loads((root / 'evidence' / name).read_text())
    assert public['status'] == 'success'
    formal = [r for r in public['records'] if not r['warmup']]
    assert len(formal) == 4
    for record in public['records']:
        assert record['four_sha_equal_stock']
        if record['arm'] == 'speedkit':
            assert record['backend']['counts']['attention'] == 50
    print(json.dumps({'public_input': name, 'four_sha_equal': True,
        'stock_mean': mean(r['seconds'] for r in formal if r['arm']=='stock'),
        'speedkit_mean': mean(r['seconds'] for r in formal if r['arm']=='speedkit')}))
