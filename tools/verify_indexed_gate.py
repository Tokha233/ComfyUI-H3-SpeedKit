"""Recompute the public PR #219 consumer evidence; does not run GPU inference."""
from pathlib import Path
import hashlib
import json
import math
import statistics

root = Path(__file__).resolve().parents[1]
bundle = json.loads((root / 'evidence/indexed-gate-consumer.json').read_text())
for relative, expected in bundle['source_sha256'].items():
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
for run in bundle['runs']:
    assert run['status'] == 'success'
    rows = run['records']
    assert len(rows) == 10
    reference = rows[0]['sha256']
    assert len(reference) == 4 and all(len(x) == 64 for x in reference)
    for row in rows:
        assert row['sha256'] == reference and row['four_sha_equal_stock']
        if row['arm'] == 'indexed_gate':
            assert 'fallback' not in row['backend']
            assert row['backend']['counts'] == {'blocks': 50, 'outproj': 50, 'fc2': 50}
            if not row['warmup']:
                assert row['backend']['verification_blocks'] == 0
            if 'forward_counts' in row:
                c = row['forward_counts']
                assert c['forwards'] == 8
                assert c['blocks'] == c['outproj'] == c['fc2'] == 400
                assert c.get('fallback_forwards', 0) == 0
                if not row['warmup']:
                    assert c.get('verification_blocks', 0) == 0
    for key in ('seconds', 'dit_seconds'):
        means = {a: statistics.mean(r[key] for r in rows if r['arm'] == a and not r['warmup'])
                 for a in ('stock', 'indexed_gate')}
        reduction = 100 * (1 - means['indexed_gate'] / means['stock'])
        assert math.isclose(reduction, run['summary']['time_reduction_percent'][key], abs_tol=1e-9)
print(json.dumps({'status': 'passed', 'runs': len(bundle['runs']),
                  'formal_requests': 8 * len(bundle['runs']),
                  'scope': 'Offline evidence and source check, not a new performance measurement'}))
