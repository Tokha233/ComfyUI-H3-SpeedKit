"""Recompute stock035/R85 paired means and compare output to historical gold."""
from pathlib import Path
from statistics import mean
import json,math
R=Path(__file__).resolve().parents[1]
d=json.loads((R/'evidence/stock035-vs-r85.json').read_text())
gold=json.loads((R/'evidence/measurements.json').read_text())['r85']['historical_gold']
assert d['status']=='complete' and len(d['measurements'])==d['n_formal']==8
assert d['n_warmup']==8 and len(d['cases'])==2
checks=0
def close(x,y):
 global checks
 assert math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-8),(x,y)
 checks+=1
for c in d['cases']:
 for mode in ('stock','r85'):
  rows=[r for r in d['measurements'] if r['case']==c['case'] and r['mode']==mode]
  assert len(rows)==c['n'][mode]==2
  for r in rows:
   assert r['signature']==gold[str(r['case'])]['signature']
   assert r['nfe']==8 and r['attention_calls']==400
  for metric,stats in c['arms'][mode].items():
   values=[r[metric] for r in rows]
   close(mean(values),stats['mean']);close(min(values),stats['min']);close(max(values),stats['max'])
 for metric,percent in c['reduction_percent'].items():
  close(100*(1-c['arms']['r85'][metric]['mean']/c['arms']['stock'][metric]['mean']),percent)
for mode,values in d['matched_work_sums'].items():
 for metric,value in values.items():close(sum(c['arms'][mode][metric]['mean'] for c in d['cases']),value)
x=d['matched_work_sums'];a=x['stock']['total_seconds'];b=x['r85']['total_seconds']
close(100*(1-b/a),d['total_reduction_percent'])
close(100*(a/b-1),d['capacity_increase_percent'])
close(100*(1-x['r85']['dit_seconds']/x['stock']['dit_seconds']),d['dit_reduction_percent'])
close((a-b)/2,d['saved_seconds_per_request'])
print(json.dumps(dict(status='verified',checks=checks,formal_requests=8,historical_four_sha_equal=True,
   total_reduction_percent=d['total_reduction_percent'],capacity_increase_percent=d['capacity_increase_percent']),ensure_ascii=False))
