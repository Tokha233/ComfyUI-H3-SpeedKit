"""Recompute published scalars from included evidence; no GPU or network required."""
from pathlib import Path
import collections, hashlib, json, math, statistics

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'evidence/measurements.json').read_text())
lora=json.loads((ROOT/'evidence/lora-quality.json').read_text())
checks=[]
def check(name,actual,expected,atol=1e-8):
 if not math.isclose(actual,expected,abs_tol=atol,rel_tol=1e-10):
  raise AssertionError((name,actual,expected))
 checks.append(dict(name=name,actual=actual,expected=expected))
def reduction(a,b): return 100*(1-b/a)
def gain(a,b): return 100*(a/b-1)

for r in data['original_v22']['comparisons']:
 check(f'original_{r["duration"]}_latency',reduction(r['service_before_seconds'],r['service_after_seconds']),r['service_reduction_pct'])
 check(f'original_{r["duration"]}_queue_rate',100*(r['throughput_after_jobs_per_hour']/r['throughput_before_jobs_per_hour']-1),r['throughput_gain_pct'])
for gpu,r in data['kitchen033_to035_same_card'].items():
 a=r['control']['mean']['total_gpu_pipeline_seconds'];b=r['candidate']['mean']['total_gpu_pipeline_seconds']
 check(f'kitchen_full_{gpu}',reduction(a,b),r['candidate']['latency_reduction_pct'])
r=data['int8_vae_0919'];a=statistics.mean(c['raw_rgb8_vs_trt']['trt_decode_seconds'] for c in r['case_data']);b=statistics.mean(c['video_decode_seconds'] for c in r['case_data'])
check('int8_0919_mean_a',a,r['trt_seconds']['mean']);check('int8_0919_mean_b',b,r['int8_seconds']['mean']);check('int8_0919_reduction',reduction(a,b),r['latency_reduction_pct'])
r=data['r129'];a=statistics.mean(c['trt_seconds'] for c in r['rows']);b=statistics.mean(c['int8_seconds'] for c in r['rows'])
check('int8_0926_mean_a',a,r['mean_trt_seconds']);check('int8_0926_mean_b',b,r['mean_int8_seconds']);check('int8_0926_reduction',reduction(a,b),r['reduction_pct'])
r=data['r85'];by=collections.defaultdict(lambda:collections.defaultdict(list));sigs={}
for g in r['raw_groups']:
 by[g['job']][g['arm']].append(g['seconds'])
 for v in g['requests']:
  assert v['dense_calls']==400
  sig=[*v['latent_sha256'],v['rgb_sha256'],v['pcm_sha256']]
  assert sig==r['historical_gold'][str(v['case'])]['signature']
  assert v['case'] not in sigs or sigs[v['case']]==sig
  sigs[v['case']]=sig
assert set(sigs)==set(range(1,16))
a=sum(statistics.mean(v['A']) for v in by.values());b=sum(statistics.mean(v['B']) for v in by.values())
check('r85_raw_group_sum_a',a,r['baseline_seconds']);check('r85_raw_group_sum_b',b,r['candidate_seconds'])
check('r85_reduction',reduction(a,b),r['reduction_pct']);check('r85_capacity_gain',gain(a,b),r['inverse_fixed_request_throughput_change_pct'])
r=data['r126'];check('r126_reduction',reduction(r['sum_baseline_group_seconds'],r['sum_r85_group_seconds']),r['reduction_pct'])
r=data['r207'];a=sum(c['base']['dual_vae_seconds'] for c in r['cases']);b=sum(c['rope']['dual_vae_seconds'] for c in r['cases'])
check('r207_dual_a',a,r['baseline_dual_sum']);check('r207_dual_b',b,r['candidate_dual_sum']);check('r207_dual_reduction',reduction(a,b),r['reduction_pct'])
video_r207=reduction(sum(c['base']['seconds'] for c in r['cases']),sum(c['rope']['seconds'] for c in r['cases']))
check('r207_video_rounded',video_r207,6.802,atol=.0005)
r=data['r234_r244']
for key in ('seconds','dual_vae_seconds'):
 a=sum(c['timing']['rope'][key] for c in r['cases']);b=sum(c['timing']['cached'][key] for c in r['cases'])
 check('r244_'+key+'_a',a,r['summary'][key]['control_sum']);check('r244_'+key+'_b',b,r['summary'][key]['candidate_sum']);check('r244_'+key+'_reduction',reduction(a,b),r['summary'][key]['reduction_pct'])
assert all(c['signatures_equal'] for c in r['cases'])
for row in data['torch214']['speed']:
 for key,values in row['samples'].items(): check(f'torch_{row["case"]}_{row["mode"]}_{key}',statistics.median(values),row['median'][key])
for cell in lora['groups']['Q1_early_five_stories']:
 assert len(cell['stories'])==5 and cell['segments']==15
 for key in ('ssim','psnr_db','lpips','waveform_corr','spectral_cosine','normalized_rmse'):
  check('Q1_'+cell['id']+'_'+key,statistics.mean(s[key] for s in cell['stories']),cell['mean'][key])
assert len(lora['groups']['Q2_kitchen_five_stories'])==31
result=dict(status='passed',arithmetic_checks=len(checks),r85_unique_cases=len(sigs),r85_formal_requests=sum(len(g['requests']) for g in data['r85']['raw_groups']),
 r207_video_reduction_pct=video_r207,scope='Offline evidence recomputation; not new GPU inference, independent media re-evaluation, or a current production audit.',checks=checks)
(ROOT/'evidence/offline-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))
