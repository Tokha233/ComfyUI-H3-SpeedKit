"""Recompute full A/B means, same-latent VAE quality, precision and signatures."""
from pathlib import Path
from statistics import mean
import json,math
P=Path(__file__).resolve().parents[1]
d=json.loads((P/'evidence/native-fp16-vs-r85.json').read_text())
q=json.loads((P/'evidence/native-fp16-vae-quality.json').read_text())
gold=json.loads((P/'evidence/measurements.json').read_text())['r85']['historical_gold']
checks=0
def close(a,b):
 global checks
 assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)
 checks+=1
assert d['status']==q['status']=='complete'
assert d['n_formal']==len(d['measurements'])==8 and d['n_warmup']==4
assert len(d['sources'])==4 and len(d['cases'])==2
assert d['video_vae_identities']['stock']['storage_dtypes']=={'F16':562}
assert d['video_vae_identities']['stock']['sha256']=='7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522'
assert d['video_vae_identities']['r85']['sha256']=='52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e'
assert q['weights']['A']==d['video_vae_identities']['stock']
assert q['weights']['B']==d['video_vae_identities']['r85']
for c in d['cases']:
 rows=[r for r in d['measurements'] if r['case']==c['case']]
 assert len({r['gpu_uuid'] for r in rows})==1
 for mode in ['stock','r85']:
  arm=[r for r in rows if r['mode']==mode];assert len(arm)==c['n'][mode]==2
  assert len({tuple(r['signature']) for r in arm})==1
  for r in arm:
   assert r['signature'][:2]==gold[str(r['case'])]['signature'][:2]
   assert r['signature'][3]==gold[str(r['case'])]['signature'][3]
   if mode=='r85':assert r['signature']==gold[str(r['case'])]['signature']
   assert r['nfe']==8 and r['attention_calls']==400
  for key,v in c['arms'][mode].items():
   close(v['mean'],mean(r[key] for r in arm));close(v['min'],min(r[key] for r in arm));close(v['max'],max(r[key] for r in arm))
 for key,v in c['reduction_percent'].items():close(v,100*(1-c['arms']['r85'][key]['mean']/c['arms']['stock'][key]['mean']))
for mode,sums in d['matched_work_sums'].items():
 for key,v in sums.items():close(v,sum(c['arms'][mode][key]['mean'] for c in d['cases']))
s=d['matched_work_sums']
for label,key in [('total','total_seconds'),('dit','dit_seconds'),('video','video_seconds')]:close(d[label+'_reduction_percent'],100*(1-s['r85'][key]/s['stock'][key]))
close(d['capacity_increase_percent'],100*(s['stock']['total_seconds']/s['r85']['total_seconds']-1))
close(d['saved_seconds_per_request'],(s['stock']['total_seconds']-s['r85']['total_seconds'])/2)
assert len(q['cases'])==q['pcm_equal_cases']==15 and {x['case'] for x in q['cases']}==set(range(1,16))
assert len(q['measurements'])==60
for x in q['cases']:
 assert x['pcm_equal'] and x['rgb_sha256']['B']==gold[str(x['case'])]['signature'][2]
 assert x['metrics']['audio']['spectral_cosine']==1.0 and x['metrics']['audio']['normalized_rmse']==0.0
 assert x['metrics']['reference_probe']['frames']==x['metrics']['candidate_probe']['frames']==x['raw']['frames']
 close(x['raw']['psnr_db'],10*math.log10(255**2/x['raw']['mse']))
 for arm in 'AB':
  rows=[r for r in q['measurements'] if r['case']==x['case'] and r['arm']==arm]
  assert len(rows)==x['timing'][arm]['n']==2
  assert all(r['rgb_sha256']==x['rgb_sha256'][arm] for r in rows)
  close(x['timing'][arm]['mean'],mean(r['seconds'] for r in rows))
 for r in d['measurements']:
  if r['case']==x['case']:assert r['signature'][2]==x['rgb_sha256'][{'stock':'A','r85':'B'}[r['mode']]]
for key,stats in q['raw_summary'].items():
 vals=[x['raw'][key] for x in q['cases']]
 close(stats['mean'],mean(vals));close(stats['min'],min(vals));close(stats['max'],max(vals))
for label,group,key in [('ssim','pixels','ssim'),('psnr','pixels','psnr_db'),('lpips','lpips','mean')]:
 vals=[x['metrics'][group][key] for x in q['cases']];stats=q['perceptual_summary'][label]
 close(stats['mean'],mean(vals));close(stats['min'],min(vals));close(stats['max'],max(vals))
for arm,v in q['decode_sums'].items():close(v,sum(x['timing'][arm]['mean'] for x in q['cases']))
close(q['decode_reduction_percent'],100*(1-q['decode_sums']['B']/q['decode_sums']['A']))
print(json.dumps(dict(status='verified',checks=checks,formal_full_requests=8,formal_decode=60,quality_clips=15,
 native_dtype='FP16',latents_pcm_equal=True,r85_rgb_historical_equal=True,total_reduction_percent=d['total_reduction_percent'],ssim=q['perceptual_summary']['ssim']['mean']),ensure_ascii=False))
