from pathlib import Path
import os,subprocess,time,json
r=Path('/tmp/h3-kitchen-prs-0930')
while not (r/'build-negative-scale-r2.exit').exists(): time.sleep(10)
assert (r/'build-negative-scale-r2.exit').read_text().strip()=='0'
assert (r/'negative-scale/comfy_kitchen/backends/cuda/_C.abi3.so').exists()
def run(name,args,cwd=None,arm='negative-scale'):
 env=os.environ.copy();env['PYTHONPATH']=str(r/arm)+':'+str(r/'build-deps')+':/last/deps';env['OMP_NUM_THREADS']='4'
 with (r/(name+'.log')).open('w') as f:code=subprocess.call(args,env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
 (r/(name+'.exit')).write_text(str(code));print(name,code,flush=True)
 return code
files=['tests/test_int8.py','tests/test_int8_residual.py','tests/test_int8_input_act.py','tests/test_int8_attention.py','tests/test_rms_rope.py','tests/test_gemm_bias_contract.py']
assert run('nonpositive-regression',['python','-m','pytest','-q',*files],str(r/'negative-scale'))==0
for pair,arms in enumerate([['base','negative-scale'],['negative-scale','base']]):
 for arm in arms:assert run(f'nonpositive-probe-{arm}-{pair}',['python',str(r/'nonpositive-probe.py'),'--arm',arm,'--pair',str(pair)])==0
for arm in ['base','negative-scale']:
 assert run(f'nonpositive-sampler-{arm}-0',['python',str(r/'nonpositive-sampler.py'),'--arm',arm,'--pair','0','--repeats','2'])==0
print('ALL DONE',flush=True)
