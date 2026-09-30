from pathlib import Path
import os,subprocess,time
r=Path('/tmp/h3-kitchen-prs-0930')
while not (r/'build-combined-nonpositive.exit').exists():time.sleep(10)
assert (r/'build-combined-nonpositive.exit').read_text().strip()=='0'
env=os.environ.copy();env['PYTHONPATH']=str(r/'combined')+':'+str(r/'build-deps')+':/last/deps';env['OMP_NUM_THREADS']='4'
files=['test_int8','test_int8_residual','test_int8_input_act','test_int8_attention','test_rms_rope','test_gemm_bias_contract','test_convrot_register','test_int8_attention_tma','test_int8_attention_layout','test_int8_indexed_gate','test_indexed_norm','test_indexed_linear','test_h3_qkv']
with (r/'combined-nonpositive-regression.log').open('w') as f:code=subprocess.call(['python','-m','pytest','-q',*[f'tests/{f}.py' for f in files]],cwd=r/'combined',env=env,stdout=f,stderr=subprocess.STDOUT)
(r/'combined-nonpositive-regression.exit').write_text(str(code));assert code==0
with (r/'nonpositive-sampler-combined-6.log').open('w') as f:code=subprocess.call(['python',str(r/'nonpositive-sampler.py'),'--arm','combined','--pair','6','--repeats','2'],env=env,stdout=f,stderr=subprocess.STDOUT)
assert code==0
print('ALL DONE',flush=True)
