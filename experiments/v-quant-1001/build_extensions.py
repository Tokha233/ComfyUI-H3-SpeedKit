from pathlib import Path
import os,subprocess,shlex,shutil,json
r=Path('/tmp/h3-kitchen-followup-1001');dr=Path('/tmp/h3-kitchen-prs-0930');build=dr/'build-combined-clean';repo=dr/'combined';env=os.environ.copy();env.update(NVCC_PREPEND_FLAGS='--cudadevrt=none',LIBRARY_PATH='/legacy/toolchain/cuda/lib',PYTHONPATH=str(dr/'build-deps'),PATH=str(dr/'build-deps/bin')+':/legacy/toolchain/cuda/bin:'+env['PATH'])
commands=subprocess.check_output(['ninja','-C',str(build),'-t','commands'],env=env,text=True).splitlines()
command=next(x for x in commands if ' -c '+str(repo/'comfy_kitchen/backends/cuda/sage_attention/quant_v_int8.cu') in x)
pin=Path('/tmp/h3-pdmd-gate-1001/d64-neutral-build.json');base_link=json.loads(pin.read_text())['link']
record={}
for arm in ['base','candidate']:
 src=r/('quant_v_base.cu' if arm=='base' else 'quant_v_candidate.cu');obj=r/(arm+'.o');args=shlex.split(command);args[args.index('-o')+1]=str(obj);args[args.index('-c')+1]=str(src)
 if '-MF' in args:args[args.index('-MF')+1]=str(obj)+'.d'
 subprocess.run(args,cwd=build,env=env,check=True)
 link=[str(obj) if x.endswith('sage_attention/quant_v_int8.cu.o') else x for x in base_link];link[link.index('-o')+1]=str(r/(arm+'.so'))
 assert str(obj) in link
 subprocess.run(link,cwd=build,env=env,check=True)
 overlay=r/arm
 shutil.copytree('/tmp/h3-pdmd-gate-1001/main12389a3/comfy_kitchen',overlay/'comfy_kitchen',ignore=shutil.ignore_patterns('*.so','__pycache__'),dirs_exist_ok=True)
 shutil.copy2(r/(arm+'.so'),overlay/'comfy_kitchen/backends/cuda/_C.abi3.so')
 record[arm]=dict(compile=args,link=link)
(r/'extensions-build.json').write_text(json.dumps(record,indent=2))
print('BUILT',flush=True)
