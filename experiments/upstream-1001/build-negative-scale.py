from pathlib import Path
import os,subprocess,shutil,json
r=Path('/tmp/h3-kitchen-prs-0930');base=r/'base';build=r/'build-base-clean';env=os.environ.copy();env.update(NVCC_PREPEND_FLAGS='--cudadevrt=none',LIBRARY_PATH='/legacy/toolchain/cuda/lib',PYTHONPATH=str(r/'build-deps'),PATH=str(r/'build-deps/bin')+':/legacy/toolchain/cuda/bin:'+env['PATH'])
# Reuse independently compiled unchanged main objects, and compile every changed
# translation unit with the same CMake command, against its branch source.
commands=subprocess.check_output(['ninja','-C',str(build),'-t','commands'],env=env,text=True).splitlines()
for arm in ['negative-scale']:
 repo=r/arm
 for p in base.iterdir():
  if p.name not in ['build','comfy_kitchen','tests','third_party'] and p.is_file() and not (repo/p.name).exists():shutil.copy2(p,repo/p.name)
 (repo/'third_party').mkdir(exist_ok=True)
 for dep in ['cutlass','flash-attention']:
  if not (repo/'third_party'/dep).exists():(repo/'third_party'/dep).symlink_to(r/dep,target_is_directory=True)
 if arm=='attention':
  # Archive includes changed files only. Fill unchanged source from main.
  for p in (base/'comfy_kitchen').rglob('*'):
   target=repo/p.relative_to(base)
   if p.is_file() and not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  for p in (base/'tests').glob('*'):
   if p.is_file() and not (repo/'tests'/p.name).exists():shutil.copy2(p,repo/'tests'/p.name)
 outdir=r/('objects-'+arm);outdir.mkdir(exist_ok=True)
 changed=['sage_attention/sage_attn_launcher.cu']
 mapping={}
 for f in changed:
  template=f if f not in ['ops/cutlass_gemm_indexed_gate.cu','sage_attention/dense_tma_sm120.cu'] else 'ops/cutlass_gemm_int8.cu' if arm=='gate' else 'sage_attention/sage_attn_launcher.cu'
  command=next(c for c in commands if ' -c '+str(base/'comfy_kitchen/backends/cuda'/template) in c)
  # CMake flags, exact pinned CUTLASS, and actual branch Python bindings.
  command=command.replace(str(base),str(repo))
  import shlex
  args=shlex.split(command);dest=outdir/(Path(f).name+'.o')
  args[args.index('-o')+1]=str(dest)
  args[args.index('-c')+1]=str(repo/'comfy_kitchen/backends/cuda'/f)
  if '-MF' in args:args[args.index('-MF')+1]=str(dest)+'.d'
  if arm=='attention':
   args.append('-DCOMFY_HAVE_SM120_TMA')
   if f.endswith('dense_tma_sm120.cu'):
    args=[x.replace('compute_120,code=[sm_120]','compute_120a,code=[sm_120a]') for x in args];args.append('--maxrregcount=168')
  print('COMPILE',arm,f,flush=True);p=subprocess.run(args,cwd=build,env=env)
  if p.returncode:raise SystemExit(p.returncode)
  orig='CMakeFiles/_C.dir/dlpack_bindings.cpp.o' if f=='dlpack_bindings.cpp' else 'CMakeFiles/comfy_kitchen_cuda_kernels.dir/'+f+'.o'
  mapping[orig]=str(dest)
 link=next(c for c in commands if ' -shared ' in c and '_C.abi3.so' in c)
 # Remove CMake's shell prefix/suffix if present.
 if link.startswith(': && '):link=link[5:]
 if link.endswith(' && :'):link=link[:-5]
 for orig,new in mapping.items():
  if orig in link:link=link.replace(orig,new)
  else:link=link.replace(' -o ', ' '+new+' -o ',1)
 link=link.replace(str(base/'comfy_kitchen/backends/cuda/_C.abi3.so'),str(repo/'comfy_kitchen/backends/cuda/_C.abi3.so'))
 print('LINK',arm,flush=True);p=subprocess.run(link,cwd=build,env=env,shell=True)
 if p.returncode:raise SystemExit(p.returncode)
 (outdir/'build.json').write_text(json.dumps({'arm':arm,'base':'19ea55b9ebdaf77942dab36223e1222009d3ce11','changed':changed,'commands':mapping,'link':link},indent=2))
 print('BUILT',arm,flush=True)
