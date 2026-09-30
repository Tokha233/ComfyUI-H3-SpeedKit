from pathlib import Path
import subprocess,os,json
r=Path('/tmp/h3-kitchen-prs-0930');src=r/'qkv-quant/comfy_kitchen/backends/cuda/ops/h3_qkv'
env=os.environ.copy();env['LIBRARY_PATH']='/legacy/toolchain/cuda/lib'
report={}
for name,arch,cutlass in [('sm80_only','80',True),('no_cutlass','120',False)]:
    lib=r/f'qkv-probe-{name}.so'
    command=['/legacy/toolchain/cuda/bin/nvcc','-O3','-std=c++20','--use_fast_math','--expt-relaxed-constexpr','--expt-extended-lambda',f'-gencode=arch=compute_{arch},code=sm_{arch}','--shared','--cudart=shared','--cudadevrt=none','-Xcompiler=-fPIC','-Xlinker=-Bsymbolic','-I'+str(r/'cutlass/include')]
    if cutlass:command+=['-DCOMFY_HAVE_CUTLASS']
    command += [str(src/n) for n in ['qkv_quant.cu','sample_k.cu','anchor_v.cu']]+['-o',str(lib)]
    subprocess.run(command,env=env,check=True)
    code='import ctypes,torch,json; torch.cuda.init(); lib=ctypes.CDLL('+repr(str(lib))+'); fn=lib.h3_qkv_available; fn.restype=ctypes.c_int; value=fn(); assert value==0,value; x=torch.ones(32,device="cuda")+1; torch.cuda.synchronize(); assert x.sum().item()==64; print(json.dumps({"available":value,"subsequent_cuda_ok":True}))'
    output=subprocess.check_output(['python','-c',code],env=env,text=True);report[name]=json.loads(output)
    (r/'qkv-unavailable-probe-1001.json').write_text(json.dumps(report,indent=2));print(name,output,flush=True)
