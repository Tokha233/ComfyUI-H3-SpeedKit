from pathlib import Path
import sys,json,hashlib,time
r=Path('/tmp/h3-kitchen-prs-0930');sys.path[:0]=[str(r/'combined'),str(r/'combined/tests'),str(r/'build-deps'),'/last/deps']
import torch
from test_h3_qkv import operands
from comfy_kitchen.h3_qkv import _op
records=[]
for m in [87142,200000]:
    args=list(operands(m));expected=_op(*args,1e-5)
    padded=torch.empty((m,5377),device='cuda',dtype=torch.int8);padded[:,:5376]=args[0];old=args[0];args[0]=padded[:,:5376]
    torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();actual=_op(*args,1e-5);torch.cuda.synchronize()
    row={'m':m,'seconds':time.perf_counter()-start,'peak_extra_bytes':torch.cuda.max_memory_allocated()-before,'peak_total_bytes':torch.cuda.max_memory_allocated(),'all_equal':all(torch.equal(a,b) for a,b in zip(actual,expected)),'input_unchanged':torch.equal(args[0],old)}
    assert row['all_equal'] and row['input_unchanged'];records.append(row);(r/'qkv-large-1001.json').write_text(json.dumps(records,indent=2));print(json.dumps(row),flush=True)
    del actual,expected,args,padded,old;torch.cuda.empty_cache()
