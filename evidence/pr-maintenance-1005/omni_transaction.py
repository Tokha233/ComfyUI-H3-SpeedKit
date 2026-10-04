import sys,time,json,pathlib,statistics,asyncio,importlib.util
sys.path.insert(0,'/tmp/h3-pr-fixes-1005/sglang-omni')
import torch
from sglang_omni.comm import stage_io
from sglang_omni.proto.request import StagePayload,OmniRequest
from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.relay.shm import ShmRelay
root=pathlib.Path('/tmp/h3-pr-fixes-1005')
spec=importlib.util.spec_from_file_location('baseline_stage_io',root/'stage_io_baseline.py');baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
functions={'baseline':baseline.pack_tensors,'candidate':stage_io.pack_tensors}
async def run():
 torch.set_num_threads(4);torch.manual_seed(19)
 report={'scope':'actual stage_io GPU/CPU pack + SHM write/read/ack + device restoration; no model compute','torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'records':[]}
 for source in ('cpu','cuda'):
  for count,size in ((1,132*5120),(4,1024*5120),(16,1024*5120)):
   tensors={str(i):torch.randn(size,device=source,dtype=torch.bfloat16) for i in range(count)}
   tensors['prefix']=torch.tensor([1,2,3],dtype=torch.uint8);tensors['tail']=torch.tensor([1.25],dtype=torch.float64)
   a,ma=functions['baseline'](tensors,device='cpu');b,mb=functions['candidate'](tensors,device='cpu');assert torch.equal(a,b) and ma==mb
   payload=StagePayload(request_id='bench',request=OmniRequest(inputs='test'),data=tensors)
   relay=ShmRelay('bench',device='cpu',credits=2)
   async def transfer():
    ref,op=await stage_io.write_payload(relay,'bench',payload,transport=TransportKind.SHM)
    out=await stage_io.read_payload(relay,'bench',ref,local_device='cuda:0')
    op.mark_receiver_done();await op.wait_for_completion();return out
   measurements=[]
   for arm in ('baseline','candidate','candidate','baseline'):
    stage_io.pack_tensors=functions[arm]
    for _ in range(3):await transfer()
    times=[]
    for _ in range(15):
     torch.cuda.synchronize();begin=time.perf_counter();out=await transfer();torch.cuda.synchronize();times.append(1000*(time.perf_counter()-begin))
    assert all(torch.equal(t,out.data[k]) and t.device==out.data[k].device for k,t in tensors.items())
    measurements.append({'arm':arm,'median_ms':statistics.median(times),'times_ms':times})
   relay.close();entry={'source':source,'count':count+2,'bytes':a.numel(),'bit_exact':True,'measurements':measurements};report['records'].append(entry);(root/'omni-transaction.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in entry.items() if k!='measurements'}),[x['median_ms'] for x in measurements],flush=True)
asyncio.run(run())
