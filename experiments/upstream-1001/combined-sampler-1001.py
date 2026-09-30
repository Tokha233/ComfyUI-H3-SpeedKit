from pathlib import Path
import sys, json, time, hashlib, argparse, importlib.util
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['base','combined'],required=True);p.add_argument('--pair',type=int,default=0);p.add_argument('--repeats',type=int,default=2);a=p.parse_args()
r=Path('/tmp/h3-kitchen-prs-0930');sys.path[:0]=[str(r/a.arm),str(r/'comfy-current'),str(r/'current-deps'),str(r/'build-deps'),'/last/deps']
import comfy.options
comfy.options.enable_args_parsing();sys.argv=['combined-sampler','--disable-cuda-malloc']
import main, torch, nodes, comfy.sd, comfy.ops, comfy.model_management
import comfy_kitchen as ck
from comfy_kitchen.backends import cuda
from comfy_kitchen.tensor.base import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from comfy.ldm.minimax import model as h3
from comfy.ldm.modules.attention import attention_comfy_kitchen_int8
from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
assert str(r/a.arm) in ck.__file__
report={'arm':a.arm,'pair':a.pair,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'kitchen':ck.__file__,'cuda_extension':cuda._C.__file__,'comfy':'8cfe5e1ecb97512dea8deaac15e1228d7e6feeb1','counts':{},'records':[]}
counts={'norm':0,'qkv':0,'gate':0}
def save():(r/f'combined-sampler-{a.arm}-{a.pair}.json').write_text(json.dumps(report,indent=2))
def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def norm_input(x,norm,shift,scale,rows):
    weight=comfy.model_management.cast_to(norm.weight,device=x.device)
    out=ck.indexed_norm_convrot(x,weight,shift,scale,rows,eps=norm.eps)
    counts['norm']+=1
    return out

def project(linear,x,q,qs):
    with comfy.ops.CastBiasWeightContext(linear,x,offloadable=True,compute_dtype=x.dtype,want_requant=True) as (w,bias):
        assert isinstance(w,QuantizedTensor) and w._layout_cls=='TensorWiseINT8Layout' and bias is None
        w,ws=TensorWiseINT8Layout.get_plain_tensors(w)
        out=torch.empty((len(x),w.shape[0]),dtype=x.dtype,device=x.device)
        empty=torch.empty(0,dtype=x.dtype,device=x.device)
        assert cuda._C.cutlass_int8_dequant(*(cuda._wrap_for_dlpack(t) for t in (q,w,qs,ws.reshape(-1).contiguous(),empty,out)),2,torch.cuda.current_stream().cuda_stream)
        return out

def gate(linear,x,scale,rows,residual,act=None):
    with comfy.ops.CastBiasWeightContext(linear,x,offloadable=True,compute_dtype=x.dtype,want_requant=True) as (w,bias):
        assert isinstance(w,QuantizedTensor) and w._layout_cls=='TensorWiseINT8Layout' and bias is None
        w,ws=TensorWiseINT8Layout.get_plain_tensors(w)
        out=ck.int8_linear_indexed_gate(x,w,ws.reshape(-1).expand(w.shape[0]).contiguous(),scale.to(x.dtype).contiguous(),rows,residual,input_act=act)
    counts['gate']+=1
    return out

def fused(self,x,t_emb,mod_segments,rope_freqs,transformer_options={},attention=None):
    assert attention is None and not transformer_options.get('patches') and not transformer_options.get('patches_replace')
    sa,ca,ga,sm,cm,gm=self.adaln_proj(t_emb)
    rows=torch.empty(len(x),device=x.device,dtype=torch.int32)
    for start,stop,row in mod_segments:rows[start:stop]=row
    q,qs=norm_input(x,self.norm1,sa,ca,rows)
    with comfy.ops.CastBiasWeightContext(self.attn.qkv_proj,x,offloadable=True,compute_dtype=x.dtype,want_requant=True) as (w,bias):
        assert isinstance(w,QuantizedTensor) and w._layout_cls=='TensorWiseINT8Layout' and bias is None
        w,ws=TensorWiseINT8Layout.get_plain_tensors(w)
        qw=comfy.model_management.cast_to(self.attn.q_norm.weight,device=x.device)
        kw=comfy.model_management.cast_to(self.attn.k_norm.weight,device=x.device)
        packed=ck.h3_qkv_prequantize(q,w,qs.reshape(-1,1),ws.reshape(-1,1).contiguous(),rope_freqs,qw,kw,eps=self.attn.q_norm.eps)
    del q,qs
    counts['qkv']+=1
    out=ck.int8_attention_from_prequantized(packed,output_layout='BSHD').reshape(len(x),7168)
    del packed
    x=gate(self.attn.out_proj,out,ga,rows,x)
    del out
    q,qs=norm_input(x,self.norm2,sm,cm,rows)
    hidden=project(self.mlp.fc1,x,q,qs)
    del q,qs
    return gate(self.mlp.fc2,hidden,gm,rows,x,'swiglu')

torch.set_num_threads(4)
with torch.inference_mode():
    model=comfy.sd.load_diffusion_model('/study/speedkit036/public-model/larry_v4_int8_r2.safetensors')
    model=MiniMaxH3SigmaShift.execute(model,12.,4.)[0];model.set_model_optimized_attention(attention_comfy_kitchen_int8)
    dm=model.model.diffusion_model
    if a.arm=='combined':
        spec=importlib.util.spec_from_file_location('h3_embed_1001',r/'model-release-1001.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        dm._embed_and_pack=mod.MiniMaxH3Model._embed_and_pack.__get__(dm)
        dm._forward=mod.MiniMaxH3Model._forward.__get__(dm)
        h3.DiTBlock.forward=fused
    saved=torch.load('/study/speedkit036/public-run-r5/condition.pt',map_location='cpu',weights_only=False)
    positive,latent=saved['positive'],saved['latent'];negative=nodes.ConditioningZeroOut().zero_out(positive)[0]
    for repeat in range(-1,a.repeats):
        for k in counts:counts[k]=0
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        sample=nodes.common_ksampler(model,42,8,1.,'euler','beta',positive,negative,latent,denoise=1.)[0]
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        row={'repeat':repeat,'warmup':repeat<0,'dit_seconds':elapsed,'sha256':[sha(t) for t in sample['samples'].unbind()],'counts':dict(counts),'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        assert counts==({'norm':800,'qkv':400,'gate':800} if a.arm=='combined' else {'norm':0,'qkv':0,'gate':0})
        report['records'].append(row);save();print(json.dumps(row),flush=True)
        if repeat==a.repeats-1:torch.save([t.cpu() for t in sample['samples'].unbind()],r/f'combined-latents-{a.arm}.pt')
        del sample
    report['status']='passed' if len({tuple(row['sha256']) for row in report['records']})==1 else 'mismatch';save()
