"""Finalized temporal chunk only: FP32 mul, add, clamp, RGB8 trunc in one pass."""
import torch
import triton
import triton.language as tl

@triton.jit
def _finalize(X,Y,STD,MEAN,N:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
              S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK).to(tl.int64)
    c=i%3;w=i//3%W;h=i//(3*W)%H;f=i//(3*W*H)
    raw=tl.load(X+f*S0+h*S1+w*S2+c*S3,i<N,other=0).to(tl.float32)
    std=tl.load(STD+c).to(tl.float32);mean=tl.load(MEAN+c).to(tl.float32)
    v=raw*std
    v=v+mean
    v=tl.minimum(tl.maximum(v,0.),1.)
    v=v*255.
    v=tl.minimum(tl.maximum(v,0.),255.)
    v=tl.where(raw==raw,v,0.)
    tl.store(Y+i,v.to(tl.uint8),i<N)

def quantize_raw(frames,std,mean):
    assert frames.is_cuda and frames.dtype in (torch.float16,torch.float32) and frames.ndim==4 and frames.shape[-1]==3
    assert not frames.requires_grad and std.numel()==mean.numel()==3
    std=std.to(device=frames.device,dtype=torch.float32).contiguous()
    mean=mean.to(device=frames.device,dtype=torch.float32).contiguous()
    out=torch.empty(frames.shape,device=frames.device,dtype=torch.uint8)
    _finalize[(triton.cdiv(frames.numel(),1024),)](frames,out,std,mean,frames.numel(),frames.shape[1],frames.shape[2],*frames.stride(),1024,num_warps=4,enable_fp_fusion=False)
    for t in [frames,std,mean]:t.record_stream(torch.cuda.current_stream(frames.device))
    return out
