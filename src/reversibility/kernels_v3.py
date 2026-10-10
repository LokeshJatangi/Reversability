"""Experimental RMSNorm kernels for the frozen one-dimensional norm shape."""
import types
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _rms_forward(X, W, O, RS, WIDTH:tl.constexpr, BLOCK:tl.constexpr, EPS:tl.constexpr):
    row=tl.program_id(0); col=tl.arange(0,BLOCK)
    x=tl.load(X+row*WIDTH+col,col<WIDTH,other=0.).to(tl.float32)
    w=tl.load(W+col,col<WIDTH,other=0.).to(tl.float32)
    r=tl.rsqrt(tl.sum(x*x,0)/WIDTH+EPS)
    tl.store(O+row*WIDTH+col,x*r*w,col<WIDTH)
    tl.store(RS+row,r)


@triton.jit
def _rms_dx(X,W,G,RS,DX,WIDTH:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK)
    x=tl.load(X+row*WIDTH+col,col<WIDTH,other=0.).to(tl.float32)
    w=tl.load(W+col,col<WIDTH,other=0.).to(tl.float32)
    g=tl.load(G+row*WIDTH+col,col<WIDTH,other=0.).to(tl.float32)*w
    r=tl.load(RS+row)
    correction=tl.sum(g*x,0)/WIDTH*r*r
    tl.store(DX+row*WIDTH+col,(g-x*correction)*r,col<WIDTH)


@triton.jit
def _rms_dw_partials(X,G,RS,P,ROWS:tl.constexpr,WIDTH:tl.constexpr,COLS:tl.constexpr,TILE:tl.constexpr):
    part=tl.program_id(0)
    rows=part*TILE+tl.arange(0,TILE)
    cols=tl.arange(0,COLS)
    mask=(rows[:,None]<ROWS)&(cols[None,:]<WIDTH)
    x=tl.load(X+rows[:,None]*WIDTH+cols[None,:],mask,other=0.).to(tl.float32)
    g=tl.load(G+rows[:,None]*WIDTH+cols[None,:],mask,other=0.).to(tl.float32)
    r=tl.load(RS+rows,rows<ROWS,other=0.)
    # Match forward multiplication order (x*r) before multiplying gradient.
    value=tl.sum(g*(x*r[:,None]),0)
    tl.store(P+part*WIDTH+cols,value,cols<WIDTH)


@triton.jit
def _rms_dw_reduce(P,DW,WIDTH:tl.constexpr,PARTS:tl.constexpr,BLOCK:tl.constexpr):
    col=tl.program_id(0);part=tl.arange(0,BLOCK)
    values=tl.load(P+part*WIDTH+col,part<PARTS,other=0.)
    tl.store(DW+col,tl.sum(values,0))


class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,eps):
        if x.device.type!='cuda' or x.dtype!=torch.float32 or weight.dtype!=torch.float32:
            raise ValueError('RMSNorm v3 requires CUDA FP32 residuals and weights; no dtype substitution')
        if not x.is_contiguous() or weight.ndim!=1 or x.shape[-1]!=weight.numel():
            raise ValueError('Contiguous input and one-dimensional matching norm weight required')
        width=x.shape[-1];rows=x.numel()//width
        output=torch.empty_like(x);rs=torch.empty(rows,device=x.device,dtype=torch.float32)
        _rms_forward[(rows,)](x,weight,output,rs,width,triton.next_power_of_2(width),eps,num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(x,weight,rs)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        x,weight,rs=ctx.saved_tensors;gradient=gradient.contiguous()
        width=x.shape[-1];rows=x.numel()//width;tile=64;parts=triton.cdiv(rows,tile)
        dx=torch.empty_like(x);dw=torch.empty_like(weight)
        partial=torch.empty((parts,width),device=x.device,dtype=torch.float32)
        _rms_dx[(rows,)](x,weight,gradient,rs,dx,width,triton.next_power_of_2(width),num_warps=4,enable_fp_fusion=False)
        _rms_dw_partials[(parts,)](x,gradient,rs,partial,rows,width,triton.next_power_of_2(width),tile,num_warps=8,enable_fp_fusion=False)
        _rms_dw_reduce[(width,)](partial,dw,width,parts,triton.next_power_of_2(parts),num_warps=4,enable_fp_fusion=False)
        return dx,dw,None


def enable_rmsnorm(model):
    def forward(self,x):
        eps=self.eps if self.eps is not None else torch.finfo(x.dtype).eps
        return FusedRMSNorm.apply(x,self.weight,eps)
    for module in model.modules():
        if isinstance(module,torch.nn.RMSNorm):
            module.forward=types.MethodType(forward,module)
