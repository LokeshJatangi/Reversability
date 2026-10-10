"""Experimental T4 kernels. CUDA gates must pass before performance claims.

CE keeps the audited Half-logsoftmax -> FP32 NLL policy, including Half
rounding before exp in backward. Tiled reductions avoid a 50k-wide CTA.
Approximate exp/reduction order is separately validated, not bitwise promised.
"""
import types
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _ce_partials(X, PM, PS, COLS: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * COLS + col, col < COLS, other=-float("inf")).to(tl.float32)
    m = tl.max(x, 0)
    s = tl.sum(tl.exp(x - m), 0)
    tl.store(PM + row * TILES + tile, m)
    tl.store(PS + row * TILES + tile, s)


@triton.jit
def _ce_rows(X, Y, PM, PS, RM, LD, LOSS, COLS: tl.constexpr, TILES: tl.constexpr, REDUCE: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.arange(0, REDUCE)
    m = tl.load(PM + row * TILES + tile, tile < TILES, other=-float("inf"))
    s = tl.load(PS + row * TILES + tile, tile < TILES, other=0.)
    maximum = tl.max(m, 0)
    denominator = tl.log(tl.sum(s * tl.exp(m - maximum), 0))
    target = tl.load(Y + row)
    valid = target != -100
    x = tl.load(X + row * COLS + tl.where(valid, target, 0)).to(tl.float32)
    logp = ((x - maximum) - denominator).to(X.dtype.element_ty).to(tl.float32)
    tl.store(RM + row, maximum)
    tl.store(LD + row, denominator)
    tl.store(LOSS + row, tl.where(valid, -logp, 0.))


@triton.jit
def _ce_grad(X, Y, RM, LD, G, DX, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * COLS + col, col < COLS, other=0.).to(tl.float32)
    logp = ((x - tl.load(RM + row)) - tl.load(LD + row)).to(X.dtype.element_ty).to(tl.float32)
    target = tl.load(Y + row)
    # Original NLL writes negative upstream FP32 gradient, then casts to Half.
    negative = (-tl.load(G)).to(X.dtype.element_ty).to(tl.float32)
    dx = tl.where(col == target, negative, 0.) - tl.exp(logp) * negative
    dx = tl.where(target != -100, dx, 0.)
    tl.store(DX + row * COLS + col, dx, col < COLS)


class TiledCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        if logits.device.type != "cuda" or logits.dtype not in (torch.float16, torch.float32):
            raise ValueError("Triton CE requires CUDA FP16/FP32 logits; no silent fallback")
        if logits.ndim != 2 or not logits.is_contiguous() or labels.device != logits.device or labels.dtype != torch.int64 or labels.numel() != logits.shape[0]:
            raise ValueError("Contiguous [tokens,vocab] logits and matching CUDA int64 labels required")
        labels = labels.contiguous()
        rows, cols = logits.shape; block = 1024; tiles = triton.cdiv(cols, block)
        pm = torch.empty((rows, tiles), device=logits.device, dtype=torch.float32)
        ps = torch.empty_like(pm)
        rm = torch.empty(rows, device=logits.device, dtype=torch.float32)
        ld = torch.empty_like(rm); losses = torch.empty_like(rm)
        _ce_partials[(rows, tiles)](logits, pm, ps, cols, tiles, block, num_warps=4, enable_fp_fusion=False)
        _ce_rows[(rows,)](logits, labels, pm, ps, rm, ld, losses, cols, tiles, triton.next_power_of_2(tiles), num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(logits, labels, rm, ld)
        return losses.sum()

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        logits, labels, rm, ld = ctx.saved_tensors
        rows, cols = logits.shape
        dx = torch.empty_like(logits)
        _ce_grad[(rows, triton.cdiv(cols, 1024))](logits, labels, rm, ld, gradient.contiguous(), dx, cols, 1024, num_warps=4, enable_fp_fusion=False)
        return dx, None


def cross_entropy_sum(logits, labels):
    return TiledCrossEntropy.apply(logits, labels)


@triton.jit
def _swiglu_forward(U, O, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base = (i // WIDTH) * (2 * WIDTH) + i % WIDTH
    gate = tl.load(U + base, i < N, other=0.).to(tl.float32)
    value = tl.load(U + base + WIDTH, i < N, other=0.).to(tl.float32)
    silu = (gate / (1. + tl.exp(-gate))).to(U.dtype.element_ty).to(tl.float32)
    tl.store(O + i, silu * value, i < N)


@triton.jit
def _swiglu_backward(U, G, DU, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base = (i // WIDTH) * (2 * WIDTH) + i % WIDTH
    gate = tl.load(U + base, i < N, other=0.).to(tl.float32)
    value = tl.load(U + base + WIDTH, i < N, other=0.).to(tl.float32)
    upstream = tl.load(G + i, i < N, other=0.).to(tl.float32)
    sigmoid = 1. / (1. + tl.exp(-gate))
    silu = (gate * sigmoid).to(U.dtype.element_ty).to(tl.float32)
    dg_intermediate = (upstream * value).to(U.dtype.element_ty).to(tl.float32)
    dg = dg_intermediate * (sigmoid * (1. + gate * (1. - sigmoid)))
    tl.store(DU + base, dg, i < N)
    tl.store(DU + base + WIDTH, upstream * silu, i < N)


class FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, up):
        if up.device.type != "cuda" or up.dtype not in (torch.float16, torch.float32) or not up.is_contiguous() or up.shape[-1] % 2:
            raise ValueError("Fused SwiGLU requires contiguous CUDA FP16/FP32 and even final width")
        width = up.shape[-1] // 2
        output = torch.empty((*up.shape[:-1], width), device=up.device, dtype=up.dtype)
        _swiglu_forward[(triton.cdiv(output.numel(), 256),)](up, output, output.numel(), width, 256, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(up)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        up, = ctx.saved_tensors
        gradient = gradient.contiguous()
        du = torch.empty_like(up)
        _swiglu_backward[(triton.cdiv(gradient.numel(), 256),)](up, gradient, du, gradient.numel(), up.shape[-1]//2, 256, num_warps=4, enable_fp_fusion=False)
        return du


def enable_swiglu(model):
    def forward(mlp, x):
        return mlp.down(FusedSwiGLU.apply(mlp.up(x)))
    for block in model.blocks:
        block.mlp.forward = types.MethodType(forward, block.mlp)
