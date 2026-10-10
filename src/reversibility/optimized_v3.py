"""Experimental per-layer workspace reduction; no full-model microbatching."""
import types
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from .optimized import amp_context, ReusedMidpoint, hidden_states
from .reversible import forward_pair


def mlp_value(x, up, down, fused):
    u = F.linear(x, up)
    if fused:
        from .kernels import FusedSwiGLU
        value = FusedSwiGLU.apply(u)
    else:
        gate, value = u.chunk(2, -1)
        value = F.silu(gate) * value
    return F.linear(value, down)


class TiledMLP(torch.autograd.Function):
    """Save only normalized input and weights, recompute bounded token tiles.

    Attention still sees all sequences, and each optimizer update consumes one
    entire physical batch. This tiles a token-independent sublayer, not the model.
    Parameter gradients return through autograd inputs; AMP/tied-weight safe.
    """
    @staticmethod
    def forward(ctx, x, up, down, tokens, fused):
        if tokens < 1:
            raise ValueError('Positive MLP tile size required')
        ctx.tokens, ctx.fused = tokens, fused
        ctx.device = x.device.type
        ctx.amp = torch.is_autocast_enabled(ctx.device)
        ctx.dtype = torch.get_autocast_dtype(ctx.device)
        flat = x.reshape(-1, x.shape[-1])
        output = None
        with torch.autocast(ctx.device, enabled=ctx.amp, dtype=ctx.dtype, cache_enabled=False):
            for start in range(0, flat.shape[0], tokens):
                tile = mlp_value(flat[start:start+tokens], up, down, fused)
                if output is None:
                    output = torch.empty((flat.shape[0], down.shape[0]), device=x.device, dtype=tile.dtype)
                output[start:start+tokens].copy_(tile)
        ctx.save_for_backward(x, up, down)
        return output.reshape(*x.shape[:-1], down.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, up, down = ctx.saved_tensors
        flat = x.reshape(-1, x.shape[-1]); dy = gradient.reshape(-1, gradient.shape[-1])
        dx = torch.empty_like(flat)
        dup, ddown = torch.zeros_like(up), torch.zeros_like(down)
        with torch.autocast(ctx.device, enabled=ctx.amp, dtype=ctx.dtype, cache_enabled=False):
            for start in range(0, flat.shape[0], ctx.tokens):
                with torch.enable_grad():
                    local_x = flat[start:start+ctx.tokens].detach().requires_grad_(True)
                    y = mlp_value(local_x, up, down, ctx.fused)
                    local = torch.autograd.grad(y, (local_x, up, down), dy[start:start+ctx.tokens])
                dx[start:start+ctx.tokens].copy_(local[0])
                dup.add_(local[1]); ddown.add_(local[2])
                del y, local_x, local
        return dx.reshape_as(x), dup, ddown, None, None


class SplitMidpoint(torch.autograd.Function):
    """MLP VJP first, then attention VJP, avoiding overlapping saved graphs.

    A detached attention proxy retains the original add/cast graph. Its VJP
    supplies the full direct+MLP attention cotangent without manual AMP casts.
    Attention recomputation is the disclosed extra-compute tradeoff.
    """
    @staticmethod
    def forward(ctx, p, q, blocks, h, *parameters):
        ctx.blocks, ctx.h = blocks, h
        ctx.device = p.device.type
        ctx.amp = torch.is_autocast_enabled(ctx.device)
        ctx.dtype = torch.get_autocast_dtype(ctx.device)
        for block in blocks:
            p, q = forward_pair(block, p, q, 'midpoint', h)
        ctx.save_for_backward(p, q, *parameters)
        return p, q

    @staticmethod
    @once_differentiable
    def backward(ctx, dp, dq):
        p, q, *_ = ctx.saved_tensors
        gradients = []
        with torch.autocast(ctx.device, enabled=ctx.amp, dtype=ctx.dtype):
            for block in reversed(ctx.blocks):
                with torch.no_grad():
                    attention = block.attn(block.attn_norm(p))
                with torch.enable_grad():
                    previous_q = p.detach().requires_grad_(True)
                    proxy = attention.detach().requires_grad_(True)
                    mlp = block.mlp(block.mlp_norm(previous_q + proxy))
                    force = 2 * ctx.h * (proxy + mlp)
                    previous_p = (q.detach() - force.detach()).requires_grad_(True)
                    mlp_parameters = (*block.mlp_norm.parameters(), *block.mlp.parameters())
                    local = torch.autograd.grad(
                        (previous_q, previous_p + force),
                        (previous_p, previous_q, proxy, *mlp_parameters), (dp, dq))
                next_dp, q_mlp, attention_cotangent = local[:3]
                mlp_grads = local[3:]
                del attention, proxy, mlp, force, local
                # MLP saved tensors have been released before rebuilding attention.
                with torch.enable_grad():
                    attn_input = previous_q.detach().requires_grad_(True)
                    attention = block.attn(block.attn_norm(attn_input))
                    attention_parameters = (*block.attn_norm.parameters(), *block.attn.parameters())
                    attn_local = torch.autograd.grad(attention, (attn_input, *attention_parameters), attention_cotangent)
                dp, dq = next_dp, q_mlp + attn_local[0]
                mapping = {id(w):g for w,g in zip((*attention_parameters,*mlp_parameters),(*attn_local[1:],*mlp_grads))}
                gradients[0:0] = [mapping[id(w)] for w in block.parameters()]
                p, q = previous_p.detach(), previous_q.detach()
                del attn_input, attention, attn_local, attention_cotangent, q_mlp, mapping, mlp_grads
        return dp, dq, None, None, *gradients


def install_v3(model, *, core_mode='reference', mlp_tokens=0, swiglu=False, rmsnorm=False):
    signature = (core_mode, mlp_tokens, swiglu, rmsnorm)
    if hasattr(model, '_v3_signature'):
        if model._v3_signature != signature:
            raise ValueError('Cannot replace an installed v3 configuration')
        return model
    if core_mode != 'reference':
        if getattr(model, 'method', None) != 'midpoint' or model.backward_mode != 'reconstruct':
            raise ValueError('Optimized core requires reconstructed midpoint')
        core_function = {'reuse': ReusedMidpoint, 'split': SplitMidpoint}[core_mode]
        def core(self, x):
            _, y = core_function.apply(x, x, self.blocks, self.step_size, *self.blocks.parameters())
            return y
        model.core = types.MethodType(core, model)
    if mlp_tokens:
        for block in model.blocks:
            if block.mlp.up.bias is not None or block.mlp.down.bias is not None:
                raise ValueError('Tiled MLP experiment supports frozen bias=False only')
            def forward(self, x):
                return TiledMLP.apply(x, self.up.weight, self.down.weight, mlp_tokens, swiglu)
            block.mlp.forward = types.MethodType(forward, block.mlp)
    elif swiglu:
        from .kernels import enable_swiglu
        enable_swiglu(model)
    if rmsnorm:
        from .kernels_v3 import enable_rmsnorm
        enable_rmsnorm(model)
    model._v3_signature = signature
    return model


def cached_head_backward(model, ids, labels, denominator, chunk_tokens, scaler=None,
                         precision='fp32', reuse=False, loss_backend='torch'):
    """One head weight cast per update; preserve FP32 per-chunk gradient adds.

    A detached cast leaf prevents released-graph reuse. Do NOT accumulate its
    Half gradient across chunks: promote each chunk before adding to tied .grad.
    Delete the head copy before core backward; optimizer steps remain external.
    """
    if reuse or chunk_tokens < 1 or denominator < 1:
        raise ValueError('Use installed core mode and positive normalization/chunk')
    with amp_context(ids.device, precision):
        hidden = hidden_states(model, ids)
    leaf = hidden.detach().reshape(-1, hidden.shape[-1]).requires_grad_(True)
    weight = model.lm_head.weight
    dtype = weight.dtype if ids.device.type != 'cuda' or precision == 'fp32' else {'fp16':torch.float16,'bf16':torch.bfloat16}[precision]
    head = weight.detach().to(dtype).requires_grad_(True)
    targets = labels.reshape(-1)
    total = torch.zeros((), device=ids.device, dtype=torch.float32)
    for start in range(0, targets.numel(), chunk_tokens):
        with amp_context(ids.device, precision, cache=False):
            logits = F.linear(leaf[start:start+chunk_tokens], head)
            if loss_backend == 'triton':
                from .kernels import cross_entropy_sum
                loss = cross_entropy_sum(logits, targets[start:start+chunk_tokens])
            elif loss_backend == 'torch':
                loss = F.cross_entropy(logits, targets[start:start+chunk_tokens], reduction='sum', ignore_index=-100)
            else:
                raise ValueError('Unknown loss backend')
            normalized = loss / denominator
            (normalized if scaler is None else scaler.scale(normalized)).backward()
        chunk_grad = head.grad.detach().to(weight.dtype)
        if weight.grad is None:
            weight.grad = chunk_grad
        else:
            weight.grad.add_(chunk_grad)
        head.grad = None
        total += loss.detach().float()
        del logits, loss, normalized, chunk_grad
    del head
    with amp_context(ids.device, precision):
        hidden.backward(leaf.grad.reshape_as(hidden))
    return total
