"""Experimental training paths; frozen model/trainer are unchanged."""
import contextlib
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from .reversible import forward_pair, midpoint_force


class ReusedMidpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, q, blocks, h, *parameters):
        ctx.blocks, ctx.h = blocks, h
        ctx.device = p.device.type
        ctx.amp = torch.is_autocast_enabled(ctx.device)
        ctx.dtype = torch.get_autocast_dtype(ctx.device)
        for block in blocks:
            p, q = forward_pair(block, p, q, "midpoint", h)
        ctx.save_for_backward(p, q, *parameters)
        return p, q

    @staticmethod
    @once_differentiable
    def backward(ctx, dp, dq):
        p, q, *_ = ctx.saved_tensors
        gradients = []
        with torch.autocast(ctx.device, enabled=ctx.amp, dtype=ctx.dtype):
            for block in reversed(ctx.blocks):
                # Current p IS previous q. One force graph serves inversion and VJP.
                with torch.enable_grad():
                    previous_q = p.detach().requires_grad_(True)
                    force = 2 * ctx.h * midpoint_force(block, previous_q)
                    previous_p = (q.detach() - force.detach()).requires_grad_(True)
                    # Keep the reference tuple VJP/identity accumulation structure.
                    local = torch.autograd.grad(
                        (previous_q, previous_p + force),
                        (previous_p, previous_q, *block.parameters()), (dp, dq))
                dp, dq = local[:2]
                gradients[0:0] = local[2:]
                p, q = previous_p.detach(), previous_q.detach()
                del force, local
        return dp, dq, None, None, *gradients


def hidden_states(model, ids, reuse=False):
    if ids.ndim != 2 or ids.shape[1] > model.config.max_sequence_length:
        raise ValueError("Invalid input shape/context")
    positions = torch.arange(ids.shape[1], device=ids.device)
    x = model.token_embedding(ids) + model.position_embedding(positions)
    if hasattr(model, "core"):
        if reuse:
            if model.method != "midpoint" or model.backward_mode != "reconstruct":
                raise ValueError("Force reuse requires reconstructed midpoint")
            _, x = ReusedMidpoint.apply(x, x, model.blocks, model.step_size, *model.blocks.parameters())
        else:
            x = model.core(x)
    else:
        if reuse:
            raise ValueError("Force reuse requires midpoint")
        for block in model.blocks:
            x = block(x)
    return model.final_norm(x)


def amp_context(device, precision, cache=True):
    if device.type == "cuda" and precision != "fp32":
        return torch.autocast("cuda", dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[precision], cache_enabled=cache)
    return contextlib.nullcontext()


def chunked_backward(model, ids, labels, denominator, chunk_tokens, scaler=None,
                     precision="fp32", reuse=False, loss_backend="torch"):
    """One core forward/backward; release each vocabulary graph immediately.

    Keep original cross_entropy autocast policy. Head weight remains tied;
    head gradients add to embedding gradients. Disable head cast caching to
    avoid reusing a freed cast graph across chunks or the body backward.
    """
    if chunk_tokens <= 0 or denominator <= 0:
        raise ValueError("Chunk size and normalization denominator must be positive")
    with amp_context(ids.device, precision):
        hidden = hidden_states(model, ids, reuse)
    leaf = hidden.detach().reshape(-1, hidden.shape[-1]).requires_grad_(True)
    targets = labels.reshape(-1)
    loss_total = torch.zeros((), device=ids.device, dtype=torch.float32)
    for start in range(0, len(targets), chunk_tokens):
        with amp_context(ids.device, precision, cache=False):
            logits = F.linear(leaf[start:start + chunk_tokens], model.lm_head.weight)
            if loss_backend == "triton":
                from .kernels import cross_entropy_sum
                loss = cross_entropy_sum(logits, targets[start:start + chunk_tokens])
            elif loss_backend == "torch":
                loss = F.cross_entropy(logits, targets[start:start + chunk_tokens], ignore_index=-100, reduction="sum")
            else:
                raise ValueError("Unknown loss backend")
            normalized = loss / denominator
            (normalized if scaler is None else scaler.scale(normalized)).backward()
        loss_total += loss.detach().float()
        del logits, loss, normalized
    with amp_context(ids.device, precision):
        hidden.backward(leaf.grad.reshape_as(hidden))
    return loss_total


def cpu_snapshot(value):
    """Independent CPU tensors: later optimizer steps cannot mutate a snapshot."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return value
