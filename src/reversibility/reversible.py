"""Paper equations 2.4/2.5 and 2.8/2.9 with stack-level inverse backward.

Residual states remain in the embedding dtype (FP32 during AMP). Only the final
pair is saved; one layer's autograd graph is rebuilt at a time. No stochastic
layers or higher-order gradients are supported by the reconstruction path.
"""
from __future__ import annotations

import math
import torch
from torch.autograd.function import once_differentiable

from .model import BaselineLM, Block, ModelConfig


def midpoint_force(block: Block, x: torch.Tensor) -> torch.Tensor:
    attention = block.attn(block.attn_norm(x))
    return attention + block.mlp(block.mlp_norm(x + attention))


def forward_pair(block, p, q, method, h):
    if method == "midpoint":
        return q, p + 2 * h * midpoint_force(block, q)
    next_q = q + block.attn(block.attn_norm(p))
    return p + block.mlp(block.mlp_norm(next_q)), next_q


def inverse_pair(block, p, q, method, h):
    if method == "midpoint":
        return q - 2 * h * midpoint_force(block, p), p
    previous_p = p - block.mlp(block.mlp_norm(q))
    return previous_p, q - block.attn(block.attn_norm(previous_p))


class _InverseStack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, q, blocks, method, h, *parameters):
        ctx.blocks, ctx.method, ctx.h = blocks, method, h
        ctx.device_type = p.device.type
        ctx.amp_enabled = torch.is_autocast_enabled(ctx.device_type)
        ctx.amp_dtype = torch.get_autocast_dtype(ctx.device_type)
        for block in blocks:
            p, q = forward_pair(block, p, q, method, h)
        ctx.save_for_backward(p, q, *parameters)
        return p, q

    @staticmethod
    @once_differentiable
    def backward(ctx, dp, dq):
        p, q, *parameters = ctx.saved_tensors
        gradients = []
        # Parameter references are saved for version checks and returned through
        # autograd inputs, never written directly to .grad (AMP accumulation safe).
        with torch.autocast(ctx.device_type, enabled=ctx.amp_enabled, dtype=ctx.amp_dtype):
            for block in reversed(ctx.blocks):
                with torch.no_grad():
                    p0, q0 = inverse_pair(block, p, q, ctx.method, ctx.h)
                with torch.enable_grad():
                    p0 = p0.detach().requires_grad_(True)
                    q0 = q0.detach().requires_grad_(True)
                    next_p, next_q = forward_pair(block, p0, q0, ctx.method, ctx.h)
                    local_parameters = tuple(block.parameters())
                    local = torch.autograd.grad(
                        (next_p, next_q), (p0, q0, *local_parameters), (dp, dq),
                    )
                dp, dq = local[:2]
                gradients[0:0] = local[2:]
                p, q = p0.detach(), q0.detach()
        return (dp, dq, None, None, None, *gradients)


class ReversibleLM(BaselineLM):
    def __init__(self, config: ModelConfig, method: str, step_size: float = 0.5,
                 backward_mode: str = "reconstruct") -> None:
        if method not in {"midpoint", "euler"}:
            raise ValueError("method must be midpoint or euler")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("step size must be finite and positive")
        if method == "euler" and step_size != 0.5:
            raise ValueError("Euler has unit coefficients; step_size applies only to midpoint")
        if backward_mode not in {"reconstruct", "autograd"}:
            raise ValueError("unsupported backward mode")
        if config.dropout != 0:
            raise ValueError("reconstruction requires dropout=0; RNG replay is not implemented")
        super().__init__(config)
        self.method, self.step_size, self.backward_mode = method, step_size, backward_mode

    def core(self, x):
        # Explicit study boundary choice: midpoint p[-1]=p[0]=embedding;
        # Hamiltonian p[0]=q[0]=embedding. Both share the same embedding graph.
        p, q = x, x
        if self.backward_mode == "reconstruct" and torch.is_grad_enabled():
            parameters = tuple(self.blocks.parameters())
            p, q = _InverseStack.apply(p, q, self.blocks, self.method, self.step_size, *parameters)
        else:
            for block in self.blocks:
                p, q = forward_pair(block, p, q, self.method, self.step_size)
        return q if self.method == "midpoint" else p

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        return self.lm_head(self.final_norm(self.core(x)))


def build_model(config: dict):
    model_config = ModelConfig(**config["model"])
    method = config.get("method", "baseline")
    if method == "baseline":
        return BaselineLM(model_config)
    if config.get("compile"):
        raise ValueError("compiled inverse backward is not validated")
    return ReversibleLM(model_config, method, config.get("step_size", 0.5),
                        config.get("backward_mode", "reconstruct"))
