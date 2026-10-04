# Reversible methods: specification and correctness policy

Primary source: [Gal et al., arXiv:2512.02056, PDF](https://arxiv.org/pdf/2512.02056), §§2.1, 5.1 and 5.4; reviewed 2026-10-01. The HTML rendering labels equations without section prefixes; use the PDF equation numbers below. No author implementation repository was supplied or located in the title/arXiv-ID search. The paper's GitHub reference is nanoGPT for hyperparameters, not reversible author code. This is an independent equation-based implementation, not code-level reproduction. The paper's main midpoint experiments use its generalized Eq. 3.6; the assignment requires explicit Eq. 2.4, so those experiments are not reproduced here.

## Midpoint

Let `A(x) = attention(attn_norm(x))` and `M(x) = mlp(mlp_norm(x))`. Eq. 2.5 defines `f(x) = A(x) + M(x + A(x))`; computing `Block(x) - x` would introduce unnecessary cancellation and is not used. Eq. 2.4 is

```text
p[l+1] = p[l-1] + 2*h*f_l(p[l])
p[l-1] = p[l+1] - 2*h*f_l(p[l])
```

The stack represents `(previous, current)` and returns `(current, next)` per layer. The final current state feeds the baseline final normalization and tied vocabulary projection. Seven independent blocks give seven reversible transitions. Fixed study choice: `h=0.5`, giving `2h=1`; correctness probes also use `h=0.25`. No tuning on full-run losses is planned.

## Euler terminology

“Euler/oiler” means the Hamiltonian staggered update in Eqs. 2.8–2.9, with `a=b=1` as stated there. It does not mean ordinary forward Euler.

```text
q_new = q_old + A_l(p_old)
p_new = p_old + M_l(q_new)
p_old = p_new - M_l(q_new)
q_old = q_new - A_l(p_old)
```

The head reads `p_final`. There is no additional adjustable step size in these equations; unit attention/MLP coefficients are used. The config's `step_size=0.5` is a midpoint setting and is ignored by Euler.

## Declared study choices and controls

The reviewed equations specify embedding `p0` and reversible recurrences, but do not provide a complete executable boundary initialization/backward implementation or an explicit Eq. 2.4 experiment step size. We therefore disclose these choices: midpoint starts with `p[-1]=p[0]=embedding`; Euler starts with `p[0]=q[0]=embedding`. Both use the same single embedding tensor, so backward sums the two boundary contributions. No extra embedding or learned initial state is added.

The baseline's RMSNorm, SwiGLU, positional embeddings, initialization, vocabulary and attention implementation are reused; paper `LN` is instantiated as baseline RMSNorm to preserve the experiment controls. Seed 1337 initializes identical tensors by identical names and creation order, including the depth-scaled attention output/MLP down projections. This maps fresh initial weights, never the trained baseline checkpoint. Baseline, midpoint and Euler all have **20,340,736 trainable parameters** (difference zero). Recurrences still differ, so this is a controlled architecture comparison, not an identical-function comparison.

Dropout must be zero. Reconstruction rejects nonzero dropout, and compilation is rejected until independently validated. No stochastic replay, parameter sharing across blocks, higher-order derivatives or distributed training is claimed. AMP uses FP16 attention/MLP computations and FP32 residual states/parameter storage, like the baseline residual path. FP64 probes use FP64 throughout.

## Backward and retained states

The custom autograd Function spans the entire stack. Forward saves the final pair plus references to the parameters (for PyTorch version checks), without saving layer activation graphs. Backward inverts the last transition under `no_grad`, detaches reconstructed inputs, rebuilds that layer's forward graph under the original autocast mode, computes the vector-Jacobian product, then discards the local graph and continues backward. Parameter gradients are returned as Function-input gradients, allowing the standard autograd accumulator and FP16 scaler to handle them. It does not assign `.grad` directly.

Memory is constant in the number of retained stack states, plus one layer's recomputation graph. Embedding/head/logits/cross-entropy, optimizer, gradients and temporary attention buffers still consume memory. The current full-vocabulary loss can dominate capacity; no unvalidated chunked-loss optimization is added.

## Gates and recorded amendment

`scripts/validate_reversible.py` compares every reconstructed state, independently expressed recurrence outputs, logits, scalar loss with a masked target, input gradients, every parameter gradient, and the first AdamW update (including gradient clipping and the actual FP16 scaler on CUDA). Probe grid: depths 1/3/7, lengths 8/17/32, seeds 1337/2026, midpoint steps 0.25/0.5. An additional 7-layer, width-256, 8-head, 512-token case uses a small probe vocabulary to isolate the core. The actual 20M vocabulary/head is covered by the separate data-backed smoke before training. Probes consume no counted experiment targets.

Predeclared state/output/loss/gradient tolerances (elementwise `atol + rtol*abs(reference)`):

| Precision | atol | rtol |
| --- | ---: | ---: |
| FP64 | 1e-9 | 1e-8 |
| FP32 | 3e-5 | 3e-4 |
| FP16 AMP with FP32 states | 5e-3 | 5e-2 |

Policy v1 applied these bounds to parameters after Adam as well. Before any reversible training, the full-width midpoint FP32 case failed one post-Adam parameter comparison: maximum absolute discrepancy **3.2424693927168846e-5**, while states and gradients passed. The failure remains in `runs/correctness/reversible_cpu.json` and is not represented as acceptance.

Policy v2 separates the first optimizer update from reconstruction/gradient gates: FP64 parameter atol/rtol remain unchanged; FP32 post-update parameter atol is 5e-5 with rtol 3e-4; FP16 post-update absolute bound is 1.2e-3 (twice the nominal learning rate), with no relative allowance. Additionally, global L2 difference between the two parameter updates divided by the reference update L2 must be ≤1e-7 / 1% / 5% for FP64 / FP32 / FP16 respectively. This guards against accepting a poor overall optimizer update based on a loose absolute parameter bound. Adam's normalization amplifies differences around near-zero gradients; gradient gates remain unchanged. This disclosed amendment precedes all reversible full runs; it must not be relaxed automatically after a GPU failure.

CPU acceptance does not imply FP16 GPU acceptance. CUDA gates must pass on the baseline GPU/software and exact source before either smoke/full reversible run. Any failure stops the workflow, preserves the report/log, and requires investigation.

## Local acceptance evidence

On 2026-10-01, the final v2 CPU gate passed all 40 FP64/FP32 cases, including the full-width/context core. Maximum global relative update-L2 error was 9.83e-14 (FP64) and 3.38e-5 (FP32). Report: `runs/correctness/reversible_cpu_v2_final.json`. Eighteen offline tests passed, including exact interrupted/resumed model and optimizer agreement for both reversible methods and the baseline. FP16 CUDA gates and data-backed smokes remain pending because the local host has no CUDA device.
