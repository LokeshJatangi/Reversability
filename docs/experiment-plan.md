# Experiment plan

## Scope and status

Build and validate an approximately **20M trainable parameter** language-model baseline, then compare reversible Euler and midpoint variants under controlled conditions. All implementation and experiments are pending. This document defines a protocol, not measured results.

Use Colab initially, following the supplied plan. Record the actual accelerator and runtime environment at launch; no particular GPU, memory capacity, or throughput is assumed. Save checkpoints to persistent storage so a runtime interruption does not require restarting the full budget.

## Freeze before full runs

Record an immutable manifest containing dataset source identifiers and revisions, preprocessing and deduplication rules, domain labels, tokenizer files and hashes, vocabulary size, special-token policy, packing and boundary rules, sequence length, train/validation splits, ordered target stream, and seeds. Prevent train/validation leakage and evaluate on the same held-out targets for every variant.

Freeze the architecture dimensions, optimizer, learning-rate schedule, initialization policy, precision, loss computation, evaluation cadence, and software environment. Report the exact trainable parameter count, including embeddings and output head with weight tying handled correctly. Aim for approximately 20M parameters; disclose any variant-specific differences rather than attributing their effects solely to reversibility.

The baseline uses the `v1.0.0` FineWeb-Edu `sample-10BT` source and the pinned GPT-2 tokenizer. The prepared manifest records resolved commit hashes plus training, validation, and individual tokenizer-asset hashes. Its decoder has 7 layers, width 256, 8 attention heads, a 1024-unit SwiGLU hidden dimension, tied input/output embeddings, learned positions, RMSNorm, and no linear biases. Numeric training settings are frozen in `configs/baseline.json`; FP16 is used for compatibility across common Colab GPUs. Where later architectures permit, use the same initial weights; otherwise document the initialization mapping and unmatched parameters. Pair seeds across variants and state how many seeds were actually run.

## Exact target accounting

Each full run must consume exactly **50,000,000 supervised next-token training targets** from the frozen FineWeb-Edu stream:

| Source | Counted training targets |
| --- | ---: |
| FineWeb-Edu `sample-10BT` | 50,000,000 |

Count non-ignored labels after causal shifting and masking. Input context, padding, ignored labels, validation, correctness probes, and benchmark warmup are not training targets. The preparation stream inserts and supervises GPT-2 EOS after each nonempty document. The quota applies after tokenization.

Maintain a committed global target counter. Use the frozen contiguous token stream with the same target order and context for each comparison. For a final partial batch or accumulation window, mask excess labels to hit the quota exactly; never round the budget to a whole batch. Normalize loss and accumulated gradients by the actual valid target count, including that final window. Assert the total at completion.

Count targets committed to successful optimizer updates. The FP16 gradient scaler is checkpointed. If an overflow skips an update, restore the window-entry random state, retry the same uncommitted target window, and log the overflow separately; never silently consume new targets while losing an update. Correctness probes and batch searches use disposable state; start full runs from the frozen initial state. A stopped short run must be labeled partial.

## Batch definitions and experiment order

Physical batch size is sequences per device per microstep. For regular full windows, effective batch size is physical batch size × gradient accumulation steps × data-parallel world size. Also report valid targets per microstep and optimizer update, because masked positions and partial windows break the simple fixed-length approximation.

| Stage | Configuration | Purpose |
| --- | --- | --- |
| 1 | Ordinary autograd baseline at a validated physical batch size | Establish correct training and reference loss, throughput, and memory |
| 2 | Reversible Euler at the baseline physical and effective batch sizes | Compare under matched batch conditions after correctness acceptance |
| 3 | Reversible midpoint at the baseline physical and effective batch sizes | Compare under the same controls after correctness acceptance |
| 4 | Selected reversible variant at maximum feasible physical batch size | Measure practical batch capacity and throughput under the same hardware memory budget |

Every full run in every stage has the same exact 50M-target budget. Decide the selection rule before examining results: require correctness and a predefined acceptable validation-loss difference, then compare measured memory and throughput. Choose and record the loss threshold before full runs; no threshold or winner is established yet.

Run batch-capacity searches separately from full training. Keep sequence length, precision, optimizer, hardware, and memory cap fixed; include optimizer-state allocation and representative complete updates in the fit test. Document the search range, failed sizes, repeated success checks, and memory headroom. State the largest verified feasible batch under that protocol rather than claiming an untested global maximum.

For stage 4, preserve the baseline effective batch size by adjusting accumulation when feasible. If the larger physical batch makes that impossible, explicitly label the change in effective batch size, update count, and optimization behavior. Use a schedule indexed by committed training targets and report any other adjustments. Do not interpret such a run as isolating the integrator alone.

## Correctness gates

First validate the baseline's causal labels, masks, loss reduction, target counters, parameter count, and finite losses and gradients on small deterministic inputs. Compare chunked cross-entropy loss and gradients against an unchunked reference, including ignored labels and a partial batch, before reusing notebook code.

Use the supplied *Reversing Large Language Models for Efficient Training and Fine-Tuning* (arXiv:2512.02056) as the primary specification. The required midpoint candidate maps to Eq. 2.4; the assignment's “Euler” candidate maps to the staggered Hamiltonian formulation resembling symplectic Euler in Eqs. 2.8–2.9. Before implementation, record the complete recurrence, boundary initialization, step size, parameterization, retained states, and custom backward reconstruction from the paper. Do not silently substitute ordinary forward Euler. Leapfrog Eq. 2.6 is an optional third candidate.

For each proposed reversible variant, compare reconstructed intermediate states, forward outputs, scalar losses, input gradients, and parameter gradients with a stored-activation autograd implementation of the same recurrence. Report maximum absolute and relative errors across depth, sequence length, step size, and intended precision, including a higher-precision small reference. Document handling of stochastic layers, random-number replay, normalization, and any retained state. Choose precision-specific tolerances before running acceptance checks. Require finite values, acceptable reconstruction and gradient errors, and a matching small optimizer update before full reversible training.

## Checkpoint and resume

Persist model, optimizer, FP16 gradient-scaler state, Python and framework RNG states on CPU and devices, the committed target cursor, optimizer step, target budget, and manifest/configuration hashes. The current learning-rate schedule is derived from the committed target counter and has no separate mutable state. Checkpoint only at optimizer-update boundaries.

Write checkpoints atomically and verify a completed checkpoint before using it. Restore the same stream and counters without duplicate or skipped committed targets. Test an interrupted short run against an uninterrupted run: require identical target order and counts, and parameter/optimizer agreement under the documented determinism and precision tolerances. Reject incompatible manifests at resume.

## Measurements and reporting

- **Final loss:** after exactly 50M training targets, evaluate the final checkpoint on the frozen validation set in evaluation mode. Sum negative log-likelihood over valid targets and divide by their total; report natural-log units overall and by domain. Report the final training-window loss separately with its window size. Keep a best-validation checkpoint's loss distinct from final-checkpoint loss.
- **Throughput:** after separately logged warmup, synchronize device work around a fixed measurement window of complete optimizer updates. Divide committed valid training targets by elapsed wall-clock seconds. Include forward, backward, recomputation, optimizer, and data delivery; disclose excluded validation and checkpoint time. Also report end-to-end training time with evaluation/checkpoint overhead and resumed segments accounted for, excluding offline downtime explicitly.
- **Peak memory:** use a fresh process per configuration where possible; initialize optimizer state, synchronize, reset device peak statistics, and measure representative full updates. Report peak allocated and reserved device memory separately in GiB, hardware capacity, and any CPU offload. Record startup/first-update peaks separately and consider them in batch feasibility. State precisely what the memory window includes.
- **Reproducibility:** save exact commands, configuration and manifest files, logs, checkpoints, code revision or source hash, seed, GPU model/count/VRAM, CPU/RAM, framework/compiler/driver versions, precision and accumulation dtype, attention implementation, compilation settings, and physical/effective batch sizes.

Repeat benchmark windows to expose variation and report the number of repetitions and spread. Record whether comparisons ran on identical hardware and software; qualify results when they did not. No commands or numerical outcomes should be invented before the harness exists. Store each run's configuration, metrics, and checkpoint references together and summarize them in [Progress](../Progress.md).

## Completion criteria

The baseline setup is complete when offline tests pass and the Colab handoff is verified. The baseline experiment is complete only after dataset preparation, its frozen manifest, a successful smoke run, a checkpoint/resume check, and exactly 50M committed targets have recorded evidence. The broader reversible experiment remains incomplete until reversible reconstruction/gradient checks and the staged comparisons have recorded evidence. Report failures and missing stages explicitly. Future studies belong in the [research roadmap](research-roadmap.md).
