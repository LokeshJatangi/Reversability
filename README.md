# Reversible Language-Model Training: Memory, Capacity and Throughput

A controlled study of ordinary autograd, explicit midpoint and Hamiltonian Euler on a **20,340,736-parameter dense Transformer**, trained from scratch on **exactly 50,000,000 FineWeb-Edu targets per full run** using Google Colab Tesla T4 GPUs.

**Result:** reversibility improves allocated memory and feasible physical batch size here, but does not improve end-to-end training time or estimated accelerator cost. Midpoint is the quality-qualified reversible variant. Its larger-batch run recovers some throughput, but remains slower than ordinary autograd. This is a measured tradeoff, not a universal claim for or against reversible training.

All four full runs, three separate smoke runs and the recorded capacity/correctness evidence have been audited. Source, notebooks, original lightweight logs, frozen configurations, plots and reproducible analysis are included. Large datasets/checkpoints are preserved separately rather than committed to Git.

## 1. Questions and experiment design

The study asks:

1. What is the largest repeatedly successful ordinary-autograd physical batch under a fixed memory-safety rule?
2. At the same physical and effective batch, do midpoint and Euler preserve acceptable quality, reduce memory and improve speed?
3. Which reversible method passes correctness and the predeclared quality threshold?
4. How far can its physical batch increase, and does the throughput gain offset reconstruction overhead?
5. What are the implications for time, accelerator cost and larger-scale claims?

The sequence was baseline capacity search and 50M-target training; correctness gates and separate smokes; matched midpoint/Euler 50M runs; an explicit evidence/availability review; a selected-method capacity search; and a fresh 50M-target run at its selected maximum. No full run continues another method's trained weights.

- **Physical batch:** sequences on one device per microstep.
- **Effective batch:** physical batch × gradient accumulation × number of devices; one device is used here.
- **Valid targets:** supervised next-token labels, excluding ignored/padded labels. Input context is not an additional counted training budget.
- **Throughput:** committed valid targets divided by trainer-process elapsed time, including its evaluation/checkpoint overhead.
- **Peak allocated / reserved memory:** PyTorch CUDA statistics, not isolated activation memory or total system memory.

## 2. Model, hardware and frozen controls

| Setting | Recorded value |
| --- | --- |
| Trainable parameters | **20,340,736** for every method; exact count difference **zero** |
| Model | Dense autoregressive decoder; 7 layers, width 256, 8 attention heads |
| Feed-forward / normalization | SwiGLU hidden width 1,024; RMSNorm |
| Embeddings / output | GPT-2 vocabulary 50,257; tied input/output weights; learned positions |
| Context / attention | 512 tokens / PyTorch causal scaled-dot-product attention |
| Dropout / linear biases | Zero / disabled |
| Accelerator | One Tesla T4, compute capability 7.5; VRAM 15,637,086,208 bytes |
| Software | Python 3.13.15; PyTorch 2.11.0+cu128; CUDA runtime 12.8; NumPy 2.1.3 |
| Precision | FP16 autocast and gradient scaling; FP32 parameter/residual storage |
| Compilation / seed | Disabled / 1337 |
| Optimizer | AdamW; betas (0.9, 0.95), weight decay 0.1, gradient clipping 1.0 |
| Learning rate | 0.0006 maximum, 0.00006 minimum; 1M-target warmup; target-indexed cosine schedule |
| Evaluation | Same 1M held-out targets; every 500 updates and at completion |
| Checkpoints | Every 100 updates and at completion; latest and best retained |

Sessions match accelerator class/software, not necessarily the same physical card. Baseline training was on 2026-09-27; reversible training was on 2026-10-04. The paper's normalization notation is instantiated with the baseline's RMSNorm to keep controls fixed. Seeded initialization maps identical named tensors, including depth-scaled output projections. Recurrences differ, so this is a controlled architecture comparison, not an identical-forward-function experiment.

## 3. Data and exact target accounting

FineWeb-Edu `sample-10BT`, requested revision `v1.0.0`, resolves to **`fc9850dff5e2d0f8f776efe41b24a1c49556cfc5`**. GPT-2 tokenizer revision is **`607a30d783dfa663caf39e06633721c8d4cfcd7e`**. Preparation streams documents, inserts EOS after each document, and creates the validation token stream first, followed by training. Stored IDs are uint16. Comparisons use the same ordered binaries, tokenizer assets, split and seed.

Every full run commits exactly **50,000,000 training targets**; four runs total 200M. Each smoke has a separate 65,536-target budget. Validation, correctness tests, capacity updates and failed probes do not count toward full training. Final partial batches are masked rather than rounded up; padding never counts.

| Accounting | Baseline and matched methods | Maximum midpoint |
| --- | ---: | ---: |
| Physical / accumulation / effective batch | 29 / 2 / 58 | 33 / 2 / 66 |
| Regular targets per microstep | 14,848 | 16,896 |
| Regular targets per update | 29,696 | 33,792 |
| Optimizer updates | 1,684 | 1,480 |
| Final partial-update targets | 21,632 | 21,632 |
| Completed training targets | 50,000,000 | 50,000,000 |

| Integrity record | SHA-256 |
| --- | --- |
| Data manifest | `71a12f5765cf37b7f5fb8753d51fa0828915a6db313ae20aef7fa9fb2e14df8a` |
| Training binary | `8d71872e4d7024801e459a6c44942907a89ae946105eb0905968746a0652c738` |
| Validation binary | `d5058ea280216a4504ac6c899028adfd1c5b236c3af9de5ff813dfc3e5ed61b1` |
| Reversible execution-source manifest | `74aba6d276986fba0115c61a2779a4f998947e71adec171df9db3557dfbce919` |

[Recorded data manifest](results/recorded/data/data_fineweb_edu_gpt2_50m_v1/manifest.json) includes tokenizer hashes and preprocessing provenance. Frozen Colab configurations—not starter defaults in `configs/`—are the source of truth for measured batches and paths.

## 4. Reversible methods and reconstruction

Primary source: [Reversing Large Language Models for Efficient Training and Fine-Tuning](https://arxiv.org/pdf/2512.02056). This is an independent equation-based implementation, not a pinned author-code reproduction. The explicit midpoint below is not the generalized Eq. 3.6 used in the paper's main midpoint experiments; paper-scale savings are not reproduced results here.

Let `A(x) = attention(attn_norm(x))` and `M(x) = mlp(mlp_norm(x))`.

### Explicit midpoint

Eq. 2.4 uses `f(x) = A(x) + M(x + A(x))` from Eq. 2.5:

```text
Forward:      p[l+1] = p[l-1] + 2*h*f_l(p[l])
Reconstruct:  p[l-1] = p[l+1] - 2*h*f_l(p[l])
```

Training fixes **h=0.5**, so `2h=1`. Both boundary states equal the same input embedding: `p[-1]=p[0]=embedding`. Seven independent blocks give seven reversible transitions; the final current state feeds final normalization and the tied head. Correctness probes also cover h=0.25; full-run losses were not used to tune h.

### Hamiltonian Euler

“Euler” means the staggered Hamiltonian, symplectic-Euler-like updates in Eqs. 2.8–2.9, with unit coefficients—not ordinary forward Euler:

```text
Forward:      q_new = q_old + A_l(p_old)
              p_new = p_old + M_l(q_new)
Reconstruct:  p_old = p_new - M_l(q_new)
              q_old = q_new - A_l(p_old)
```

Both starting states equal the embedding; the head reads `p_final`. No Euler step-size tuning is used; the config's midpoint step-size field does not change Euler's unit coefficients.

### Retained states and backward

A stack-level custom autograd function saves the final hidden-state pair and parameter references instead of every layer's activation graph. Backward reconstructs a layer under `no_grad`, rebuilds that layer's graph under the original autocast mode, computes its gradients, releases the local graph and proceeds backward. Shared initial-state gradient contributions are summed.

Retained stack states are constant in depth, plus one local recomputation graph. Reconstruction adds computation. Parameters, gradients, Adam states, embedding/head activations, logits, loss buffers and attention workspaces still consume memory. Dropout is zero; stochastic replay, distributed reversibility and higher-order derivatives are not claimed. [Detailed methods](docs/reversible-methods.md)

## 5. Correctness and quality acceptance

Both methods pass all **60 CUDA correctness cases** on the recorded T4/software. Tests compare reconstructed states, independently expressed recurrence outputs, logits/loss, input/parameter gradients and scaled/clipped AdamW updates against autograd through the same recurrence. Coverage includes FP64/FP32/FP16, seeds 1337/2026, depths 1/3/7, several lengths, midpoint step sizes and a full-width/context core probe. Separate real-data/full-model smokes cover the vocabulary/head path.

| Precision | State/output/loss/gradient atol / rtol | Maximum relative optimizer-update L2 error | Update-L2 limit |
| --- | --- | ---: | ---: |
| FP64 | 1e-9 / 1e-8 | 9.72278e-14 | 1e-7 |
| FP32 | 3e-5 / 3e-4 | 3.78595e-5 | 0.01 |
| FP16 AMP | 5e-3 / 5e-2 | 0.0173540 | 0.05 |

An earlier CPU policy-v1 test failed a post-Adam parameter comparison (maximum discrepancy 3.24247e-5) while reconstruction/gradients passed. Before reversible training, policy v2 separated optimizer tolerances and added global relative-update L2 gates. FP32 post-update bounds are 5e-5 / 3e-4; FP16 absolute bound is 1.2e-3. The failure and amendment remain recorded; no post-result threshold relaxation occurred. [Earlier CPU report](results/local-correctness/reversible_cpu.json), [accepted CUDA report](results/recorded/reversible-v1/correctness/cuda.json)

Selection was fixed before matched training: correctness first; then **final validation ≤ baseline + 0.10 nats/target**; then lowest accepted final loss, highest throughput and lowest peak allocated memory. The cutoff is:

```text
5.462294847167969 + 0.10 = 5.562294847167969
```

This allowance is a declared study choice, not proof of identical quality. **Midpoint qualifies; Euler does not**, exceeding the cutoff by 0.0033573535 nats. Selection was not made solely from a visually preferred trajectory. The explicit review considered correctness, loss trajectories, memory, time, failures and remaining availability before maximum execution. [Review decision](results/recorded/reversible-v1/review-decision.json)

## 6. Every full-run result

All four runs complete exactly 50M targets. Final validation is also best recorded validation for each, at its final update. Loss is mean nats per valid target; final training loss is the last window, not a full-stream average.

| Run | Physical / effective batch | Final / best validation | Final training-window loss | Updates | Trainer time | Valid targets/s | Peak allocated / reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary autograd | 29 / 58 | 5.4622948472 | 5.4724449745 | 1,684 | 1,119.090 s (18m39s) | 44,679.159 | 10.030179 / 12.765625 |
| Midpoint matched | 29 / 58 | 5.4430287568 | 5.4516894098 | 1,684 | 1,399.973 s (23m20s) | 35,714.975 | 8.735979 / 11.517578 |
| Euler matched | 29 / 58 | 5.5656522007 | 5.5836167194 | 1,684 | 1,401.316 s (23m21s) | 35,680.744 | 8.735124 / 11.517578 |
| Midpoint maximum | 33 / 66 | 5.5091217800 | 5.5230157615 | 1,480 | 1,254.685 s (20m55s) | 39,850.652 | 9.891305 / 13.076172 |

### Matched midpoint versus baseline

- Final validation **0.0192660903 nats lower**.
- Peak allocated memory **12.9031% lower**, approximately 1.2942 GiB saved; reserved memory **9.7766% lower**.
- Process time **25.0992% longer**; throughput **20.0635% lower**.

This is the clean matched-batch comparison: useful total CUDA-memory savings with a throughput penalty. The small loss advantage uses one paired seed and is not established as statistically significant.

### Maximum midpoint versus baseline

- Verified physical capacity **33 vs 29**, a **13.7931% increase**.
- Final validation **0.0468269329 nats higher**, still within the cutoff.
- Process time **12.1165% longer**; throughput **10.8071% lower**.
- Peak allocated memory **1.3846% lower**; reserved memory **2.4327% higher**.

Compared with matched midpoint, maximum midpoint takes **10.3779% less process time** and delivers **11.5797% higher throughput**, but final validation is 0.0660930232 nats worse. Larger batching recovers some reconstruction overhead without beating baseline.

### Explicit effective-batch confound

Physical batch 33 cannot preserve effective batch 58 with integer accumulation. Accumulation 2 yields effective batch **66**, **204 fewer updates**, three evaluations rather than four, and 15 checkpoint events rather than 17. The schedule stays target-indexed, but batch noise and update count differ. Timing improvements over matched midpoint cannot be attributed solely to GPU batching. [Batch accounting](results/recorded/reversible-v1/maximum-batch-accounting.json)

## 7. Training and validation trajectories

![Recorded training and validation trajectories](docs/figures/core-loss-2026-10-06.svg)

The x-axis is committed targets, not optimizer steps. Curves are raw sampled window losses: 169 samples per baseline/matched run, 148 for maximum midpoint; not smoothing or every-update logging.

| Committed targets | Baseline validation | Matched midpoint | Matched Euler |
| ---: | ---: | ---: | ---: |
| 14,848,000 | 6.1642629458 | 6.1250958086 | 6.1444386309 |
| 29,696,000 | 5.6779289814 | 5.6488226064 | 5.7787491606 |
| 44,544,000 | 5.4938018132 | 5.4721031528 | 5.5978780791 |
| 50,000,000 | 5.4622948472 | 5.4430287568 | 5.5656522007 |

Maximum midpoint evaluates **6.1133024404 at 16.896M**, **5.6448935596 at 33.792M**, and **5.5091217800 at 50M** targets. Every run's recorded validation decreases monotonically; matched midpoint beats baseline at all four matched evaluations. Maximum intermediate evaluations are not target-aligned with the matched runs.

Complete sampled training/validation trajectories, checkpoint events and timestamps are committed in `results/recorded/**/metrics.jsonl`, with original console logs and final `run_summary.json` files. No weights are needed to inspect them.

## 8. Capacity search and headroom

Each candidate runs in a fresh CUDA process, initializes Adam state with one full update and executes **three further complete updates** at the configured accumulation. Passing requires finite/stable losses and **10% reserved-VRAM headroom**, including startup and steady peaks. This is a safety protocol, not an absolute OOM limit.

| Method | Passing candidates | Executed but failed headroom | OOM candidates | Selected |
| --- | --- | --- | --- | ---: |
| Ordinary autograd | 1, 2, 4, 8, 16, 24, 28, 29 | 30, 32 | None in its recorded search | **29** |
| Midpoint | 1, 2, 4, 8, 16, 32, 33 | 34, 36 | 40, 48, 64 | **33** |

The midpoint search is not capped. Separate batch-33 confirmation with actual accumulation 2 passes; the full 50M run then completes at that batch.

| Midpoint probe | Peak reserved bytes | Remaining VRAM | Outcome |
| --- | ---: | ---: | --- |
| 33 | 14,040,432,640 | 10.2107% | Pass, independently confirmed |
| 34 | 14,455,668,736 | 7.5552% | Updates execute; insufficient headroom |
| 36 | 15,290,335,232 | 2.2175% | Updates execute; insufficient headroom |

Allowed reserved memory is 14,073,377,587 bytes. Batch 35 and the exact physical OOM boundary are unmeasured; 34/36 are short probes, not full sustained-safe runs. Increasing the search ceiling alone would not change the headroom boundary. [Baseline search](results/recorded/benchmarks/baseline_batch_capacity.json), [midpoint search](results/recorded/reversible-v1/benchmarks/midpoint_maximum_capacity.json), [confirmation](results/recorded/reversible-v1/benchmarks/midpoint_maximum_confirmation.json)

## 9. Smokes, failures and resume history

Smoke budgets are separate. Their startup-dominated throughput is not a full-run performance comparison.

| Smoke | Targets / updates | Final / best validation | Seconds | Targets/s | Peak allocated / reserved GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | 65,536 / 3 | 10.6698841641 | 21.533655 | 3,043.422 | 10.030179 / 12.765625 |
| Midpoint | 65,536 / 3 | 10.6920480039 | 21.764737 | 3,011.109 | 8.735979 / 11.515625 |
| Euler | 65,536 / 3 | 10.8215066875 | 30.099976 | 2,177.277 | 8.735124 / 11.515625 |

All seven full/smoke streams have one startup from zero, monotonic recorded progress and no logged overflow retries. **No partial/resumed training segments exist in the supplied exports.** Maximum starts at 2026-10-04 12:06:21.614007 UTC; final summary is at 12:27:17.030046 UTC.

Recorded issues, separate from successful training:

- Deleted/mislocated Drive artifacts initially prevented staging; baseline-reference/data recovery restored frozen inputs. Not all earlier traces survived.
- Two matched preflights reject Torch cu130 versus frozen cu128 at 09:45:11 and 09:48:15 UTC on October 4. They commit no targets; corrected orchestration starts at 10:13:17 UTC.
- A Drive transport disconnection and replacement runtime were reported. No partial maximum-training segment survived in these exports. Lost work and wasted GPU time are **unknown**, not assumed zero.
- Maximum preflight at 12:01:29.982221 UTC fails on missing temporary local data. Restaging the original data fixes it; successful orchestration begins at 12:04:24.627148 UTC. The failed invocation does not start training.
- OOM/headroom probes and the earlier CPU optimizer-policy failure are preserved, not hidden as successes.

Checkpoints save model, optimizer, scaler, committed target cursor, configuration/data identity and Python/NumPy/CPU/CUDA RNG state. Resume replays uncommitted work without counting committed targets twice; offline tests verify interrupted/resumed agreement. Completed runs did not use checkpoint resume. Restarting the workflow/runtime is different from resuming training.

## 10. Time and estimated cost

Trainer time includes evaluation/checkpoint overhead, but excludes preparation/staging, setup, correctness gates, smokes, capacity searches and notebook downtime. Four full trainers total **5,175.064 seconds (1.4375 GPU-hours)**; known smokes add **73.398 seconds**. Successful maximum orchestration spans approximately **1,372.403 seconds**, including capacity/setup/confirmation; this is a timestamp window, not a billable-session measurement.

For an illustrative GPU-only estimate, Google Cloud lists **Iowa (us-central1) T4 at USD 0.35/GPU-hour**, checked 2026-10-06. This named-service scenario is **not measured Colab spend** or a complete VM quote. [Google Cloud GPU pricing](https://cloud.google.com/products/compute/gpus-pricing?hl=en)

| Full run | Trainer GPU-hours | Illustrative GPU-only USD |
| --- | ---: | ---: |
| Baseline | 0.310858 | 0.10880 |
| Midpoint matched | 0.388881 | 0.13611 |
| Euler matched | 0.389254 | 0.13624 |
| Midpoint maximum | 0.348524 | 0.12198 |

Formula: `trainer_seconds / 3600 × hourly_GPU_price`. Total is approximately **USD 0.50313**, excluding gates, probes, smokes, setup, failures/lost attempts, idle allocation, VM CPU/RAM, storage, transfers and taxes. Actual Colab compute units/spending and storage/transfer costs are unknown, not zero.

At identical constant accelerator rates, matched midpoint costs **25.0992% more** trainer time than baseline; maximum midpoint costs **12.1165% more**. **No monetary saving is demonstrated.**

## 11. Findings and limits

**Memory:** total peak allocated CUDA memory improves at matched batches. Activation-only savings were not separately profiled.

**Capacity:** 29 → 33 is verified under the same headroom rule. Batch 33 is not the absolute hardware limit, and 100% VRAM occupancy is not a throughput objective.

**Speed:** no win is measured. Larger batches partly recover reconstruction overhead; changed update/evaluation/checkpoint counts qualify the maximum-run comparison.

**Quality:** midpoint passes the declared cutoff; matched midpoint has the lowest recorded loss. Euler fails. One seed and a short fixed target budget do not establish statistical significance, time-to-equal-quality or a sufficiently trained large model.

**Plausible bottleneck, not measured attribution:** the code materializes full `[batch, context, vocabulary]` logits. At batch 33/context 512, one FP16 logits tensor is approximately **1.582 GiB**, or **3.163 GiB** in FP32. CUDA autocast cross-entropy uses FP32. The batch-40 OOM request of 3.84 GiB is consistent with a full FP32 vocabulary buffer, but exact allocation/lifetime attribution requires profiling. Non-reversible head/loss memory may limit the benefit of removing stack activations. [PyTorch 2.11 AMP reference](https://docs.pytorch.org/docs/2.11/amp.html#cuda-ops-that-can-autocast-to-float32)

**Utilization:** no compute-utilization, bandwidth, power, MFU or synchronized steady-state trace was recorded. Reserved-minus-allocated peaks do not measure simultaneous fragmentation. VRAM occupancy is not GPU compute saturation.

**Inflection and scaling:** no point of inflection is located. This single 20M/seven-layer/context-512 T4 point cannot determine when reversibility becomes faster or cheaper. No B200, 1K/2K/4K-context or 70B measurement/calibrated projection exists. Large-model analysis would need parameter/gradient/optimizer/activation/workspace decomposition, sharding/offload, communication and measured utilization; multiplying the small-model percentage is not defensible. Longer pretraining alone magnifies the current penalty, not savings. Scaling and additional optimization work are parked.

## 12. Evidence and persistent storage

| Location | Contents |
| --- | --- |
| [Recorded results](results/recorded) | Original lightweight metrics, summaries, console/orchestration logs, frozen configs, capacity attempts, CUDA gates, manifests and source snapshots |
| [CPU reports](results/local-correctness) | Original failed/accepted local numerical policies |
| [Submission inventory](results/submission-manifest.json) | Copied-file checksums and original raw-file inventory |
| [Raw duplicate lookup](results/raw-deduplication.json) | Removed duplicate paths, retained identical copies and recovery mapping |
| [Structured audit](docs/analysis/core-audit-2026-10-06.json) | Commands, configs, summaries, events, probes and checkpoint integrity |
| [Loss figure](docs/figures/core-loss-2026-10-06.svg) | All four full trajectories |
| [Detailed methods](docs/reversible-methods.md) / [extended report](docs/core-report-2026-10-06.md) | Numerical policy, reconstruction and provenance |

Final raw exports are split across `artifacts-20261004T124446Z-1-001` and `artifacts-20261004T124446Z-1-002`. Part 002 contains maximum midpoint `latest.pt` and matched Euler `best.pt`; it must not be omitted. Ten reversible checkpoints pass CPU checks for budgets/cursors, configuration/data hashes, finite model/optimizer tensors and resume fields. Baseline checkpoints were audited in an earlier export; the later recovery export intentionally omits them. Its inherited inventory is not a fresh complete export manifest.

Raw exports, ZIPs, datasets, recovery bundles and checkpoints are excluded from Git. Checksum-verified cleanup removed **80 byte-identical redundant raw copies (2,725,406,564 bytes)** while preserving **all 96 unique raw files** at their canonical original locations, preferring the final split export. No unique result was deleted. Git history preserves committed reports/evidence, not ignored binary exports. The curated submission contains one copy of each distinct lightweight evidence file, verified byte-for-byte. Persistent Colab results remain at:

```text
MyDrive/Reversability/baseline-recovery-7bldlite/artifacts/
└── reversible-v1/
    ├── runs/          # metrics.jsonl, console.log, run_summary.json, latest.pt, best.pt
    ├── benchmarks/    # search and confirmation
    ├── configs/      # frozen measured configurations
    ├── correctness/  # CUDA gates
    └── maximum-orchestration.log
```

Colab `/content` is temporary. Drive logs/checkpoints persist but depend on a functioning mount. After replacing a runtime, stage the same hashed dataset again; preserve Drive artifacts. Do not overwrite an incomplete run or silently substitute a newly prepared manifest in a comparison.

`submission-manifest.json` is the original pre-cleanup inventory; `raw-deduplication.json` records completed cleanup and current canonical copies. Older local export directories may now be partial. To restore a removed path, copy its `retained_path` to `original_path` from the lookup and verify the recorded SHA-256. The final two-part export remains complete, and no Google Drive content was changed.

## 13. Reproduction and verification

The [baseline Colab notebook](https://colab.research.google.com/github/LokeshJatangi/Reversability/blob/main/notebooks/baseline_colab.ipynb) prepares/stages Drive-backed data, searches capacity, runs its smoke and trains/resumes baseline. The [reversible notebook](https://colab.research.google.com/github/LokeshJatangi/Reversability/blob/main/notebooks/reversible_colab.ipynb) validates the frozen environment/data, CUDA correctness, smokes/matched runs and review before maximum execution. Completed runs are skipped. [Colab instructions](docs/colab-handover.md)

Original staged commands:

```bash
python3 scripts/run_reversible_study.py matched --artifacts /content/drive/MyDrive/Reversability/baseline-recovery-7bldlite/artifacts
python3 scripts/run_reversible_study.py review --artifacts /content/drive/MyDrive/Reversability/baseline-recovery-7bldlite/artifacts
# After persisting the recorded review-decision.json:
python3 scripts/run_reversible_study.py maximum --artifacts /content/drive/MyDrive/Reversability/baseline-recovery-7bldlite/artifacts --max-batch 256
```

Trainers call `scripts/train_baseline.py --config <frozen config> --device cuda`. Measured configurations are committed in `results/recorded/configs/` and `results/recorded/reversible-v1/configs/`, retaining original absolute Colab paths. Exact startup commands/environment/hashes are in `metrics.jsonl`; candidate commands/errors are in capacity records. Starter `configs/baseline.json` has provisional batch settings and is **not** the measured baseline config. Reproduction after relocation needs explicit path adaptation in a new run, not edits to recorded evidence.

Offline tests:

```bash
python3 -m pytest -q
```

Recreate the figure from committed logs without weights or dataset downloads:

```bash
python3 docs/analysis/plot_matched_results.py --artifacts results/recorded --include-maximum --output /tmp/reversibility-loss.svg
```

Reaudit the complete original raw exports when available:

```bash
python3 docs/analysis/audit_core_results.py \
  --parts artifacts-20261004T124446Z-1-001 artifacts-20261004T124446Z-1-002 \
  --output /tmp/reversibility-audit.json
```

The audit verifies seven streams, ten reversible checkpoints, all 25 frozen execution-source files, data/tokenizer hashes, review evidence and capacity. It inspects evidence on CPU; it does not rerun GPU training or recompute validation losses. Source snapshots remain included for source-level reproduction. Training code, numerical policy and recorded measurements are unchanged during submission preparation.
