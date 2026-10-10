# Reversability profiling: bottleneck diagnosis

10 October 2026. Evidence: `20261010T143701Z-b8bda223-profiling.zip`. All nine CUDA passes completed. ZIP CRC and all 34 exported files match the extracted folder. Runtime matches the baseline exactly: Tesla T4, PyTorch 2.11.0+cu128, NumPy 2.1.3, CUDA 12.8, Python 3.13.15. All 25 frozen execution-source hashes and the embedded profiling harness hash verify. Dataset/config/source provenance agree across all workers.

**Finding:** the full vocabulary projection/cross-entropy path dominates memory and accounts for most of the baseline GPU kernel time. Reversible inverse reconstruction plus local recomputation explains most of the additional matched-batch kernel work. The next optimization should address output/loss memory for both methods, then benchmark throughput again. Larger physical batch alone did not improve throughput in these probes.

## Training-only throughput

Each timing pass has five warmup updates and 20 uninstrumented synchronized complete optimizer updates. The model has 20,340,736 parameters, context 512 and accumulation 2. These are fresh short training-path probes, not full-run validation or billable session measurements.

| Configuration | Effective sequences/update | Mean update seconds | Valid targets/s | Allocated / reserved GiB |
|---|---:|---:|---:|---:|
| baseline_29 | 58 | 0.528418 | 56,197.9 | 10.030 / 12.766 |
| midpoint_29 | 58 | 0.656114 | 45,260.4 | 8.736 / 11.516 |
| midpoint_33 | 66 | 0.771685 | 43,789.9 | 9.891 / 13.076 |

Matched midpoint is **24.17% slower per equal target budget** than baseline, while retaining the prior **12.9% allocated-memory saving**. Midpoint 33 has **3.25% lower targets/s than midpoint 29**, and is **28.34% slower per equal target budget** than baseline. Per-update coefficient of variation is 0.45%, 0.93% and 0.51%, respectively; this describes within-pass variation, not cross-session confidence intervals.

The earlier full maximum-batch run was faster than matched midpoint. That measured full-run result remains valid. These new probes isolate a different workload: fresh early training, no evaluation/checkpoint I/O, one update boundary synchronization, and a single runtime. The full run changed update/evaluation/checkpoint counts. We cannot assign their discrepancy to one cause or claim that capacity automatically improves throughput. Select the fastest measured feasible batch, rather than automatically choosing the largest fitting batch.

## Allocator evidence: four vocabulary-sized buffers

The allocator history shows four simultaneously live large buffers around the loss-backward peak. Shapes and stack traces corroborate their roles.

| Buffer | Batch 29 GiB | Batch 33 GiB | Evidence |
|---|---:|---:|---|
| FP16 output logits | 1.389935 | 1.581651 | Projection allocation and vocabulary GEMM |
| FP16 log-softmax result | 1.389935 | 1.581651 | cross-entropy allocation and Half log-softmax trace |
| FP32 NLL input cast | 2.779871 | 3.163301 | Autocast NLL allocation/copy stack |
| FP32 dense NLL backward gradient | 2.779871 | 3.163301 | structured_nll_loss_backward allocation stack |
| **Total** | **8.339613** | **9.489903** | Simultaneous near-peak allocations |

Those four buffers alone equal **95.46% of midpoint-29’s 8.736 GiB peak** and **95.94% of midpoint-33’s 9.891 GiB peak**. For baseline 29 they equal **83.15% of the 10.030 GiB peak**. These fractions compare confirmed large-buffer sizes with measured total allocated peaks; they are not activation-only memory savings.

Midpoint-29 cross-entropy increases live allocation by up to **4.170 GiB**; loss backward adds another **2.780 GiB** at its peak. Its inverse/recompute/local-gradient phases peak around **2.20 GiB total live allocation**, well below the **8.74 GiB loss-backward peak**. Phase peaks overlap and must not be added.

The recorded build does **FP16 log-softmax followed by an FP32 NLL cast**, rather than computing the entire cross-entropy pipeline in FP32. This refines the earlier documentation-based hypothesis. Do not insert a blanket `.float()` or change loss precision while calling the experiment numerically unchanged. [Pinned PyTorch cross-entropy implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/LossNLL.cpp), [autocast registrations](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/autocast_mode.h).

Allocator replay reconstructs request-sized allocations near the peak and differs slightly from allocator-stat peaks because request sizes and actual reused/rounded block sizes differ. The audit keeps the discrepancy explicit (under 1%) and uses CUDA allocated statistics as the peak authority. The four large allocation sizes and their simultaneous lifetime are the ownership evidence; this does not measure non-PyTorch GPU allocations or fragmentation.

## GPU kernel attribution

To avoid double counting, the audit counts each raw `kernel` event once and joins its External id to the corresponding CPU leaf operator. Vocabulary GEMMs are classified by dimensions; vocabulary-sized loss/cast/gradient operations are classified by their input shapes. Reversible phases are identified through the smallest containing CPU annotation. CPU/GPU annotation duplicate rows and nested inclusive totals are not added.

| GPU kernel work per traced update | Baseline 29 ms | Midpoint 29 ms | Midpoint 33 ms |
|---|---:|---:|---:|
| Head forward projection | 73.644 | 76.491 | 93.976 |
| Head weight gradient | 30.824 | 32.079 | 41.197 |
| Head hidden-state gradient | 39.719 | 44.283 | 63.417 |
| Vocabulary loss/casts/gradient buffers | 196.821 | 196.746 | 225.022 |
| **Head + vocabulary loss total** | 341.007 | 349.599 | 423.612 |
| Inverse reconstruction | 0.000 | 56.579 | 67.065 |
| Local recomputation | 0.000 | 57.034 | 66.742 |
| **All captured kernel time** | 523.079 | 651.110 | 775.985 |

The head/loss path consumes **65.19% / 53.69% / 54.59%** of captured kernel time. Midpoint 29 adds **56.579 ms inverse reconstruction + 57.034 ms local recomputation = 113.613 ms**. The matched total-kernel gap is **128.031 ms**; these extra passes are roughly 89% of that gap. This is a one-update trace explanation of extra GPU work, not a guaranteed counterfactual wall-time saving.

At batch 33, head+loss time grows from **349.6 to 423.6 ms** (21.2%) for only 13.8% more tokens. The hidden-gradient GEMM rises from **44.3 to 63.4 ms**. This corroborates poorer per-token efficiency at that shape; exact kernel-choice/utilization causes would need additional shape sweeps and hardware counters.

The large CPU time charged to `loss_scalar` is predominantly host waiting for queued CUDA work at `.item()`. It must not be interpreted as hundreds of milliseconds spent computing a scalar on the CPU. Transfer and optimizer kernels are comparatively small. These traces do not establish GPU compute saturation, bandwidth utilization, power or MFU.

## Next experiment

1. Implement a bounded-memory output-projection plus loss path, shared by baseline and midpoint. The goal is to avoid materializing all `[tokens, vocabulary]` logits, log probabilities, FP32 NLL inputs and dense loss gradients at once.
2. Start with token chunks such as 512/1024/2048 and measure the tradeoff. A chunked projection/loss is a testable starting point; a fused projection-loss implementation may reduce copies further. Neither speed nor memory capacity improvement is guaranteed before measurement.
3. Bound the lifetime of every chunk’s autograd buffers. Chunking only after forming full logits cannot remove the logits allocation. Summing all chunk losses and delaying backward can retain every chunk graph. Repeatedly backpropagating each chunk through the whole Transformer can multiply reversible reconstruction work. Accumulate the gradient with respect to final hidden states across chunks, then perform one core backward; preserve tied-head and embedding gradient contributions.
4. Validate summed loss, ignored labels, partial target windows, tied weights, scaled gradients, clipping and optimizer updates against the current implementation. A change from the recorded Half-log-softmax/FP32-NLL behavior to FP32 accumulation is an explicit numerical-policy change requiring its own gates and quality validation.
5. Repeat the same matched timing/memory probes at batch 29 for both methods, then sweep physical batches for each optimized method under the same reserved-memory headroom rule. Test integer accumulation/unequal final microbatch arrangements when preserving a common effective target budget per update.
6. Confirm promising throughput configurations with quality-qualified full runs before declaring financial savings. Optimize the baseline equally; a faster reversible implementation alone does not prove it beats the strongest baseline.

**Decision:** prioritize output/loss memory and kernel work before more integrators, B200 spending or a larger-batch claim. Continue with midpoint as the accepted reversible method. Keep original source/results frozen and label all optimization outputs separately.

## Evidence limits and provenance

One traced update, one synchronized memory diagnostic update and 20 timed updates per scenario; one runtime and seed. Fresh weights/short warmup, no evaluation/checkpoint I/O. The tiny loss differences between fresh instrumented processes are below 0.00001 nats; these do not establish trained-model quality or deterministic CUDA equivalence. No optimizer patch, GPU rerun or B200 extrapolation was performed during this audit.

Archive SHA-256: `0d52d1fc9e491e2483679a6b77f5e1ba81c8a9b59f49b07e7a32c2062600a374`.

Files alongside this report: `profiling-audit-2026-10-10.json` contains detailed attribution, allocation frames and evidence hashes; `audit_profiling_results.py` reproduces the read-only audit. Original ZIP and extracted artifacts remain untouched.
