# Capacity results and comparability — 2026-10-10

## Audited outcome

Physical50 is confirmed for ordinary baseline and midpoint, with accumulation1. The selected paths confirm physical285 baseline and physical814 midpoint in three fresh5-warmup/20-timed-update processes. Last confirmations include CPU checkpoint staging. All are FP16, context512,20,340,736 parameters, original ordered data, seed1337 and T4. The selected paths' next batches286/815 execute but exceed the fixed90%-reserved limit; they are headroom failures rather than hardware OOM. Hardware OOM occurs at384/1024 in the discovery sweep.

| Selected path | Physical maximum | Peak allocated GiB | Peak reserved GiB | Reserved % | Median targets/s |
|---|---:|---:|---:|---:|---:|
| baseline | 285 | 12.035 | 13.096 | 89.923 | 58,269 |
| midpoint | 814 | 11.682 | 13.082 | 89.830 | 38,647 |

Baseline maximum uses chunk1024 + Triton CE + fused SwiGLU + expandable allocator. Midpoint maximum uses chunk1024 + Triton CE, original SwiGLU and default allocator; force reuse and fused AdamW are off. The2.856× capacity ratio compares these selected paths, not identical-feature configurations or every possible optimization. Selecting the smallest reserved peak at batch50 does not guarantee the best capacity at larger shapes.

## Why the old maximum33 changed

The original implementation materializes vocabulary logits, log-softmax, FP32 NLL input and its dense FP32 gradient across every token. At physical33/context512/vocab50257 those four overlapping buffers alone represent about9.49GiB. Reversibility does not remove vocabulary buffers.

V2 processes the **whole physical batch through the Transformer**, then projects/loss-processes at most1024 output tokens at once, releasing each head graph and accumulating its hidden/tied-head gradients. It sends the completed hidden gradient through the Transformer once. For physical814 there are416,768 body tokens and407 head chunks, followed by one optimizer update. The Triton CE path removes the full log-softmax and FP32 NLL buffers while preserving the tested Half rounding policy. This is output-head tiling, not accumulation of smaller Transformer batches.

Shared head optimization raises ordinary baseline capacity29→285 too. Reversibility's activation reconstruction then becomes visible as a much larger core-memory advantage. The old33 used the original head/loss and accumulation2; the new814 uses a different head implementation and accumulation1. Architecture, context, device and data remain the same. The old maximum is valid historical evidence for its original implementation, not a contradictory result.

## What is directly comparable

The strongest existing controlled memory comparison is physical50, accumulation1, chunk1024, Triton CE, original SwiGLU, default allocator, no force reuse and no fused optimizer for both models:

| Matched physical50 CE-only probe | Peak allocated GiB | Peak reserved GiB | Targets/s |
|---|---:|---:|---:|
| baseline | 2.871 | 2.955 | 60,373 |
| midpoint | 1.033 | 1.096 | 48,666 |

Midpoint uses64.0% less allocated and62.9% less reserved memory, with19.4% lower throughput in these single probes. This directly supports a memory advantage under matched shared optimizations. It is not an equal-final-loss comparison.

New unoptimized physical29 reference probes use accumulation1 (effective29); the historical full baseline used accumulation2 (effective58). New speed comparisons are valid within the declared single-batch probe protocol, but cannot be described as identical to the historical training regime. References also drift from roughly54k targets/s at the start to50k at the end; preserve bracketing samples rather than using one convenient reference.

At physical50, three selected baseline repeats reach median63,587 targets/s and pass no-speed-loss versus the fastest current reference. Three selected midpoint repeats reach48,553 and fail that optional speed criterion. A separate midpoint CE+SwiGLU+force-reuse probe reaches57,479 targets/s with1.148GiB reserved, but lacks three fresh repeats; it is a promising optional speed candidate, not confirmed no-speed-loss.

## Correctness and checkpoint evidence

ZIP CRC and all329 member hashes audited. All110 standalone result records match their suite entries.108 complete,2 OOM,7 completed probes rejected by reserved headroom. All seven numerical gates pass every1024/2048/4096 chunk for baseline/midpoint across two updates. Worst gradient relative L2≤.008805, loss relative≤3.88e-6, per-step optimizer coordinate≤.00119650 against inherited.0012, whole-update relative L2≤.02933 against.05. These short B8/context512 gates do not establish long-run validation equivalence.

Embedded source receipts match the delivered v2 notebook; every worker reports those hashes. Frozen source/config/data hashes and runtime match; all25 frozen project sources remain unchanged. Colab Triton3.6.0 executed the kernels successfully; the earlier local compilation used3.5.1. Original artifacts are untouched. CPU checkpoint records pass reported CPU roundtrips, but checkpoint binaries are intentionally absent from the compact ZIP, so their bytes/checksums cannot be independently re-read from this export.

- baseline max checkpoint: CPU copy0.247s, Drive write1.105s, 295,621,857bytes, separately recorded and excluded from targets/s.
- midpoint max checkpoint: CPU copy0.220s, Drive write5.957s, 295,621,857bytes, separately recorded and excluded from targets/s.

At814, peak reserved89.83% leaves25.4MiB before the fixed90% boundary; at285,89.92% leaves11.4MiB. The prescribed10% reserve remains available; these smaller numbers are room inside the experimental90% budget. Full training/evaluation can change peaks, so the runtime limit must remain enforced. GPU busy sampling is near100%, which does not imply100% allocated memory, occupancy or arithmetic efficiency.

## Remaining bottleneck and next experiment

The failure stacks have moved from vocabulary loss to the core: baseline384 fails allocating a fused-MLP output during forward; midpoint1024 fails inside a reconstructed block's local backward/VJP. At814, reserved13.082GiB versus allocated11.682GiB also exposes allocator slack. Next targeted capacity experiment can test already-gated CE+SwiGLU+force-reuse and expandable segments near the large-batch boundary, with identical-feature baseline controls and fresh confirmations. It must independently gate any new feature composition; do not assume its batch50 ranking predicts large-batch capacity.

For the final-quality experiment, preserve the current verified814 path as the capacity result and test exactly50M targets, final validation1M targets, loss≤5.56229485, accumulation1, same seed/model/data/optimizer/target-indexed schedule. A fresh initialization is required; probe checkpoints are not production resumes. Use bounded validation head/loss and target-based evaluation/checkpoint cadence. Record all checkpoints on CPU with copy/write stalls separately and production resume verification.

Physical814 gives120 optimizer updates for50M targets; selected baseline285 gives343; new physical29/accumulation1 gives3368; historical physical29/accumulation2 gave1684. Thus capacity growth substantially changes optimization, and the same loss cannot be inferred from short training probes. Retain batch50 as a practical quality control and a fresh physical29/accumulation1 baseline to isolate the protocol change. Do not increase accumulation to match effective batches, raise the+.10 loss limit or present raw memory capacity as equal-quality savings. If814 fails the loss threshold, the experiment still establishes memory capacity, while a smaller physical batch may be needed for equal-quality training.

No full50M run or validation result is in this archive. Speed and monetary savings remain unproved for the final-quality objective. Measure end-to-end equal-quality accelerator time, including validation/checkpoints, before pricing savings.
