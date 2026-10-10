# Memory and quality experiments — 2026-10-10

## Outcome required

Find and confirm the largest feasible **physical** midpoint batch on the frozen T4 experiment; then train it on exactly50M valid targets and require validation loss **≤5.56229485**, retaining the existing +.10 allowance over baseline5.46229485. Speed improvement is optional. The uploaded v1 run stopped at a correctness gate; it established no new capacity or throughput. Physical33 remains the last verified reversible maximum under the original accumulation2 protocol.

## Bottlenecks and implemented candidates

The unbounded vocabulary head/loss dominates current memory. At physical50/context512/vocab50257, four overlapping buffers from the audited AMP path extrapolate to14.3786GiB, exceeding the T4's13.107GiB reserved-memory budget before the core and optimizer. The recorded precision path is Half log-softmax followed by FP32 NLL. The fixed **10% reserved headroom** is retained.

1. Bound the output projection/loss to1024/2048/4096 output tokens; backpropagate each head graph immediately, then send the summed hidden gradient through the Transformer once. Tied embedding/head gradients remain additive. The core receives the entire physical batch; output-token chunking is not gradient accumulation.
2. Reuse one midpoint force graph for inverse reconstruction and its local VJP, removing a duplicate force evaluation. Preserve the original recurrence and tuple-VJP structure.
3. Test a tiled CE forward/backward kernel that keeps the observed Half rounding while avoiding full log-softmax and FP32 NLL workspace/gradient matrices. Test fused SwiGLU forward/backward to reduce intermediate activation buffers. Approximate exp and reduction order require numerical validation.
4. Separately gate built-in fused AdamW. Kernel CE/SwiGLU combination and its midpoint force-reuse composition each receive their own gate. Ungated feature combinations are excluded.
5. Try expandable allocator segments in fresh processes after a batch50 candidate fits. Record peak allocated/reserved memory, available reserved budget and coarse GPU busy samples. Fitting memory, utilization and kernel speed are distinct measurements.
6. Independently copy checkpoint model/optimizer/scaler/RNG/cursor/config/data state onto CPU. Record CPU-copy time, disk-write time, bytes and checksum separately; include checkpoint staging in the capacity limit. Training state remains GPU-resident. This is CPU disk-checkpoint staging, not activation offload, and its synchronous stalls are real overhead.

## Numerical policy and failed-run correction

The previous notebook added a FP16 coordinate cap2e-5, inconsistent with the frozen project's1.2e-3 optimizer policy. All12 uploaded loss/gradient/whole-update checks passed their limits; only that extra cap rejected the sweep. Preserve the failed artifact. V2 applies inherited per-step coordinate bounds1.2e-3 FP16 /5e-5 FP32 and cumulative divergence relative to one update L2.05/.01. Gradient relative L2 limits.02/5e-5 and relative loss.002/2e-6 remain. Two single-batch updates, tied weights, masked labels, scale128 and clipping are checked against frozen forward/CE. Cumulative parameter max error is reported separately. V1 lacks saved pre-step tensors, so it cannot be retrospectively passed under the per-step policy.

Shared chunk, force reuse, fused optimizer and kernel options gate independently. A failed feature/chunk is excluded; absence of any passing shared chunk blocks optimized probes. CUDA failures and precision checks are not replaced by CPU results or offline compilation.

## Capacity protocol

Use context512, the20,340,736-parameter model, original frozen ordered data/initialization and optimizer. **Accumulation1 for every experiment**, including unoptimized physical29 reference. Physical29 means effective29; physical50 means effective50. Six same-session baseline29 references bracket candidate probes. Compare valid targets/s, not raw update time. Historical56,197.916 targets/s used accumulation2 and is contextual only.

Test baseline and midpoint at physical29/40/50 with passing chunks and features. Kernel ablations apply to ordinary baseline too, preventing shared head savings from being attributed solely to reversibility. Select lowest reserved memory at the largest fitting tested shape, then hold that optimization path fixed while growing capacity:

29→33→36→40→44→48→50→64→96→128→192→256→384→512→768→1024.

Discovery uses3 warmup/3 timed updates per fresh process. Stop at OOM or reserved>90%; refine the integer boundary under the monotonic-capacity assumption for that fixed path. Unexpected runtime/overflow errors mark the search unresolved rather than becoming memory boundaries. Three fresh **5-warmup/20-timed-update** confirmations are required at the selected maximum, including a CPU checkpoint on the last. If the ceiling1024 fits, it is a lower bound requiring extension. This establishes capacity for a measured path, not a proof that every possible algorithm has been exhausted.

Batch50 receives a separate three-repeat milestone report. Speed has no bearing on memory acceptance. Optional no-speed-loss evidence requires slowest batch50 repeat≥fastest of six current-session physical29 references. Report optimized baseline versus midpoint as well.

## Final quality phase after GPU capacity review

Use the highest confirmed midpoint physical batch, accumulation1, original seed1337, FP16, architecture, ordered50M-target stream,1M validation targets, AdamW betas(.9,.95), decay.1, clip1 and target-indexed warmup1M/cosine50M. Keep selected optimization flags and all hashes in the full-run receipt. Handle the final partial target budget with masked labels and exact valid-target normalization; never add extra training targets to pad the run. Evaluation must use bounded head/loss as well, otherwise full-vocabulary validation buffers can reintroduce OOM. Evaluation batch can be reduced independently because it is not the training physical-capacity claim.

Run matched physical29/accumulation1 baseline control and shared-optimized ordinary baseline where feasible. Changing physical batch changes optimizer update count even at a common target budget; report this confound. The historical baseline remains the fixed loss threshold reference, not an identical-update-count control. Numerical gates are prerequisites; they do not prove final validation quality. Full50M training is not launched automatically by the probe notebook.

Retain original full-run checkpoints. A production CPU checkpoint/resume path must preserve exact config, optimizer/scaler/RNG/cursor and selected features, and be validated before full-run adoption; probe checkpoints are explicitly marked and not production resumes. Report checkpoint/evaluation cadence in valid targets, because equal update intervals would impose different overhead at different physical batches.

If the full run meets≤5.56229485, compute equal-quality end-to-end accelerator seconds (including checkpoint/evaluation stalls), then apply a named dated GPU price. No savings or reversible advantage is assumed. If ordinary baseline achieves the same capacity/quality more efficiently, report that result. If loss fails, investigate numerical drift/physical-batch optimization effects rather than raising the tolerance. Speed/kernel launch tuning follows memory and quality acceptance.

## Run now

Open `capacity_colab_v2.ipynb`, select T4 and Run all. It embeds all new sources, clones the frozen commit, checks25 frozen source hashes, pins Torch2.11.0+cu128/NumPy2.1.3, declares Python/runtime differences, and stages existing hash-verified data. No GitHub push is required.

Evidence goes to `MyDrive/Reversability/capacity-v2/<unique-run-id>`. Return the exported `*-capacity-v2.zip`; large CPU checkpoint binaries stay separately in Drive. V1 files/results remain preserved. The local machine has no CUDA GPU; CPU execution and sm75 offline compilation validate preparation only.
