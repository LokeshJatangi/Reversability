# Current training-path profiling

Authorized on 2026-10-10 as a separate profiling phase. The completed core study remains the frozen reference. No optional architecture, output/loss optimization, scaling experiment or full training run is part of this phase.

Open `notebooks/profiling_colab.ipynb` in Colab, select a **T4 GPU**, and run cells in order. The notebook embeds the tested profiling harness and clones the original implementation at `d071e1e2d8e4575d2fc9ca5ccb1ef4e3d5e185b7`; it does not require publishing these local additions to GitHub. It verifies the 25 frozen source hashes, baseline configuration checksum and original dataset. Torch/CUDA/NumPy and GPU class match the recorded study. Python version differences are declared in receipts, rather than silently presented as an identical environment.

The staging tool reuses the existing Drive export or baseline recovery ZIP. If it reports multiple roots, change the notebook's `ARTIFACTS` path to the intended export. Missing or changed data stops execution; no replacement data is downloaded. Probe output is a new timestamped/unique Drive directory under `Reversability/profiling-v1`, independent of training checkpoints and recorded results. Existing output directories are never overwritten.

## Probe protocol

- Baseline physical batch 29, midpoint physical batch 29, midpoint physical batch 33; context 512; accumulation 2; original model/optimizer/seed/target-indexed schedule; FP16 autocast and scaling.
- Three fresh processes per scenario: **timing**, **trace**, and **memory**. Each initializes fresh weights and Adam state, then performs five warmup updates on the same ordered training stream.
- Timing: 20 complete optimizer updates, CUDA synchronized at boundaries; no profiler or component wrappers. Report per-update durations, mean/median/variability, valid targets/s, and steady peak allocated/reserved memory. Warmup/startup peaks are separate. Per-update numerical checks happen outside the timed interval, except the scaler overflow gate required to reject skipped updates.
- Trace: one complete update with CPU/CUDA events, shapes, stacks and memory events. Mark data/transfer, forward, output projection, cross-entropy, backward, inverse reconstruction, local recomputation/VJP, attention/MLP forward calls, unscale/clipping and optimizer phases. Export Chrome/Perfetto trace and per-operator table/JSON. Missing CUDA kernels fails that pass explicitly.
- Memory: one complete update with synchronized nested phase boundaries and bounded allocator history; export raw phase peaks/live changes and an allocator snapshot. Nested resets preserve parent peaks, but synchronization changes allocator reuse, so use timing-pass peaks as the uninstrumented reference.
- No evaluation/checkpoint I/O is included. These probes measure the training path, not historical end-to-end training time or billable session cost. All probe/warmup targets are separately counted and never added to the completed 50M runs.

Batch 29 has effective batch 58; batch 33 has effective batch 66. Their throughput configurations are disclosed separately; no equal-update-budget quality claim is made. These fresh short probes do not measure validation quality, trained-state behavior or reconstruction stability over a full training budget.

## Interpreting evidence

Inclusive region times overlap: attention/MLP forward calls occur inside initial forward, inverse reconstruction and local recomputation. Do not add these ranges. Use operator **self-device** time for non-overlapping kernel rankings. Attention/MLP backward kernels remain visible in the operator trace; module forward annotations do not represent their entire gradient computation.

Memory region peaks describe all live PyTorch CUDA allocations during that region. They are not exclusive component-owned bytes or activation-only savings. A phase's rise above starting allocation is a useful diagnostic, not a complete ownership attribution. Allocator snapshots include allocation stack traces and reuse. The snapshots do not measure non-PyTorch CUDA allocations; reserved-minus-allocated peaks alone are not a fragmentation measurement.

The trace profiler retains extra references when recording shapes/stacks. Its peaks can exceed uninstrumented training and even OOM at a previously fitting batch. The suite keeps successful modes and failure logs, continues independent passes, and marks partial failure. It never relaxes the core study's capacity/headroom result. If a trace fails, inspect its log and declare a smaller-batch trace as a separate diagnostic rather than silently substituting it for batch 29/33.

Inspect output-projection/logits, FP32 cross-entropy/cast/log-softmax buffers, backward GEMMs, inverse reconstruction and local recomputation before selecting any optimization. The large theoretical logits size remains a hypothesis until the live peak/trace corroborates it.

## Outputs and return workflow

The notebook displays `REPORT.md`, timings, inclusive CUDA region rows, diagnostic memory rows and failures, then saves/downloads a ZIP containing all records, logs, traces, allocator snapshots and receipts. Return that ZIP for bottleneck attribution. Traces can be viewed in [Perfetto](https://ui.perfetto.dev/); snapshots can be viewed in the [PyTorch memory viewer](https://pytorch.org/memory_viz).

CLI, after preparing the frozen dataset:

```bash
python scripts/profile_training.py --suite \
  --config results/recorded/configs/baseline_colab.json \
  --data-dir /content/reversibility-profile-data/data_fineweb_edu_gpt2_50m_v1 \
  --output /content/drive/MyDrive/Reversability/profiling-v1/NEW_UNIQUE_RUN
```

`--cpu-smoke` is restricted to tiny configurations and explicitly labels results as CPU plumbing checks. It cannot stand in for CUDA measurements.

## Local verification

No CUDA device is available on the preparation machine. Tests compare complete baseline/midpoint optimizer updates with and without trace/memory annotations, require bit-exact CPU model/optimizer equality, check restoration after exceptions, reject tampered data and refuse existing outputs. The full nine-process tiny CPU suite exercises timing/trace/memory export and suite reporting; it is not GPU evidence. Notebook code cells and embedded-source checksum are also verified. The original training/model files and the 25-file frozen execution manifest remain unchanged.

Primary API references: [PyTorch profiler](https://docs.pytorch.org/docs/2.11/profiler.html), [CUDA allocator snapshots](https://docs.pytorch.org/docs/2.11/torch_cuda_memory.html).
