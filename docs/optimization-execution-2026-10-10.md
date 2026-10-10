# Optimization execution — local verification

Implemented shared bounded-memory vocabulary projection/loss, midpoint force-graph reuse, gated fused AdamW comparison, a fresh-process batch/chunk sweep, and independent CPU checkpoint snapshots with separate copy/write overhead records. Custom kernels remain parked.

Executed **58 tests**, all passed. Re-executed the final **18-pass CPU suite**, all completed: numerical gates, six bracketing references, shared/midpoint ablations, candidate probes and repeated finalists. Both final CPU checkpoints round-tripped successfully. Tests verify exact next-update continuation from CPU checkpoint state, FP64/FP32 midpoint state and VJP equivalence, ignored labels, tied-weight gradients and accumulation.

The full **20,340,736-parameter** FP32 numerical gate also passed under the project's inherited optimizer policy, at CPU B2/context64. Worst per-parameter gradient relative L2: baseline6.03e-7; midpoint6.35e-6. Worst whole-update relative L2: baseline1.05e-6; midpoint1.34e-5. The initial proposed 2e-6 coordinate cap rejected one midpoint coordinate (7.44e-6); that failed evidence is preserved in the verification JSON. The final protocol uses the frozen v2 FP32 optimizer coordinate cap5e-5 plus whole-update limit.01, while retaining the strict gradient checks. GPU thresholds are fixed before GPU measurement; no GPU result was used to amend them.

All **25 frozen source files** retain their recorded hashes. Notebook code cells compile, embedded source hashes match the implementation, and staging/imports execute in a separate copy of the frozen checkout. Original profiling script and completed training artifacts are preserved.

**GPU experiment pending.** No local CUDA device or connected Colab session is available. No batch50 fit, speed improvement, allocated/reserved memory change or monetary savings is claimed. The T4 notebook performs full-context GPU gates for chunks1024/2048/4096, fixed10% reserved headroom, batches29/40/50, repeated throughput acceptance and separate CPU checkpoint overhead. Physical50/accumulation2 means effective100 versus reference58; a quality-controlled full run remains subsequent work.

Run `optimization_colab.ipynb` on a T4 using **Run all**, then return the exported `*-optimization.zip`. CPU checkpoint binaries remain separately in the unique Drive experiment folder.
