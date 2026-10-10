# Colab experiment notebooks

The four notebooks prepared on October 10, 2026 (published October 11) are listed below. Existing baseline and reversible-study notebooks remain unchanged.

| Notebook | Purpose / status |
| --- | --- |
| [Profiling](profiling_colab.ipynb) | Profile the original baseline and reversible midpoint; retained for reproducibility. |
| [Optimization v1](optimization_colab.ipynb) | Historical first experiment. Its uploaded correctness gate failed; it uses the earlier accumulation-2 protocol. Do not use as the current physical-capacity experiment. |
| [Physical capacity v2](capacity_colab_v2.ipynb) | Accumulation 1, chunked vocabulary loss and optional kernels. Uploaded T4 evidence confirmed baseline 285 and midpoint 814 at 10% reserved-memory headroom. These are short capacity probes, not completed full-budget loss runs. |
| [Midpoint optimization v3](midpoint_optimization_v3.ipynb) | **Run this next.** Independent correctness gates, fixed-shape optimization probes, three confirmations, 10% primary and separate 5% exploratory headroom. Full training and automatic maximum search remain disabled. |

[Open midpoint v3 in Colab](https://colab.research.google.com/github/LokeshJatangi/Reversability/blob/main/notebooks/midpoint_optimization_v3.ipynb). Select a T4 and run all cells using the original hash-verified Drive data. Each notebook embeds its experiment sources and clones the pinned original study; its recorded source hashes identify the actual code used, independently of later main-branch changes.

V3 tests cached head casts, midpoint force reuse, split attention/MLP backward graphs, 8192-token MLP recomputation tiles, optional fused RMSNorm and expandable allocator segments. Attention receives the full physical batch; output/MLP token tiling is shared with ordinary baseline controls. CPU checkpoint staging overhead is reported separately. FlashAttention-3 is not compatible with T4; the notebook diagnoses native SDPA backends.

Local validation: 83 tests passed, 5 CUDA tests skipped; 18 fresh-process CPU smoke passes, full-model CPU numerical gate and MLP tile-boundary gate passed. All seven v3 embedded sources match repository bytes; all four notebooks compile; all 25 frozen original source hashes remain unchanged. Four RMSNorm kernels compile offline for T4. V3 GPU correctness, capacity, throughput and full-run loss remain unverified until Colab execution.

See [capacity audit](../docs/capacity-results-review-2026-10-10.md) and [v3 local verification](../docs/analysis/midpoint-v3-local-verification.json). Preserve every failure and distinguish the 5% exploratory results from the 10% primary comparison. Review optimizations before selecting the final largest-batch experiment. The future full-budget acceptance ceiling remains 5.56229485.
