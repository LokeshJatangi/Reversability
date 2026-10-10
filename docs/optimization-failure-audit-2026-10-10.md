# Uploaded optimization failure audit — 2026-10-10

The archive contains only a failed correctness gate and orchestration evidence. **No capacity, throughput or CPU-checkpoint experiment ran.** The last verified reversible maximum remains physical batch33 under the original protocol.

Archive CRC passed; all five extracted files are byte-identical to ZIP contents. Embedded source hashes match the delivered v1 notebook. Reported frozen core, configuration and data-manifest hashes match the study; all25 frozen source files remain unchanged. Runtime: T4, Torch2.11.0+cu128, NumPy2.1.3, Python3.13.15, device memory15,637,086,208bytes.

Twelve numerical rows cover baseline/midpoint, chunks1024/2048/4096 and two updates. All loss-relative, gradient-relative-L2 and whole-update-relative-L2 limits passed. The newly added parameter-coordinate limit2e-5 alone rejected the sweep. Worst errors:

| Metric | Maximum | Old limit | Passed all rows |
|---|---:|---:|---|
| gradient_relative_l2 | 0.0096782353 | 0.02 | True |
| loss_relative | 2.9090384e-06 | 0.002 | True |
| parameter_max_absolute | 0.0021499174 | 2e-05 | False |
| optimizer_relative_update_l2 | 0.033852206 | 0.05 | True |

This is a gate-policy error, not evidence that batch50 OOMed. The frozen project already declares FP16 optimizer coordinate tolerance1.2e-3 and whole-update relative L2.05. V2 compares **each step's update increment** for the coordinate bound; it reports cumulative parameter divergence separately and still bounds cumulative divergence relative to an update. The old artifact lacks pre-step tensors, so its second-step increment cannot be reconstructed. It is preserved as failed; no retrospective GPU pass is claimed.

V2 also replaces accumulation2 with exactly one full physical batch per update, including the physical29 baseline references. Output-token chunks bound only the vocabulary head/loss workspace; the Transformer receives the entire physical batch. Numerical feature gates, memory-first capacity search and fresh confirmations precede any50M quality run. The10% reserved-memory margin remains fixed. Validation-loss acceptance remains baseline5.46229485 + .10 = **5.56229485**. Speed and savings remain unmeasured.
