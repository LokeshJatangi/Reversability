# Reversibility experiment documentation

All four full runs completed exactly 50M targets; source/data/checkpoints have been audited. The [core report](core-report-2026-10-06.md) is submitted for Admin review, with [machine-readable evidence](analysis/core-audit-2026-10-06.json) and [all loss trajectories](figures/core-loss-2026-10-06.svg). Midpoint improves memory/capacity but remains slower than baseline; Euler fails the loss cutoff. No utilization or large-scale savings claim is established. Optional work remains inactive pending acceptance or reprioritization.

Recommended reading order:

1. [Working rules](../AGENTS.md): session preference, model scale, exact training budget, and correctness requirements.
2. [Progress](../Progress.md): confirmed decisions, inherited findings, completed work, blockers, and next steps.
3. [Colab handoff](colab-handover.md): named dataset, smoke, full-run, and resume commands.
4. [Experiment plan](experiment-plan.md): controlled baseline and reversible comparisons, token accounting, validation, and measurement.
5. [Research roadmap](research-roadmap.md): follow-up studies and the evidence needed for larger-scale claims.
6. [Baseline artifact audit](baseline-artifact-audit-2026-10-01.md): completed-run measurements, hashes and limitations.
7. [Reversible methods](reversible-methods.md): equations, study choices and precision-specific correctness policy.
8. [Matched-run audit](matched-run-audit-2026-10-04.md): measured losses/memory/time, restart history, CUDA acceptance, artifact integrity and planning-review findings.
9. [Core assignment report](core-report-2026-10-06.md): maximum-batch results, effective-batch confound, all run/probe/failure accounting, cost scenario and evidence limits.

The agreed target is approximately 20M trainable parameters with exactly 50,000,000 FineWeb-Edu training targets per full run. The [reversible Colab notebook](https://colab.research.google.com/github/LokeshJatangi/Reversability/blob/main/notebooks/reversible_colab.ipynb) remains available for reproduction; completed runs do not need repeating. Keep the original Drive artifacts and frozen source intact. Measured outcomes and acceptance status are recorded in Progress.md and the core report.
