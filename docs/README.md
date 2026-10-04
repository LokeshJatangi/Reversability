# Reversibility experiment documentation

The ordinary-autograd baseline is complete at exactly 50M targets and its imported logs/checkpoints have been audited. Midpoint and Hamiltonian Euler implementations pass CPU gates; CUDA gates and reversible training remain pending.

Recommended reading order:

1. [Working rules](../AGENTS.md): session preference, model scale, exact training budget, and correctness requirements.
2. [Progress](../Progress.md): confirmed decisions, inherited findings, completed work, blockers, and next steps.
3. [Colab handoff](colab-handover.md): named dataset, smoke, full-run, and resume commands.
4. [Experiment plan](experiment-plan.md): controlled baseline and reversible comparisons, token accounting, validation, and measurement.
5. [Research roadmap](research-roadmap.md): follow-up studies and the evidence needed for larger-scale claims.
6. [Baseline artifact audit](baseline-artifact-audit-2026-10-01.md): completed-run measurements, hashes and limitations.
7. [Reversible methods](reversible-methods.md): equations, study choices and precision-specific correctness policy.

The agreed target is approximately 20M trainable parameters with exactly 50,000,000 FineWeb-Edu training targets per full run. The next execution entry point is the [reversible Colab notebook](https://colab.research.google.com/github/LokeshJatangi/Reversability/blob/main/notebooks/reversible_colab.ipynb). Keep the baseline Drive artifacts and use the same GPU/software; the workflow stops before training if controls differ or correctness fails. Measured outcomes and pending stages are recorded in Progress.md.
