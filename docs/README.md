# Reversibility experiment documentation

This repository contains dataset preparation and ordinary-autograd baseline code. FineWeb-Edu download, hardware benchmarking, and training remain pending for the user to run in Colab; no experimental performance results exist.

Recommended reading order:

1. [Working rules](../AGENTS.md): session preference, model scale, exact training budget, and correctness requirements.
2. [Progress](../Progress.md): confirmed decisions, inherited findings, completed work, blockers, and next steps.
3. [Colab handoff](colab-handover.md): named dataset, smoke, full-run, and resume commands.
4. [Experiment plan](experiment-plan.md): controlled baseline and reversible comparisons, token accounting, validation, and measurement.
5. [Research roadmap](research-roadmap.md): follow-up studies and the evidence needed for larger-scale claims.

The agreed target is approximately 20M trainable parameters with exactly 50,000,000 FineWeb-Edu training targets per full run. The former English/code/math mixture is retired. The reversible Euler implementation awaits the supplied paper. Treat all experiments as pending until their commands, configuration, artifacts, and measured outcomes are recorded in Progress.md.
