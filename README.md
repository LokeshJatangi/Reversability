# FineWeb-Edu reversible-training study

The ordinary-autograd baseline completed exactly 50,000,000 FineWeb-Edu targets on a Tesla T4: 20,340,736 parameters, final/best validation loss 5.4623, 18m39s training-process time, 44,679 valid targets/s including evaluation/checkpoints, and 10.03 GiB peak allocated GPU memory. See the [artifact audit](docs/baseline-artifact-audit-2026-10-01.md) for evidence and measurement limits.

**Core assignment complete; report submitted for Admin review.** All four full runs completed exactly 50M targets and are artifact-audited. Midpoint saves 12.9% peak allocated memory at matched batches and increases verified physical capacity from 29 to 33, but its maximum-batch run remains 12.1% slower than baseline. Euler fails the fixed loss cutoff. No speed/cost saving or full GPU utilization is established. Read the [core report](docs/core-report-2026-10-06.md), [audit evidence](docs/analysis/core-audit-2026-10-06.json) and [loss trajectories](docs/figures/core-loss-2026-10-06.svg). Optional optimization/scaling work remains inactive pending acceptance or reprioritization.

## Named stages

| Name | Type | Purpose |
| --- | --- | --- |
| `data_fineweb_edu_gpt2_50m_v1` | Dataset preparation | Stream pinned FineWeb-Edu, tokenize with pinned GPT-2, and write exactly 50M training targets plus 1M validation targets |
| `baseline_20m_fineweb_edu_50m_v1_smoke` | Validation run | Check the Colab environment, data hashes, forward/backward path, checkpoint, and evaluation on a deliberately small target budget |
| `baseline_20m_fineweb_edu_50m_v1` | Full experiment | Train the ordinary-autograd baseline on exactly 50M committed training targets |
| `<midpoint/euler>_20m_fineweb_edu_50m_v1_matched` | Completed full experiments | Each variant trained on 50M targets at physical batch 29 and effective batch 58; audited |
| `midpoint_20m_fineweb_edu_50m_v1_maximum` | Complete: 50M targets | Verified physical batch 33, effective batch 66; final loss 5.5091; changed effective batch disclosed |

The smoke run is not a performance result and must use a separate output directory. The full run configuration is [configs/baseline.json](configs/baseline.json). The recommended execution path is the [GitHub-hosted Colab notebook](https://colab.research.google.com/github/LokeshJatangi/Reversability/blob/main/notebooks/baseline_colab.ipynb), which clones source into ephemeral Colab storage and persists only experiment artifacts to Drive; see [the handoff](docs/colab-handover.md) for details.

The [reversible Colab notebook](https://colab.research.google.com/github/LokeshJatangi/Reversability/blob/main/notebooks/reversible_colab.ipynb) remains a reproducibility entry point, not a request to rerun completed experiments. It enforces the recorded T4/software/data and correctness controls and skips completed training. The completed study's actual Drive root is `MyDrive/Reversability/baseline-recovery-7bldlite/artifacts/reversible-v1`; preserve it. See the [historical reversible handoff](docs/colab-handover.md#next-stage-matched-reversible-runs), [method specification](docs/reversible-methods.md) and completed core report.

## Layout

- `scripts/prepare_fineweb_edu.py`: streaming dataset preparation with pinned revisions, exact counts, binary hashes, and manifest.
- `scripts/train_baseline.py`: exact-target baseline training, validation, atomic checkpoints, and resume guards.
- `src/reversibility/model.py`: approximately 20M-parameter decoder-only Transformer.
- `src/reversibility/reversible.py`: midpoint and Hamiltonian Euler with reconstructed backward passes and stored-state references.
- `scripts/validate_reversible.py`: precision-specific reconstruction, gradient, and optimizer-update gates.
- `scripts/run_reversible_study.py`: persistent matched, review, and maximum phases with frozen controls.
- `tests/test_baseline.py`: offline shape, gradient, target-count, parameter-count, and split-construction tests.
- `notebooks/baseline_colab.ipynb`: persistent, auto-resuming Colab orchestration with a fresh-process GPU batch-capacity search and live Drive logs.
- `notebooks/reversible_colab.ipynb`: GitHub-backed reversible workflow using the completed baseline artifacts.
- `scripts/benchmark_batch_capacity.py`: repeatedly tests complete optimizer updates in isolated processes and enforces reserved-VRAM headroom.
- `docs/experiment-plan.md`: comparison and measurement protocol.
- `Progress.md`: verified status and pending work.
