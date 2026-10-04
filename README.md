# FineWeb-Edu reversible-training study

The ordinary-autograd baseline completed exactly 50,000,000 FineWeb-Edu targets on a Tesla T4: 20,340,736 parameters, final/best validation loss 5.4623, 18m39s training-process time, 44,679 valid targets/s including evaluation/checkpoints, and 10.03 GiB peak allocated GPU memory. See the [artifact audit](docs/baseline-artifact-audit-2026-10-01.md) for evidence and measurement limits.

Midpoint and Hamiltonian Euler implementations are ready with CPU correctness checks. Their CUDA gates and training runs remain pending. The assignment is incomplete until the matched reversible runs, planning review, selected maximum-batch run, and comparison report are finished.

## Named stages

| Name | Type | Purpose |
| --- | --- | --- |
| `data_fineweb_edu_gpt2_50m_v1` | Dataset preparation | Stream pinned FineWeb-Edu, tokenize with pinned GPT-2, and write exactly 50M training targets plus 1M validation targets |
| `baseline_20m_fineweb_edu_50m_v1_smoke` | Validation run | Check the Colab environment, data hashes, forward/backward path, checkpoint, and evaluation on a deliberately small target budget |
| `baseline_20m_fineweb_edu_50m_v1` | Full experiment | Train the ordinary-autograd baseline on exactly 50M committed training targets |
| `<midpoint/euler>_20m_fineweb_edu_50m_v1_matched` | Pending full experiments | Train each accepted reversible variant on 50M targets at the baseline physical batch 29 and effective batch 58 |
| `<selected>_20m_fineweb_edu_50m_v1_maximum` | Pending full experiment | After the planning review, capacity-test and train the selected variant at its largest verified physical batch |

The smoke run is not a performance result and must use a separate output directory. The full run configuration is [configs/baseline.json](configs/baseline.json). The recommended execution path is the [GitHub-hosted Colab notebook](https://colab.research.google.com/github/LokeshJatangi/Reversability/blob/main/notebooks/baseline_colab.ipynb), which clones source into ephemeral Colab storage and persists only experiment artifacts to Drive; see [the handoff](docs/colab-handover.md) for details.

For the next stage, open the [reversible Colab notebook](https://colab.research.google.com/github/LokeshJatangi/Reversability/blob/main/notebooks/reversible_colab.ipynb), select a T4, and run all cells with the existing baseline Drive artifacts intact. It checks the baseline GPU/software/data, runs CUDA correctness gates and separate smokes, trains both matched variants, and writes the review proposal. Results and intermediate logs go directly to `MyDrive/Reversability/artifacts/reversible-v1`. It pauses before maximum-batch training for the required planning session. See the [reversible handoff](docs/colab-handover.md#next-stage-matched-reversible-runs) and [method specification](docs/reversible-methods.md).

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
