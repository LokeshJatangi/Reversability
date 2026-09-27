# FineWeb-Edu reversible-training study

This repository prepares a frozen FineWeb-Edu token stream and trains an ordinary-autograd decoder baseline before any reversible variants are implemented. No full experiment has been run in this repository.

## Named stages

| Name | Type | Purpose |
| --- | --- | --- |
| `data_fineweb_edu_gpt2_50m_v1` | Dataset preparation | Stream pinned FineWeb-Edu, tokenize with pinned GPT-2, and write exactly 50M training targets plus 1M validation targets |
| `baseline_20m_fineweb_edu_50m_v1_smoke` | Validation run | Check the Colab environment, data hashes, forward/backward path, checkpoint, and evaluation on a deliberately small target budget |
| `baseline_20m_fineweb_edu_50m_v1` | Full experiment | Train the ordinary-autograd baseline on exactly 50M committed training targets |

The smoke run is not a performance result and must use a separate output directory. The full run configuration is [configs/baseline.json](configs/baseline.json). The recommended execution path is the [GitHub-hosted Colab notebook](https://colab.research.google.com/github/LokeshJatangi/Reversability/blob/main/notebooks/baseline_colab.ipynb), which clones source into ephemeral Colab storage and persists only experiment artifacts to Drive; see [the handoff](docs/colab-handover.md) for details.

## Layout

- `scripts/prepare_fineweb_edu.py`: streaming dataset preparation with pinned revisions, exact counts, binary hashes, and manifest.
- `scripts/train_baseline.py`: exact-target baseline training, validation, atomic checkpoints, and resume guards.
- `src/reversibility/model.py`: approximately 20M-parameter decoder-only Transformer.
- `tests/test_baseline.py`: offline shape, gradient, target-count, parameter-count, and split-construction tests.
- `notebooks/baseline_colab.ipynb`: persistent, auto-resuming Colab orchestration with a fresh-process GPU batch-capacity search and live Drive logs.
- `scripts/benchmark_batch_capacity.py`: repeatedly tests complete optimizer updates in isolated processes and enforces reserved-VRAM headroom.
- `docs/experiment-plan.md`: comparison and measurement protocol.
- `Progress.md`: verified status and pending work.
