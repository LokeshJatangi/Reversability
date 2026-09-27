# Colab handoff

The recommended entry point is the [GitHub-hosted Colab notebook](https://colab.research.google.com/github/LokeshJatangi/Reversability/blob/main/notebooks/baseline_colab.ipynb). Select a GPU runtime and run all cells. The notebook clones or refreshes `https://github.com/LokeshJatangi/Reversability.git` under `/content/Reversability`, then performs setup, tests, dataset preparation, a fresh-process batch-capacity search, smoke acceptance, and the full baseline in order. The source checkout is ephemeral; no repository copy is required in Drive.

All irreplaceable artifacts are written directly to `MyDrive/Reversability/artifacts`: frozen data, tokenizer assets and hashes, capacity-search attempts, environment logs, source hashes, live console logs, structured JSONL metrics, configs, latest and best checkpoints, and final summaries. On a new runtime, the notebook recopies the frozen data to local disk for training speed and automatically resumes an incomplete run from the Drive checkpoint. Google Drive for desktop can mirror that known folder to the laptop without a browser download dialog.

The cells below remain the manual equivalent. Run them from the repository root in a Colab GPU runtime. Dataset preparation uses streaming and writes about 102 MB for training plus 2 MB for validation; Hugging Face caches and transient network usage require additional disk space.

## 1. Environment setup

```bash
!pip install -q -r requirements.txt
```

Restart the runtime if Colab asks. Then return to the repository root.

## 2. Offline correctness tests

```bash
!python -m pytest -q
```

Expected outcome: six tests pass. These tests use synthetic data and do not download FineWeb-Edu or train an experiment. They include exact interrupted/resumed agreement at an optimizer-update boundary.

## 3. Dataset preparation: `data_fineweb_edu_gpt2_50m_v1`

```bash
!python scripts/prepare_fineweb_edu.py \
  --output-dir data/fineweb_edu_gpt2_50m_v1 \
  --train-targets 50000000 \
  --validation-targets 1000000
```

Keep `data/fineweb_edu_gpt2_50m_v1/manifest.json` with the binary files. The command refuses to overwrite a nonempty output directory. Successful completion records resolved dataset and tokenizer revisions, file hashes, target counts, source document count, software versions, and elapsed preparation time.

## 4. Baseline batch-capacity search

Each candidate runs in a fresh process. It initializes Adam state, performs three repeated complete optimizer updates, and must retain 10% reserved-VRAM headroom:

```bash
!python scripts/benchmark_batch_capacity.py \
  --config configs/baseline.json \
  --output runs/baseline_batch_capacity.json \
  --max-batch 256 \
  --repetitions 3 \
  --headroom-fraction 0.10
!python - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('configs/baseline.json').read_text())
capacity = json.loads(Path('runs/baseline_batch_capacity.json').read_text())
physical = capacity['selected_physical_batch_size']
nominal_effective = config['physical_batch_size'] * config['gradient_accumulation_steps']
candidates = {max(1, nominal_effective // physical), max(1, -(-nominal_effective // physical))}
accumulation = min(candidates, key=lambda value: abs(physical * value - nominal_effective))
config['physical_batch_size'] = physical
config['gradient_accumulation_steps'] = accumulation
config['capacity_search_report'] = 'runs/baseline_batch_capacity.json'
Path('configs/baseline.runtime.json').write_text(json.dumps(config, indent=2) + '\n')
print({'physical_batch': physical, 'accumulation': accumulation,
       'effective_batch': physical * accumulation})
PY
```

Do not change `configs/baseline.runtime.json` after a checkpoint exists. The Drive-backed notebook enforces this automatically.

## 5. Validation run: `baseline_20m_fineweb_edu_50m_v1_smoke`

Make a temporary smoke configuration so its artifacts cannot overwrite the full run:

```bash
!python - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('configs/baseline.runtime.json').read_text())
config['experiment_name'] = 'baseline_20m_fineweb_edu_50m_v1_smoke'
config['output_dir'] = 'runs/baseline_20m_fineweb_edu_50m_v1_smoke'
config['eval_interval'] = 2
config['checkpoint_interval'] = 2
Path('/tmp/baseline_smoke.json').write_text(json.dumps(config, indent=2) + '\n')
PY
!python scripts/train_baseline.py \
  --config /tmp/baseline_smoke.json \
  --max-targets 65536 \
  --device cuda
```

This is a partial validation run, not the full experiment. Confirm that the startup JSON event reports the smoke name and parameter count, the committed target count ends at exactly 65,536, validation loss is finite, and `runs/baseline_20m_fineweb_edu_50m_v1_smoke/latest.pt` exists. `metrics.jsonl` is flushed throughout the run.

## 6. Full experiment: `baseline_20m_fineweb_edu_50m_v1`

```bash
!mkdir -p runs/baseline_20m_fineweb_edu_50m_v1
!python scripts/train_baseline.py \
  --config configs/baseline.runtime.json \
  --device cuda 2>&1 | tee runs/baseline_20m_fineweb_edu_50m_v1/console.log
```

The full run completes only when `committed_targets` is exactly `50000000`, `completed_full_target_budget` is true, and final validation loss is recorded. Physical batch size comes from the frozen capacity report; effective batch size is the selected physical batch times the frozen accumulation value. The final update is target-masked to end at the exact budget. FP16 gradient-scaler state is checkpointed, and overflowed updates retry the same uncommitted target window.

If Colab interrupts after a checkpoint, resume with:

```bash
!python scripts/train_baseline.py \
  --config configs/baseline.runtime.json \
  --resume runs/baseline_20m_fineweb_edu_50m_v1/latest.pt \
  --device cuda 2>&1 | tee -a runs/baseline_20m_fineweb_edu_50m_v1/console.log
```

When using the notebook, no end-of-session copy is needed: these files already live in Drive and the dataset has a persistent Drive copy. Do not use the smoke checkpoint to resume the full experiment because its configuration hash and target budget differ.
