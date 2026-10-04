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

Expected outcome with the current source: 28 tests pass. These tests use synthetic data and do not download FineWeb-Edu or train a full experiment. They include reconstruction/gradient checks, exact interrupted/resumed agreement at an optimizer-update boundary for the baseline and both reversible variants, baseline dataset discovery/hash rejection/recovery, and checkpoint-free baseline reference checks.

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

## Next stage: matched reversible runs

The baseline has completed and been [audited](baseline-artifact-audit-2026-10-01.md). Preserve its artifacts and open the [reversible Colab notebook](https://colab.research.google.com/github/LokeshJatangi/Reversability/blob/main/notebooks/reversible_colab.ipynb). Select a Tesla T4 runtime and run all cells. It clones GitHub source by default; an old `reversible-source.zip` in Drive is ignored unless the explicit offline-bundle flag is enabled.

The notebook requires these existing Drive paths under `MyDrive/Reversability/artifacts`:

- `configs/baseline_colab.json`;
- `runs/baseline_20m_fineweb_edu_50m_v1/`, including metrics and summary; baseline weights are optional in Colab and remain preserved locally;
- `data/data_fineweb_edu_gpt2_50m_v1/`, including the exact baseline manifest, binaries and tokenizer assets.

If these folders have moved or an exported bundle is nested elsewhere in `MyDrive`, the dataset cell searches for baseline config/run logs there. `scripts/stage_baseline_data.py` requires the manifest hash from the baseline startup event and verifies every binary/tokenizer asset before copying. It logs resolved paths in `reversible-v1/data-staging.log`. Existing valid runtime data is reused; other runtime data is preserved when replaced. A newly generated manifest is not a substitute for the recorded baseline manifest.

If the baseline artifacts were deleted from Drive, upload the local **`artifacts/baseline-inputs-recovery.zip` (74 MB)** to **`MyDrive/Reversability`** once, keeping that filename. Upload it to Drive, not `/content/Reversability` (the ephemeral Git checkout). Reopen the updated notebook and run all cells. The dataset cell automatically extracts it into a new `baseline-recovery-*/artifacts` directory, verifies the baseline config/data, and creates `/content/reversibility-data/data_fineweb_edu_gpt2_50m_v1` for training. It reuses the restored folder on later sessions rather than creating duplicate exports. No baseline retraining is required. The ZIP includes frozen data, configs, logs, capacity evidence and original source snapshot; all checkpoint files remain preserved in the local export. The runner checks baseline summary/metrics/final validation agreement without loading baseline weights; checkpoints remain required for newly trained reversible runs.

The small archive contains 21 hash-verified files, totals 74,211,452 bytes, and has SHA-256 `0e5698f03f087625648d84b32171496cc6a560cabc8b4321734e89f6ceabd84c`. Build it with `python scripts/bundle_baseline_reference.py --artifacts <export>/artifacts --output artifacts/baseline-inputs-recovery.zip --without-checkpoints`. The previously supplied full archive is also accepted if already uploaded.

The optional full archive contains 23 independently hash-verified files, including baseline latest/best checkpoints. File size: 528,997,010 bytes (approximately 529 MB); SHA-256: `36c3c448c8d1f3c67bff3dc760d94909abd31705bf18e0f7dbe25bdfcff5f7a6`. To build it, use `python scripts/bundle_baseline_reference.py --artifacts <export>/artifacts --output artifacts/baseline-reference-recovery.zip`. The bundler refuses to overwrite an existing archive.

The runner enforces the recorded baseline environment: T4, FP16, PyTorch 2.11.0+cu128, CUDA runtime 12.8, NumPy 2.1.3 and Python 3.13.15. If Colab supplies different versions, preserve the error log and reconcile the environment before training; changing the comparison controls needs a recorded protocol amendment. The local five-asset data manifest is not a replacement for the actual two-asset Colab manifest.

CUDA reconstruction/gradient/optimizer gates run first, followed by separate 65,536-target smokes and exactly 50M-target midpoint and Euler runs. Both use physical batch 29, accumulation 2, effective batch 58, seed 1337 and the frozen baseline target/evaluation stream. No baseline rerun is needed. A correctness failure stops the workflow; CPU success does not establish FP16 GPU acceptance.

Intermediate console logs, structured metrics, immutable configs/source snapshots, latest/best checkpoints and the review proposal are saved under `artifacts/reversible-v1` as work proceeds. Rerunning the notebook resumes incomplete training from the latest checkpoint and validates already completed runs.

After both matched runs, share the `reversible-v1` artifacts for the required planning session. Review loss trajectories, correctness, throughput, memory, failures and remaining Colab budget. Record the decision before enabling the maximum cell. The predeclared loss cutoff is 5.562294847167969; a method must also pass correctness. The assignment finishes after the selected maximum-batch experiment and the full comparison report.
