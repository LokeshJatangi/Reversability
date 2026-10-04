# Progress

Last updated: 2026-10-04.

## Confirmed decisions

- Session preference: **fully autonomous**; foundation creation took place in Default mode.
- Model target: approximately **20M trainable parameters**. The implemented baseline uses 7 layers, width 256, 8 heads, a 1024-unit SwiGLU MLP, tied GPT-2 embeddings, and a 512-token context. Its exact verified count is **20,340,736** trainable parameters.
- Every full run uses **exactly 50,000,000 FineWeb-Edu training targets**. The former 25M English, 12.5M code, and 12.5M math mixture is retired by user direction.
- Dataset preparation is pinned to `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, requested revision `v1.0.0`, and the GPT-2 tokenizer commit encoded in the preparation script. Resolved commits and binary hashes will be frozen in the generated manifest.
- Experiment order: baseline; Euler and midpoint at the baseline physical and effective batch sizes; selected reversible variant at its maximum feasible physical batch size under a fixed memory budget.
- Execution venue: Colab. The completed baseline ran on a Tesla T4 with FP16; its capacity, timing and memory measurements are recorded below. Reversible runs require the same recorded hardware/software controls.
- Primary reversible specification: *Reversing Large Language Models for Efficient Training and Fine-Tuning* (arXiv:2512.02056). Midpoint maps to Eq. 2.4 and “Euler/oiler” to the Hamiltonian staggered update resembling symplectic Euler in Eqs. 2.8–2.9. Both are implemented with passing CPU correctness gates; FP16 CUDA acceptance remains pending.
- Moonwalk (arXiv:2402.14212) is an optional post-core feasibility study, not a replacement for the required midpoint/Euler runs.
- RevFFN, Moonwalk, leapfrog, scaling grids, and 70B engineering are explicitly deferred until the six-stage core assignment is complete and its report is accepted or the Admin reprioritizes the work. The active 20M experiment remains a dense Transformer.
- A recorded planning checkpoint will occur after the baseline and matched-batch midpoint/Euler runs, before the selected reversible maximum-batch run and scaling extensions.

## Inherited findings

The supplied plan reports that an old notebook was inspected and that its loss checks and chunked cross-entropy are reusable. The notebook is not present in this workspace, so this session has not independently verified that finding. Obtain its path and validate equivalence before incorporating code.

## Completed work

- Inspected the workspace and checked for ancestor working instructions; no existing project documents or notebook were found.
- Created [working rules](AGENTS.md), the [documentation index](docs/README.md), the [experiment plan](docs/experiment-plan.md), and the [research roadmap](docs/research-roadmap.md).
- Defined target accounting, comparison controls, correctness gates, checkpoint/resume requirements, and reporting requirements in the experiment plan.
- Implemented the named `data_fineweb_edu_gpt2_50m_v1` streaming preparation stage with exact train/validation target counts and a hashed manifest.
- Implemented `baseline_20m_fineweb_edu_50m_v1`: an ordinary-autograd decoder, exact committed-target accounting, partial-final-batch masking, validation, atomic checkpointing, and guarded resume.
- Added offline tests and a [Colab handoff](docs/colab-handover.md) with separately named smoke and full experiments.
- Added a GitHub-cloning, Drive-backed [Colab notebook](notebooks/baseline_colab.ipynb) that refreshes source under `/content`, prepares and verifies persistent Drive data, runs a fresh-process batch-capacity search with 10% reserved-VRAM headroom, freezes physical/effective batch settings, runs the smoke gate, projects runtime from measured throughput, and automatically starts or resumes the full baseline.
- Frozen the baseline precision to FP16 for common Colab GPU compatibility. Gradient-scaler state is checkpointed; overflowed updates retain the target cursor and retry with restored window-entry random state.
- Added durable `metrics.jsonl`, run summaries, hardware/software metadata, peak CUDA memory fields, individual tokenizer-asset hashes, and a deterministic interrupted/resumed equivalence test.
- Prepared and independently validated the local frozen dataset artifact `data/fineweb_edu_gpt2_50m_v1`: exactly 50,000,000 training targets and 1,000,000 validation targets from 48,464 source documents. Resolved FineWeb-Edu revision: `fc9850dff5e2d0f8f776efe41b24a1c49556cfc5`; GPT-2 tokenizer revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`; training SHA-256: `8d71872e4d7024801e459a6c44942907a89ae946105eb0905968746a0652c738`; validation SHA-256: `d5058ea280216a4504ac6c899028adfd1c5b236c3af9de5ff813dfc3e5ed61b1`. All five tokenizer assets also passed byte-size and SHA-256 validation.
- Added `scripts/benchmark_batch_capacity.py` and integrated it into the Colab notebook. Each candidate runs in a fresh process, initializes Adam state, completes three repeated updates, and must retain 10% reserved-VRAM headroom. No GPU capacity measurement has run yet.

## Verification and commands

Workspace inspection commands: `pwd`, `rg --files -g 'AGENTS.md' -g 'Progress.md' -g '*.md' -g '*.ipynb' -g '!node_modules' -g '!.git'`, and `ls -la`. The file search returned no matches before creation.

Foundation verification passed: all five expected files are nonempty, all 16 local Markdown links resolve, no trailing whitespace was found, domain quotas sum to 50M, and the autonomy, paper prerequisite, exact-budget, and no-results statements are present. The check ran with `python3 - <<'PY'` using `pathlib` assertions and a Markdown-link regular expression. This reproducible file/link check also passed:

```bash
python3 - <<'PY'
from pathlib import Path
import re
files = [Path(p) for p in ('AGENTS.md', 'Progress.md', 'docs/README.md', 'docs/experiment-plan.md', 'docs/research-roadmap.md')]
links = 0
for path in files:
    assert path.is_file() and path.stat().st_size > 0, path
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text()):
        assert (path.parent / target.split('#')[0]).is_file(), (path, target)
        links += 1
print(f'PASS: {len(files)} files and {links} local links')
PY
```

The first offline test run found that the 8-layer draft had 21,389,824 parameters, outside the test's 19M–21M acceptance band. The configuration was changed to 7 layers and revalidated at 20,340,736 parameters. On 2026-09-28, `python3 -m pytest -q` passed 6 tests in 4.61 seconds, including bit-exact model and optimizer agreement between uninterrupted and optimizer-boundary-resumed CPU runs; Python compilation and notebook-cell compilation passed. The test process emitted one environment warning about the installed Requests dependency versions; it did not affect the offline tests. Dataset preparation wrote a valid, fully hash-verified artifact, then the local Python 3.12 process aborted during third-party extension finalization; this occurred after the manifest and all files were flushed and is recorded as an environment anomaly, not a training result. No smoke or full training experiment has run.

## Results and artifacts

The imported Colab artifacts were audited on 2026-10-01. **The baseline completed exactly 50,000,000 training targets**, with final validation and both final/best checkpoints intact. The run timestamps are 2026-09-27 18:58:53–19:17:32 UTC; the export folder date is not the experiment date. Earlier no-results statements above describe historical local verification before these artifacts were available.

- Artifact root: `artifacts-20260930T183134Z-1-001/artifacts` (ignored by Git; preserve separately).
- Tesla T4, FP16, PyTorch 2.11.0+cu128, CUDA runtime 12.8; 20,340,736 trainable parameters; seed 1337; context 512.
- Capacity search selected physical batch **29**, with three repeated complete updates and 10% reserved-VRAM headroom. Batch 30 failed the headroom gate. Accumulation **2**, effective batch **58** sequences, regular update **29,696** valid targets.
- Full baseline: **1,684 optimizer updates**, final partial update **21,632** targets, final/best validation loss **5.462294847167969** on **1,000,000** held-out targets; final training-window loss **5.472444974459135**.
- Measured training-process elapsed time **1,119.090 seconds** (18m39s), including validation/checkpoint overhead; **44,679.159 valid targets/s** end to end. Peak allocated/reserved CUDA memory **10.030/12.766 GiB**. This is total GPU allocation, not isolated activation memory or a separately timed steady-state throughput result.
- Smoke completed its separate **65,536-target** budget in three updates, with finite validation loss **10.6698841640625**. It is excluded from the full-run budget. Six offline tests passed on Colab.
- One full-run startup, no logged overflow retries or resumed segments, and no failure/truncation in the full training logs. Both checkpoints load on CPU, contain the 50M cursor and complete optimizer/scaler/RNG fields, and have finite model tensors. Dataset binaries, both recorded tokenizer assets, config/manifest hashes, and all 18 files in the source snapshot passed independent verification.
- Detailed evidence, commands, validation trajectory, hashes, and measurement limitations: [baseline artifact audit](docs/baseline-artifact-audit-2026-10-01.md).

No restart or resume of this completed baseline is needed. Monetary cost and Colab compute-unit use were not supplied; no estimate is presented as a measurement.

## Pending work and blockers

| Work | Status / prerequisite |
| --- | --- |
| Baseline implementation | Implemented; local and Colab offline tests pass; Colab smoke accepted |
| Frozen data and evaluation manifest | Complete and independently hash-validated locally; notebook will validate a bundled copy or prepare/persist its own Drive copy |
| Older notebook reuse | Unverified inherited suggestion; not required for the implemented full-vocabulary loss pipeline |
| Reversible Euler implementation | Eqs. 2.8–2.9 implemented with inverse backward and ordinary-autograd reference; CPU gates pass, CUDA gates pending |
| Midpoint implementation | Eqs. 2.4–2.5 implemented, h=0.5, with inverse backward/reference; CPU gates pass, CUDA gates pending |
| Colab smoke validation | Complete: 65,536 targets; finite validation and preserved checkpoint |
| Hardware and batch-size benchmarking | Complete on Tesla T4: batch 29 passes, batch 30 fails 10% headroom |
| Full baseline training | Complete and audited: exactly 50M targets, final validation/checkpoints preserved |
| Reversible comparisons | Implemented staged Colab workflow; CUDA correctness/smoke/matched training pending |

## Next steps

1. Preserve the imported baseline artifact bundle and its source snapshot; do not rerun or resume the completed baseline.
2. Open the [reversible Colab notebook](https://colab.research.google.com/github/LokeshJatangi/Reversability/blob/main/notebooks/reversible_colab.ipynb) on a T4 with the baseline Drive artifacts intact. Run all cells to verify GPU/software/data controls and CUDA correctness before training. Methods, CPU gates and loss cutoff are already documented/frozen.
3. Let the notebook run separate smokes and each accepted reversible variant for exactly 50M targets using physical batch 29, accumulation 2, effective batch 58, and the frozen data/evaluation controls. Bring its `artifacts/reversible-v1` results back for review.
4. Hold the required focused planning checkpoint after both matched runs, then finish selected-variant capacity/training and the core report. Deferred extensions remain out of scope.

For each future work entry, record the date, completed work, exact commands and configuration, artifacts, findings, blockers, and next steps. For experiments also include hardware, precision, seed, target counters, and whether the run is a probe, partial run, or completed full run.

## Reversible implementation and verification — 2026-10-01

- User authorized implementing required midpoint/Euler methods and preparing experiments through the selected maximum-batch stage. Preserved the existing fully autonomous preference and scope lock.
- Read the supplied paper's PDF equations 2.4/2.5 and 2.8/2.9 and experimental details. No author code repository was located; nanoGPT is only the cited hyperparameter reference. This is an independent paper-based implementation. See [method specification](docs/reversible-methods.md).
- Implemented a stack-level custom autograd Function in `src/reversibility/reversible.py`. It retains the final hidden-state pair and parameter references, reconstructs one layer at a time, and returns gradients through normal autograd. Both variants share the baseline architecture and exactly **20,340,736** parameters (zero count difference). Explicit boundary choice: both starting states equal the embedding; midpoint h=0.5; Euler unit coefficients. Dropout/compilation/higher-order gradients are not supported by the reconstruction path.
- Added local config templates, an immutable staged study runner, and a separate [Colab notebook](notebooks/reversible_colab.ipynb). It preserves the completed baseline; source, configs, logs, smoke/matched runs, review proposal/decision, capacity and maximum-run artifacts use `artifacts/reversible-v1`. Loss cutoff is frozen at baseline +0.10 nats before reversible runs. GPU, software and exact baseline dataset manifest are checked.
- Capacity probes now exercise the actual configured accumulation, distinguish numerical failures from memory failures, stop on non-memory errors, and persist attempt history. Maximum phase requires both completed matched runs and the recorded planning checkpoint, then independently confirms the chosen accumulation and logs effective-batch/update-count confounds.
- Fixed CUDA checkpoint loading to restore checkpoints through CPU so saved CPU RNG tensors remain suitable for `torch.set_rng_state`; optimizer loading then restores state to parameter devices. Synthetic baseline and both reversible interrupted/resumed equivalence tests pass.
- Verification: `OMP_NUM_THREADS=1 python3 -m pytest -q` → **18 passed in 5.79s**, with the existing Requests version warning. Python and both notebook code-cell compilation passed. `git diff --check` passed.
- `OMP_NUM_THREADS=1 python3 scripts/validate_reversible.py --device cpu --output runs/correctness/reversible_cpu_v2_final.json` → **40 cases passed**, including full-width/context core probes. Maximum relative L2 optimizer-update error: FP64 **9.83e-14**, FP32 **3.38e-5**. These are synthetic correctness measurements, excluded from training targets, not language-model quality/performance results.
- Policy v1 failed the full-width midpoint FP32 post-Adam parameter gate at maximum absolute difference **3.24247e-5**. Failure report remains in `runs/correctness/reversible_cpu.json`. The documented v2 amendment separates optimizer tolerance from state/gradient tolerance and adds a strict global update-L2 check; no reversible training preceded this amendment.
- Local PyTorch 2.9.1+cu128, Python 3.12.2, CUDA availability **false**. No FP16 GPU gate, reversible GPU smoke/full training, selected method, maximum-batch measurement, or cost measurement has run. CPU acceptance must not be described as CUDA acceptance.
- Handoff: `artifacts/reversible-source.zip` bundles current implementation/docs/tests/notebooks and CPU correctness evidence. Place it in `MyDrive/Reversability` and open `notebooks/reversible_colab.ipynb`. Run All executes matched phases, then stops for the required planning review; maximum execution remains disabled until the review is recorded. Source changes have not been committed or published in this session; existing user changes and previous notes remain intact.

## Readiness review and Git handoff — 2026-10-04

- Found the completed baseline artifacts and pending reversible implementation beyond the older session note. Rechecked metrics/summary, both checkpoints' exact 50M target cursor and step 1,684, finite tensors, and checkpoint SHA-256 values; they agree with the baseline audit. No baseline rerun is needed.
- Revalidated the implementation with `OMP_NUM_THREADS=1 python3 -m pytest -q`: **18 passed in 6.19s**; the existing Requests dependency warning remains. Both notebook code-cell compilation and all 29 local links in current top-level documentation passed; `git diff --check` passed.
- Repeated the full CPU correctness grid with `OMP_NUM_THREADS=1 python3 scripts/validate_reversible.py --device cpu --output runs/correctness/reversible_cpu_2026-10-04.json`: **40 cases passed**. Maximum relative optimizer-update L2 errors were FP64 **9.82862e-14** and FP32 **3.38282e-5**. Report SHA-256: `6e7aa36adb76d7747233cce6717ba1d9dada8d6f1f122c46aa51d00a9e388f91`; the report remains in ignored local experiment artifacts.
- Finished the GitHub handoff: the reversible notebook clones GitHub by default, ignoring old Drive ZIPs unless explicitly selected. Immutable study snapshots now also include requirements, all config templates and AGENTS.md. Updated README, documentation index and Colab instructions to reflect completed baseline results and the actual next stage.
- Local CUDA availability remains false; `nvidia-smi` is absent. CUDA gates, matched reversible training, selection/planning review, maximum-batch experiment and final comparison report remain pending. Cost and reversible savings remain unmeasured.
- Git commit includes the pending midpoint/Euler source, configs, tests, staged runner, notebook, method policy, baseline audit and handoff notes. The optional source ZIP is now a historical fallback, not the required execution path.
