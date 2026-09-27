# Progress

Last updated: 2026-09-28.

## Confirmed decisions

- Session preference: **fully autonomous**; foundation creation took place in Default mode.
- Model target: approximately **20M trainable parameters**. The implemented baseline uses 7 layers, width 256, 8 heads, a 1024-unit SwiGLU MLP, tied GPT-2 embeddings, and a 512-token context. Its exact verified count is **20,340,736** trainable parameters.
- Every full run uses **exactly 50,000,000 FineWeb-Edu training targets**. The former 25M English, 12.5M code, and 12.5M math mixture is retired by user direction.
- Dataset preparation is pinned to `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, requested revision `v1.0.0`, and the GPT-2 tokenizer commit encoded in the preparation script. Resolved commits and binary hashes will be frozen in the generated manifest.
- Experiment order: baseline; Euler and midpoint at the baseline physical and effective batch sizes; selected reversible variant at its maximum feasible physical batch size under a fixed memory budget.
- Initial execution venue: Colab, as recommended in the supplied plan. Hardware availability, memory, runtime limits, and suitability have not been measured.
- The primary reversible specification is now supplied: *Reversing Large Language Models for Efficient Training and Fine-Tuning* (arXiv:2512.02056). The assignment's midpoint maps to Eq. 2.4 and “Euler/oiler” maps to the Hamiltonian staggered update resembling symplectic Euler in Eqs. 2.8–2.9; implementation and independent validation remain pending.
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
- Added a Drive-backed [Colab notebook](notebooks/baseline_colab.ipynb) that prepares and verifies data, runs a fresh-process batch-capacity search with 10% reserved-VRAM headroom, freezes physical/effective batch settings, runs the smoke gate, projects runtime from measured throughput, and automatically starts or resumes the full baseline.
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

**No training experiments have run and no performance results exist.** FineWeb-Edu preparation is complete locally, but there are no measured training losses, throughput values, peak-memory values, model checkpoints, or training logs. Dataset preparation and offline synthetic correctness tests are validation evidence, not experimental performance evidence.

## Pending work and blockers

| Work | Status / prerequisite |
| --- | --- |
| Baseline implementation | Implemented and offline tests pass; Colab smoke validation remains pending |
| Frozen data and evaluation manifest | Complete and independently hash-validated locally; notebook will validate a bundled copy or prepare/persist its own Drive copy |
| Notebook reuse | Pending: locate the inspected notebook and verify its loss checks and chunked cross-entropy |
| Reversible Euler implementation | Paper supplied; exact symplectic-Euler-like Hamiltonian recurrence and backward implementation still pending correctness work |
| Midpoint implementation | Pending: define the exact formulation and validate states and gradients |
| Colab smoke validation | Pending: run `baseline_20m_fineweb_edu_50m_v1_smoke` through the persistent notebook |
| Hardware and batch-size benchmarking | Harness complete; measured capacity search remains pending on the assigned Colab GPU |
| Full baseline training | Pending: user runs `baseline_20m_fineweb_edu_50m_v1` after dataset and smoke checks |
| Reversible comparisons | Pending on paper, implementation, and correctness gates |

## Next steps

1. Copy the repository to `MyDrive/Reversability` (including the validated `data/fineweb_edu_gpt2_50m_v1` directory if convenient), open [the Colab notebook](notebooks/baseline_colab.ipynb), select a GPU runtime, and run all cells. It installs, tests, verifies or prepares the dataset, measures baseline batch capacity, and runs the named smoke validation with persistent artifacts.
2. Let the notebook start `baseline_20m_fineweb_edu_50m_v1` only after the smoke evidence is accepted; preserve its manifest, frozen runtime configuration, capacity report, console log, metrics, summaries, and latest/best checkpoints in Drive.
3. After the baseline completes, document and correctness-test the supplied paper's midpoint and symplectic-Euler-like Hamiltonian formulations before any reversible full run.

For each future work entry, record the date, completed work, exact commands and configuration, artifacts, findings, blockers, and next steps. For experiments also include hardware, precision, seed, target counters, and whether the run is a probe, partial run, or completed full run.
