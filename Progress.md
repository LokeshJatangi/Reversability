# Progress

Last updated: 2026-10-10. **Stages 1–6 complete; [core report](docs/core-report-2026-10-06.md) submitted for Admin review.** Earlier pending/no-results entries below are historical. The separately authorized GPU profiling phase is complete and audited; v1 GPU optimization failed its added coordinate gate before any capacity probe; corrected physical-capacity v2 and local verification are complete, with GPU v2 capacity now executed and audited; final-quality validation remains pending.

## GPU capacity v2 audited — 2026-10-10

- Uploaded `20261010T154001Z-2382516a-capacity-v2.zip`: CRC/329 members/source/config/data/runtime receipts audited;110 standalone records match suite,108 complete,2 hardware OOM,7 headroom rejections. All7 numerical gates and3 chunks pass. No full50M validation is present.
- Real physical50 confirmed for both models, accumulation1. Selected baseline max285 and midpoint max814 each pass3 fresh5-warmup/20-update confirmations including CPU checkpoint staging; next batches286/815 execute but fail reserved90% margin. These are selected-path maxima, not identical-feature/global maxima. Baseline uses CE+SwiGLU+expandable; midpoint CE-only/default allocator.
- Matched physical50 CE-only/default allocator/no reuse/no fused optimizer comparison: midpoint allocated1.033GiB vsbaseline2.871GiB (64.0% reduction), reserved1.096vs2.955GiB (62.9% reduction), throughput48,666vs60,373 (-19.4%), one probe each. Supports matched memory advantage, not full-run quality.
- Original33 limit was dominated by unbounded vocabulary buffers (~9.49GiB at33); v2 bounds output-head chunks and removes dense loss intermediates while keeping one full Transformer batch. Shared optimization also raises baseline29→285; old accumulation2 vsnew1 disclosed.
- Selected midpoint max814 reserves13.082GiB/89.83%, allocated11.682GiB, median38,647 targets/s. Baseline285 reserves13.096GiB/89.92%, allocated12.035GiB, median58,269. CPU checkpoint copy/write: midpoint.220/5.957s; baseline.247/1.105s, separate overhead. Checkpoint binaries remain Drive-side; export contains roundtrip/hash receipts.
- Optional speed candidate midpoint CE+SwiGLU+force reuse at50:57,479 targets/s single probe; needs fresh repeats. Memory-selected midpoint50 repeats48,553 fail optional no-speed-loss; optimizedbaseline50 repeats63,587 pass.
- Core MLPlayer forward/local reconstructed backward now cause high-batch OOM; allocator slack remains. Next quality phase must test actual50M targets and≤5.56229485;814 yields120 updates versus newbaseline29/accumulation1 3368 (historicalbaseline29/accumulation2 1684). No quality or savings claim yet. [Audit/review](docs/capacity-results-review-2026-10-10.md).

## Physical-capacity v2, gate correction and kernels — 2026-10-10

Supersedes the earlier accumulation2/speed-required/parked-kernel optimization protocol below. Those entries and artifacts are retained as history.

- Goal: highest verified real physical batch, accumulation1, fixed reserved≤90%, then50M-target full training with validation loss≤5.56229485 (+.10); speed optional. Physical50 is a milestone. Same-session baseline physical29 also uses accumulation1; changed update counts and historical accumulation2 differences are disclosed.
- Audited uploaded optimization ZIP: five byte-identical extracted files, CRC/source/config/data/runtime checks pass. Twelve numerical rows pass loss/gradient/whole-update limits; added FP16 coordinate2e-5 alone rejected the gate. No capacity/speed/checkpoint probe ran. Original maximum33 remains last verified. V2 uses frozen FP16 per-step optimizer coordinate1.2e-3 and whole-update L2.05, reports cumulative coordinate drift separately, and requires new GPU evidence.
- Added five experimental tiled CE/SwiGLU kernels; preserves observed Half rounding and gates approximate math independently. Shared head chunking, midpoint force reuse, fused AdamW, kernel combinations and kernel+force reuse each gate independently. Expandable allocator tested in fresh process. CPU checkpoint copy/write recorded separately and included in final capacity confirmation.
- Memory-first selection, progressive growth beyond50, OOM/headroom bracket refinement and three full-duration fresh confirmations. Last confirmation stages CPU checkpoint. Unexpected runtime failures are unresolved searches, not memory boundaries. Search ceiling fits are lower bounds, not absolute maxima. Ordinary baseline receives shared optimization too.
- Executed64 tests passed/4 CUDA skips,19 fresh-process CPU passes, full20M FP32 bounded-context shared/force gates, and offline compilation of all5 kernels for sm75. Actual CUDA execution is pending. Notebook syntax/embedded checksums/isolated CLI imports pass; all25 frozen sources and original profiling script unchanged.
- Deliverable: `notebooks/capacity_colab_v2.ipynb`; new unique Drive `Reversability/capacity-v2`. [Plan](docs/capacity-experiment-plan-2026-10-10.md), [failed-run audit](docs/optimization-failure-audit-2026-10-10.md), [execution](docs/capacity-v2-execution-2026-10-10.md). No capacity50, GPU maximum, final-quality, speed or savings win is claimed before returned evidence.

## Historical v1: target batch-50 optimization implementation and CPU execution — 2026-10-10

- User explicitly authorized optimizing the current bottleneck, fixed10% headroom, target physical50/no speed loss, separate CPU checkpoint overhead, planning and execution. Custom kernel work stays parked; broader scaling/MoE/Moonwalk remains deferred.
- Added `src/reversibility/optimized.py`, `scripts/benchmark_optimized.py`, `tests/test_optimized.py`, self-contained `notebooks/optimization_colab.ipynb`, and experiment/execution documents. Shared head/loss chunking releases each vocabulary graph immediately and performs core backward once per microstep; tied embedding/head gradients and original autocast CE policy are preserved. Midpoint backward reuses one force graph for inversion/VJP rather than evaluating the force twice.
- CPU checkpoint payloads independently clone model/optimizer/scaler/RNG/cursor/config/data metadata onto CPU; copy/write overhead is saved separately in `checkpoint-overhead.json`. Probe snapshots are explicitly marked and do not replace completed full-run checkpoints or production resume state. Frozen trainer remains unchanged.
- Protocol: T4 FP16 context512, batches29/40/50, chunks1024/2048/4096; gated built-in fused AdamW option at50; fixed reserved <=90% during startup/training/checkpoint. Six current-session baseline29 references bracket candidates. Three fresh batch50 finalist repeats per method must all fit, and slowest finalist must meet fastest reference to confirm no speed loss. Compare optimized baseline50 too. Accumulation2 means effective100 versus reference58; no equal-quality or cost claim.
- Executed final **58 tests**, all passed (6.18s); final **18-pass tiny CPU suite**, all complete; full20M FP32 B2/context64 numerical gate passed. Original arbitrary2e-6 optimizer coordinate cap rejected a 7.44e-6 midpoint coordinate despite close gradients. Preserved rejection, documented protocol amendment to inherited frozen v2 FP32 coordinate5e-5 plus whole-update L2.01. Final worst midpoint gradient relative L2 6.35e-6, whole-update relative L21.34e-5; GPU thresholds frozen before any GPU assessment.
- Verified FP64/FP32 midpoint state/VJPs, tied weights/masks/accumulation, CPU checkpoint independence and exact continuation. Notebook syntax/embedded hashes/staging/imports verified against an isolated frozen-source copy. All25 frozen source hashes still match; original profiling harness hash remains unchanged.
- Local CUDA unavailable and connected browser inventory has no Colab session: GPU execution is pending, with no measured batch50 capacity, speed or savings claim. Delivered notebook embeds all new code, needs no push, stages original data, and exports evidence while CPU checkpoint binaries stay separately in Drive.

## GPU profiling audit and bottleneck diagnosis — 2026-10-10

- User supplied `20261010T143701Z-b8bda223-profiling.zip` and its extracted folder. All nine GPU passes completed. Verified ZIP CRC and all 34 members byte-for-byte against extracted evidence, provenance receipts, run config/data hashes, profiling script hash and all 25 frozen execution-source hashes. Runtime exactly matches recorded baseline: T4, Python 3.13.15, Torch 2.11.0+cu128, NumPy 2.1.3 and CUDA 12.8.
- Training-only valid targets/s: baseline 29 **56,197.916**, midpoint 29 **45,260.411**, midpoint 33 **43,789.870**. Mean update seconds 0.528418 / 0.656114 / 0.771685. Matched midpoint is 24.17% slower per equal targets; midpoint 33 is 28.34% slower than baseline and has 3.25% lower targets/s than midpoint 29. These are fresh short probes, not historical full-run or equal-quality cost measurements. The earlier full maximum run remains unchanged; its faster end-to-end result cannot be attributed solely to physical batch size.
- Allocated peaks reproduce 10.030 / 8.736 / 9.891 GiB. Allocator history identifies four simultaneous vocabulary-sized buffers: FP16 logits and log-softmax result, FP32 NLL input cast and dense backward gradient. Total 8.339613 GiB at batch 29 and 9.489903 GiB at batch 33: 95.46%/95.94% of the midpoint allocated peaks (83.15% for baseline 29). Requested-byte replay is close but not identical to allocator rounded/reused block statistics; report uses measured stats as peak authority.
- Raw CUDA kernel events are counted once and joined to leaf CPU operators by External id, avoiding duplicate CPU/GPU annotation and operator/kernel rows. Head GEMMs and vocabulary loss/casts/gradient buffers consume 65.19% / 53.69% / 54.59% of kernel time. Midpoint 29 inverse reconstruction and local recomputation add 56.579 + 57.034 = 113.613 ms, roughly 89% of its 128.031 ms matched kernel-time gap. This is trace attribution, not a counterfactual speed guarantee.
- Actual trace refines the earlier generic AMP claim: the recorded build uses Half log-softmax then an FP32 autocast NLL cast. Preserve this numerical policy when claiming equivalence; a blanket float cast or FP32 fused loss is a separately declared numerical change. Large CPU `loss_scalar` time primarily waits on asynchronous GPU work, rather than scalar arithmetic.
- Recommended next experiment: bounded-memory output projection plus loss, shared by baseline/midpoint; avoid retaining all chunk graphs or rerunning core backward per output chunk. Validate tied-weight gradients, exact target sums/masks, FP16 scaling, clipping and updates, then repeat matched profiling and choose the fastest feasible batch. No such optimization was implemented or run during this review.
- Saved [diagnosis](docs/profiling-diagnosis-2026-10-10.md), machine-readable evidence in `docs/analysis/profiling-audit-2026-10-10.json`, and read-only reproduction script `docs/analysis/audit_profiling_results.py`. Original archive, extracted evidence, core training source and recorded full-run results remain untouched. No GPU is available on the local audit machine.

## Current bottleneck profiling preparation — 2026-10-10

- User requested profiling and selected preparation of a Colab notebook for them to run. Local CUDA is unavailable (PyTorch 2.9.1+cu128; Python 3.12.2), so no GPU timing/memory or bottleneck conclusion is claimed.
- Added standalone `scripts/profile_training.py`, `notebooks/profiling_colab.ipynb`, `tests/test_profiling.py` and [profiling handover](docs/profiling-handover.md). The notebook embeds the tested new script and clones the frozen implementation at `d071e1e2d8e4575d2fc9ca5ccb1ef4e3d5e185b7`, so it can run without pushing these additions to GitHub.
- Profiles baseline 29, midpoint 29 and midpoint 33 at context 512/accumulation 2. Each scenario uses separate fresh timing, trace and memory processes; 5 warmup updates, 20 timed updates and 1 traced/phase-memory update. Probe targets are separate from completed training. No checkpoint resume, full run, output/loss optimization or architecture extension is invoked.
- Timing pass is uninstrumented; trace labels cover forward, output/loss, inverse reconstruction, local recomputation/VJP, attention/MLP forward calls, optimizer and transfers. Synchronized memory diagnostics preserve nested phase peaks and capture allocator history. Inclusive timing/memory ranges overlap; forward annotations do not constitute full module-backward attribution. Missing CUDA events, overflow, data drift or existing output fail explicitly; independent successful passes are preserved.
- Notebook pins T4/Torch/CUDA/NumPy, records Python differences, checks all 25 frozen source hashes and baseline config checksum, stages hash-verified existing Drive data, and writes unique Drive receipts/logs/results plus a downloadable evidence ZIP. Original model/trainer/source-manifest files remain unchanged.
- Fixed relative CLI path resolution discovered during the first CPU suite attempt; failure logs were preserved in the preparation workspace. Added regression coverage and reran the complete suite successfully.
- Verification: `OMP_NUM_THREADS=1 python3 -m pytest -q`: **41 passed** in 8.92 seconds; one inherited Requests dependency warning. All nine fresh-process tiny CPU passes completed; first measured losses match across timing/trace/memory modes. CPU traces exported; CUDA memory remains explicitly unavailable. Notebook cells compile, embedded harness executes/checksums correctly, and delivered notebook bytes match the repository copy. All 25 frozen execution-source hashes still match recorded evidence.
- Next: user runs the T4 notebook and returns its ZIP. Review actual live output/loss buffers and operator self-device time against reconstruction/attention/MLP before selecting an optimization. B200 and monetary savings remain unmeasured.

## Authorized raw duplicate cleanup — 2026-10-07

- User explicitly selected removal of verified raw duplicates while preserving every unique file. The standalone README and 66 distinct lightweight submission records were committed/pushed as `f939848`; prior reports remain in Git history. Ignored raw binaries were never in Git history.
- Generated a frozen removal plan from the pre-cleanup inventory; preferred the final `artifacts-20261004T124446Z-1-001`/`002` paths and retained every unique SHA-256. Before deletion, verified all retained and duplicate files by size/hash and refused tracked, symlink, escaping or non-inventoried paths.
- Removed exactly 80 byte-identical redundant raw files, reclaiming 2,725,406,564 bytes (2.73 GB). All 96 unique raw contents survived post-cleanup hash verification. No unique result or Google Drive content was deleted; final split export stays complete. Older local exports can be reconstructed via [the lookup](results/raw-deduplication.json), copying retained paths back to removed paths.
- Cleanup command: `python3 docs/analysis/deduplicate_raw_results.py --inventory results/submission-manifest.json --plan results/raw-deduplication.json --apply`. The initial command without `--apply` generated the plan only.
- Temporary-fixture tests pass: dry-run preservation, changed-evidence abort before deletion, preference for final export, unique-file preservation, restoration mapping and rejection of repeated application. Submission inventory remains the pre-cleanup snapshot; lookup status is `completed`. All training source and measured results remain unchanged.
- Post-cleanup full audit regenerated identically to committed evidence: seven streams, ten reversible checkpoints, data assets, decision/capacity and 25 frozen source hashes pass. All 66 lightweight submission hashes remain unchanged; README links and no-assignment-reference checks pass. `python3 -m pytest -q`: 33 passed in 6.78 seconds, with the same unrelated Requests dependency warning. `git diff --check` passes.

## Submission preparation — 2026-10-06

- User parked optimization/B200 projections and requested a detailed standalone README, versioned evidence and duplicate cleanup. Root README now explains every required study question without referencing an assignment.
- Generated `results/recorded` and `results/local-correctness`: 66 distinct lightweight recorded files, each hash-verified against its original source; complete trajectories and original failures are retained. No checkpoint/dataset binary is added to Git. Original raw exports contain 56 duplicate-content groups (2,725,406,564 redundant bytes); originals have not been moved or deleted. Asked whether duplicate cleanup should include those ignored exports; previous commits do not contain them.
- User clarified not to delete results and to deduplicate before committing. External archival plan was withdrawn; original raw storage locations remain unchanged. No destructive cleanup has occurred. Previous committed reports/audits remain in Git history.
- Added an explicit Git-ignore exception for curated `results/**` so nested `runs/` and `data/` records are included; raw export directories remain ignored. Added reproducible submission export/hash-inventory utility outside the frozen execution-source manifest.
- `python3 -m pytest -q`: 33 passed in 7.66 seconds; one inherited Requests dependency warning. No GPU training was launched. Frozen training code/method/protocol files are unchanged.
- Submission verification passed: 66 original-byte-identical evidence files, no repeated content hashes within the curated export; root/result README links resolve and neither refers to an assignment. The plot reproduces from committed lightweight metrics alone. Exporter fixture tests pass for content deduplication, preservation of an existing manifest and rejection of conflicting split exports. All 25 frozen execution-source hashes remain unchanged.
- Imported `nvidia-smi` stdout contains two trailing-padding lines. Added a scoped `.gitattributes` whitespace exemption for `results/recorded/**` instead of altering original evidence bytes. All 176 original raw files remain at their original paths/sizes; only lightweight deduplicated submission evidence is staged.

## Confirmed decisions

- Session preference: **fully autonomous**; foundation creation took place in Default mode.
- Model target: approximately **20M trainable parameters**. The implemented baseline uses 7 layers, width 256, 8 heads, a 1024-unit SwiGLU MLP, tied GPT-2 embeddings, and a 512-token context. Its exact verified count is **20,340,736** trainable parameters.
- Every full run uses **exactly 50,000,000 FineWeb-Edu training targets**. The former 25M English, 12.5M code, and 12.5M math mixture is retired by user direction.
- Dataset preparation is pinned to `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, requested revision `v1.0.0`, and the GPT-2 tokenizer commit encoded in the preparation script. Resolved commits and binary hashes will be frozen in the generated manifest.
- Experiment order: baseline; Euler and midpoint at the baseline physical and effective batch sizes; selected reversible variant at its maximum feasible physical batch size under a fixed memory budget.
- Execution venue: Colab. The completed baseline ran on a Tesla T4 with FP16; its capacity, timing and memory measurements are recorded below. Reversible runs require the same recorded hardware/software controls.
- Primary reversible specification: *Reversing Large Language Models for Efficient Training and Fine-Tuning* (arXiv:2512.02056). Midpoint maps to Eq. 2.4 and “Euler/oiler” to the Hamiltonian staggered update resembling symplectic Euler in Eqs. 2.8–2.9. Both pass CPU and all 60 CUDA correctness cases; both matched 50M runs are complete and audited. Midpoint alone passes the loss cutoff.
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
| Reversible Euler implementation | Eqs. 2.8–2.9 inverse backward/reference; CPU/CUDA gates and matched training complete; final loss fails predeclared threshold |
| Midpoint implementation | Eqs. 2.4–2.5, h=0.5, inverse backward/reference; CPU/CUDA gates and matched training complete; quality-qualified for maximum stage |
| Colab smoke validation | Complete: 65,536 targets; finite validation and preserved checkpoint |
| Hardware and batch-size benchmarking | Complete on Tesla T4: batch 29 passes, batch 30 fails 10% headroom |
| Full baseline training | Complete and audited: exactly 50M targets, final validation/checkpoints preserved |
| Reversible comparisons | Both matched 50M runs and maximum midpoint complete and audited; planning recorded; core report submitted |

## Next steps

1. Preserve the imported baseline artifact bundle and its source snapshot; do not rerun or resume the completed baseline.
2. Preserve the audited export `artifacts-20261004T112021Z-1-001` and corresponding Drive study root. Read the [matched-run audit and planning findings](docs/matched-run-audit-2026-10-04.md); no matched retraining is needed.
3. Preserve both parts of the final export `artifacts-20261004T124446Z-1-001` and `artifacts-20261004T124446Z-1-002`; part 002 contains two required checkpoints. No retraining is needed.
4. Review the [completed core report](docs/core-report-2026-10-06.md). Admin acceptance/reprioritization is pending before any optimization/scaling extension; no utilization telemetry or B200/70B savings projection is established.

## Maximum-run audit and core report — 2026-10-06

- Resumed the preserved fully autonomous mode. No GPU experiment, frozen-source edit or optional extension was performed.
- Audited the two-part final export without merging/moving files: all seven full/smoke streams, ten reversible checkpoints, four recorded dataset/tokenizer assets, 25 frozen source hashes, planning evidence and capacity reports pass. Baseline recovery has no checkpoints; original baseline checkpoint evidence remains in the earlier audited export.
- Maximum midpoint completed exactly 50,000,000 targets, physical batch 33, accumulation 2, effective batch 66, 1,480 updates. Final/best validation 5.509121780029297; elapsed 1,254.684623 seconds; 39,850.651777 valid targets/s; peak allocated/reserved 9.891305/13.076172 GiB. One zero-cursor startup, no logged checkpoint resume/overflow. Effective batch changed from 58, an explicit optimization confound; maximum has three evaluations rather than four.
- Capacity: 33 passes 10% headroom and separate confirmation; 34/36 execute three updates but fail headroom; 40/48/64 OOM. Absolute OOM boundary and GPU utilization are not measured. User-reported Drive loss has no recoverable partial training log in these exports; missing local data preflight and successful restaging are documented separately.
- Completed [core report](docs/core-report-2026-10-06.md), [reproducible audit](docs/analysis/audit_core_results.py), [evidence JSON](docs/analysis/core-audit-2026-10-06.json) and four-run trajectory figure. Report includes an illustrative Google Cloud Iowa T4 accelerator-only USD 0.35/h estimate (price checked 2026-10-06), not measured Colab spend; costs for lost sessions, storage/transfer and compute units remain unknown.
- Result: matched allocated-memory saving 12.9%, verified capacity +13.8%, but maximum midpoint remains 12.1% slower than baseline. No speed/money win or scaling inflection was established. Full-vocabulary logits/loss-buffer pressure is a code-grounded hypothesis needing profiling, not measured causal attribution.
- Verification commands: `python3 docs/analysis/audit_core_results.py --parts artifacts-20261004T124446Z-1-001 artifacts-20261004T124446Z-1-002 --output docs/analysis/core-audit-2026-10-06.json`; `python3 docs/analysis/plot_matched_results.py --artifacts artifacts-20261004T124446Z-1-001/artifacts --include-maximum --output docs/figures/core-loss-2026-10-06.svg`. SVG trailing whitespace normalized mechanically. Admin acceptance of the submitted report remains pending.
- Final checks: 43 local report/index links resolve; both analysis scripts compile; generated SVG parses as XML and its PNG rendering was visually inspected; `git diff --check` passes. Plot generation emitted only a Matplotlib cache-location warning and used a temporary cache; no experiment measurement is affected.

For each future work entry, record the date, completed work, exact commands and configuration, artifacts, findings, blockers, and next steps. For experiments also include hardware, precision, seed, target counters, and whether the run is a probe, partial run, or completed full run.

## Planning checkpoint completed — 2026-10-04

- User confirmed less than one hour of Colab usage and ability to restart; exact remaining compute units/runtime and a new T4 allocation remain unknown. This is an availability statement, not a measured budget guarantee.
- Recorded midpoint selection, correctness/trajectory/loss/memory/time findings, failure history and unchanged protocol in `docs/review-decision-2026-10-04.json`.
- Added `docs/maximum-run-colab-cell.py` to verify the audited proposal/source/CUDA report, install the decision without replacing a differing one, and launch the existing maximum runner. Training source remains frozen and unchanged. Completed baseline/matched runs must not be rerun.
- Maximum capacity, maximum 50M training and the core report are still pending GPU execution. Logs/checkpoints remain Drive-backed; no optional extensions were started.
- Verification: decision policy matches the frozen selection policy; all 25 frozen execution-source hashes remain unchanged. Handoff compilation and temporary-study tests passed for decision installation, identical repeat, preservation of a conflicting decision and rejection of changed audit evidence. No GPU training was launched locally. `git diff --check` passed.

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

## Colab dataset-path recovery — 2026-10-04

- User hit `FileNotFoundError: Baseline dataset is required` before reversible orchestration. Both notebooks use the same canonical path; the exception establishes that its manifest is absent there, but does not establish whether the data moved or was deleted. The actual Drive layout is unavailable locally.
- Added `scripts/stage_baseline_data.py` and integrated it into the reversible notebook. It discovers nested baseline exports and moved datasets within `MyDrive/Reversability`, verifies the baseline config and exact manifest/binary/tokenizer hashes, stages a verified runtime copy, and preserves any replaced runtime folder. Missing/corrupt/different data produces an actionable error instead of regenerating comparison data. The resolved artifact root is used by subsequent phases.
- `OMP_NUM_THREADS=1 python3 -m pytest -q`: **22 passed in 5.49s**, including four recovery/rejection tests; existing Requests dependency warning only. Python/notebook compilation and `git diff --check` passed. The helper independently validated the actual imported 50M baseline dataset and manifest SHA-256 `71a12f5765cf37b7f5fb8753d51fa0828915a6db313ae20aef7fa9fb2e14df8a` without modifying that export.
- Next action: reopen the published reversible notebook and rerun with the baseline Drive artifacts intact. If the helper cannot find the exact data, restore the original exported dataset folder under the project Drive directory; no reversible training has been claimed from this fix.

## Deleted Drive artifacts — baseline recovery prepared, 2026-10-04

- User confirmed deleting the artifacts from Drive after the staging log found no baseline config/run-log roots. The original local exported artifacts are intact; rerunning baseline training is unnecessary.
- Created `artifacts/baseline-reference-recovery.zip`: **23 files, 528,997,010 bytes**, SHA-256 `36c3c448c8d1f3c67bff3dc760d94909abd31705bf18e0f7dbe25bdfcff5f7a6`. The archive includes the original baseline latest/best checkpoints, exact dataset/manifest/tokenizer, configs/logs/capacity evidence and source snapshot. Only separate smoke checkpoints are omitted from this recovery transfer; all original files remain preserved. The bundler independently read and verified each archived file against its input SHA-256.
- Added a reproducible archive builder and automatic recovery in the staging helper. Uploading the named ZIP to `MyDrive/Reversability` lets the notebook restore it into a new persistent folder, verify data/config, and continue using the resolved root. Later sessions reuse that folder and already valid runtime data. Notebook discovery also covers `MyDrive` rather than just the project folder for moved exports.
- Two additional tests cover archive restoration/idempotence and path-escape rejection. All **24 tests pass**; Python/notebook syntax and whitespace checks pass. No authenticated Google Drive connector or mounted local Drive uploader is available here, so the one-time ZIP upload remains the user's next action. CUDA correctness/training and the final assignment report are still pending.

## Smaller comparison-input recovery — 2026-10-04

- User questioned the baseline checkpoint requirement, then confirmed only the Git checkout exists in the runtime while staging still finds zero baseline records in Drive. Git intentionally excludes datasets/checkpoints; the runtime data directory is created by successful staging. The one-time restore upload is still required after Drive deletion.
- Removed the trained-baseline checkpoint dependency from reversible orchestration. Baseline acceptance now requires completed exact-budget summary, matching logged summary fields and final validation evidence; data/config hashes remain enforced. Baseline weights remain preserved locally and were previously audited. New smoke/matched/maximum runs still require their checkpoints and exact cursor checks.
- Built `artifacts/baseline-inputs-recovery.zip` with `--without-checkpoints`: **21 independently hash-verified files, 74,211,452 bytes**, SHA-256 `0e5698f03f087625648d84b32171496cc6a560cabc8b4321734e89f6ceabd84c`. All original manifests, token binaries, tokenizer assets and baseline result/control records are preserved. Existing full recovery ZIP remains available but is unnecessary for training comparisons.
- The staging helper accepts the small ZIP as the preferred recovery file and retains compatibility with the full ZIP. Added log/summary disagreement and checkpoint requirement tests; parametrized archive restoration over both filenames. `OMP_NUM_THREADS=1 python3 -m pytest -q`: **28 passed in 6.49s**, existing Requests warning only. Python/notebook syntax and whitespace checks passed; the actual imported baseline passes checkpoint-free reference validation.
- Integration-checked the actual 74 MB archive in an isolated temporary Drive/runtime layout: automatic extraction, exact manifest/binary/tokenizer validation, runtime copy and checkpoint-free reference acceptance all passed. This is a local recovery check, not a claim that the user's Drive upload or CUDA run has happened.
- Next action: upload **baseline-inputs-recovery.zip** to **MyDrive/Reversability**, keeping its filename, then reopen the published reversible notebook and run all cells. Do not create the runtime data folder manually or retrain the baseline. CUDA gates and reversible measurements are still pending.

## Colab CUDA-build mismatch — 2026-10-04

- Recovery succeeded in the user's Colab runtime: artifacts resolved to `MyDrive/Reversability/baseline-recovery-7bldlite/artifacts`. Matched orchestration then stopped before CUDA gates/training because installed PyTorch was `2.11.0+cu130` rather than baseline `2.11.0+cu128`. No reversible targets were committed by this attempt.
- Verified the official PyTorch previous-version instructions and CUDA 12.8 wheel index. The notebook now explicitly installs `torch==2.11.0+cu128` from that index and applies a dedicated constraint file (`numpy==2.1.3` as well). Hardware, Python, PyTorch, NumPy and runtime checks remain enforced; no comparison-control waiver is introduced.
- Moved GPU/software/data preflight checks before source/config freezing and report all environment mismatches together. Earlier aborted preflight source/config/gate records are preserved under `preflight-history` before refresh, only if no training metrics or checkpoint evidence exists. Source changes after training still fail. Source snapshots include the Colab constraint file.
- `OMP_NUM_THREADS=1 python3 -m pytest -q`: **33 passed in 5.85s**, existing Requests warning only. Tests cover complete mismatch reporting, preflight preservation, refusal to refresh training evidence, and no frozen study on environment failure. Python/notebook compilation and `git diff --check` passed. Actual CUDA 12.8 installation and CUDA gates remain to be verified on Colab, not this GPU-less host.
- Next action: reopen the published reversible notebook and run all cells. Existing recovered data is reused; no ZIP reupload or baseline rerun is needed. If Colab requests a kernel restart after installation, restart and rerun. Retain installation/preflight logs and investigate any remaining Python/GPU mismatch without silently changing the protocol.

## Matched results reported; planning review in progress — 2026-10-04

Source: user-pasted Colab `review` output dated 2026-10-04 11:01:49 UTC. Full reversible artifacts have not been imported or independently inspected locally. The review command successfully loaded completed run summaries/checkpoints and the CUDA gate's passed flag, then wrote `review-proposal.json`. User is unsure where files are saved; verify Drive presence before proceeding.

| Quantity | Baseline | Midpoint matched | Euler matched |
| --- | ---: | ---: | ---: |
| Committed targets | 50,000,000 | 50,000,000 | 50,000,000 |
| Optimizer updates | 1,684 | 1,684 | 1,684 |
| Final/best validation loss | 5.462294847167969 | 5.443028756835938 | 5.565652200683593 |
| Reported process elapsed seconds | 1,119.090000186 | 1,399.972977653 | 1,401.316090763 |
| Reported process valid targets/s | 44,679.158952086 | 35,714.975073178 | 35,680.743502186 |
| Peak allocated GiB | 10.030179 | 8.735979 | 8.735124 |
| Peak reserved GiB | 12.765625 | 11.517578 | 11.517578 |

- Both variants use the frozen physical batch 29, accumulation 2 and effective batch 58. Both share the 20,340,736-parameter configuration; final/best losses coincide in supplied summaries.
- Quality cutoff remains 5.562294847167969. Midpoint qualifies with loss delta -0.019266090332031; Euler fails with delta +0.103357353515625, exceeding the cutoff by 0.003357353515625. Do not relax it after seeing losses.
- Supplied midpoint process-summary memory is **12.9031% lower allocated** and **9.7766% lower reserved** than the baseline summary; reported process elapsed is **25.0992% higher** and throughput **20.0635% lower**. These are per-process comparisons, not established whole-run savings or timing: the user subsequently confirmed a resumed run. Aggregate process elapsed/committed-target increments and take maximum memory across segments before drawing full-run time/cost/memory conclusions. CUDA memory values are total process memory, not isolated activations.
- Proposal selects midpoint. `planning_session_recorded` is still false and remaining Colab budget is unspecified. CUDA error details, trajectories, partial/failure/overflow/resume history and source/config/hardware evidence remain to inspect. No maximum search/full run has been reported.
- Drive root: `/content/drive/MyDrive/Reversability/baseline-recovery-7bldlite/artifacts/reversible-v1`. Expected files include `review-proposal.json`, `correctness/cuda.json`, frozen configs/source snapshots, and each smoke/matched run's `metrics.jsonl`, console log, summary, latest/best checkpoints. The source code writes these directly to Drive. Keep the matched source unchanged until the maximum phase completes.
- User confirmed a resumed run and reported no retries, but has not identified the resumed variant or supplied full history yet. Requested each matched run's `console.log` and `metrics.jsonl` under that Drive root. Remaining Colab budget is still unspecified; do not finalize `review-decision.json` or maximum execution yet.

## Matched artifact audit complete — 2026-10-04

- Imported user-provided `artifacts-20261004T112021Z-1-001/artifacts` and audited both full runs, both reversible smokes, baseline reference records, CUDA correctness, source/config/data integrity and all eight reversible checkpoints. Detailed findings, exact commands, losses, measurement windows and checkpoint hashes: [matched-run audit](docs/matched-run-audit-2026-10-04.md). Added a reproducible standalone [loss plot](docs/figures/matched-loss-2026-10-04.svg); plotting code is outside the frozen training-source paths.
- **Resume clarification:** all six exported full/smoke streams have one startup from target zero; reversible logs contain no `--resume` command, overflow event or training failure. Orchestration records two failed CUDA-build preflights and a restarted workflow before training. Within this export, no checkpoint-resumed training segments need summation. User-reported resume is not substantiated as a training resume by this evidence; do not invent a missing segment.
- Therefore the full baseline/midpoint/Euler trainer times are **1,119.090 / 1,399.973 / 1,401.316 seconds**. Midpoint has **12.9031% lower peak allocated memory**, **9.7766% lower reserved memory**, **25.0992% longer trainer time** and **20.0635% lower end-to-end throughput** versus baseline. This is a measured matched-batch memory benefit and time penalty, not a time/cost win or scaling inflection.
- Both 50M reversible runs finish at 1,684 updates and the exact 21,632-target final partial window. Validation decreases monotonically at all four recorded evaluations; midpoint beats baseline at every recorded point. Midpoint loss 5.443028756835938 qualifies; Euler 5.565652200683593 exceeds the unchanged cutoff 5.562294847167969 by 0.003357353515625.
- CUDA v2 gate: **60/60 cases pass** across FP64/FP32/FP16 on T4. Maximum relative update-L2 errors **9.72278e-14 / 3.78595e-5 / 0.0173540** are within declared limits. All source gate hashes match the 25-file source snapshot, and current frozen source has no differences. All 18 architecture/optimizer/schedule/batch controls match baseline.
- Exact baseline manifest, both ordered data binaries and both tokenizer assets pass size/hash checks. All smoke/full config hashes match starts and checkpoints; summary files match final JSONL summaries. All eight latest/best checkpoints load, carry matching target/budget/step/config/manifest/validation metadata, include optimizer/scaler/RNG state, and have finite model/optimizer tensors and exactly 20,340,736 unique model parameters.
- No `review-decision.json`, selected maximum capacity report or maximum training exists in the export. Technical planning review recommends midpoint; remaining Colab budget/availability was requested again and is still pending. Keep maximum disabled until recorded. No measured price/compute-unit use is available.
- Added ignore coverage for downloaded `artifacts-*.zip` archives; original files are preserved, not committed. No training-source, notebook, tolerance, selection policy or run config was changed by this audit.

## Optimization notebooks publication — 2026-10-11

- Published the four October 10 profiling/optimization/capacity notebooks with their supporting scripts, optional kernels, tests and evidence audits. Notebook inventory distinguishes the failed historical v1 attempt, audited v2 capacity evidence, and the next v3 optimization review. Original 25 frozen source hashes remain unchanged.
- V2 uploaded evidence confirms physical baseline285 and midpoint814, accumulation1, at10% reserved-memory headroom; short probes do not establish full-budget loss or time savings. See the capacity-results review for controls and limitations.
- V3 compares cached head casts, force reuse, split attention/MLP VJPs,8192-token recomputation tiles, fused RMSNorm and allocator settings. Primary10% and explicitly exploratory5% headroom phases use independent gates and three confirmations. CPU checkpoint overhead is separate. No full50M run or automatic maximum search is enabled; optimization review comes first.
- Validation:83 tests passed/5 CUDA tests skipped,18 CPU smoke processes passed, full-architecture CPU gate and actual8192+17-token MLP boundary gate passed; four RMS kernels compiled offline for T4. Seven embedded v3 sources match repository bytes, isolated extraction/import succeeds, all four notebooks compile. GPU numerical/capacity/throughput gates and final loss remain pending Colab.
