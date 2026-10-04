# Baseline artifact audit — 2026-10-01

The baseline finished successfully. This audit inspects imported evidence; it does not rerun training or independently reevaluate checkpoint loss on a GPU.

## Evidence and outcome

Artifact root: `artifacts-20260930T183134Z-1-001/artifacts`. Paths below are relative to that root. Preserve this ignored directory outside Git.

`runs/baseline_20m_fineweb_edu_50m_v1/metrics.jsonl`, `console.log`, `run_summary.json`, `latest.pt`, and `best.pt` agree: 50,000,000 committed targets, target budget 50,000,000, step 1,684. Final validation, latest checkpoint, best checkpoint, and summary are all logged. There is one startup from target zero, 169 progress events, 17 checkpoint events, four validation and four best-checkpoint events, and no overflow event. No interrupted partial baseline or resumed segment is present in this bundle. The final update contains 21,632 valid targets; preceding full updates contain 29,696 each.

Training timestamps: 2026-09-27 18:58:53.224939–19:17:32.646260 UTC. The date in the downloaded folder name is not the run date. No evidence identifies a notebook UI stall; the training process itself completed.

## Frozen setup

- Config: `configs/baseline_colab.json`; seed 1337; FP16; no compilation; physical batch 29; accumulation 2; effective batch 58 sequences; sequence length 512.
- Ordinary autograd: 7 layers, width 256, 8 heads, SwiGLU hidden width 1024, vocabulary 50257, tied embeddings, dropout zero, no biases; 20,340,736 unique parameters.
- AdamW: betas (0.9, 0.95), weight decay 0.1, gradient clipping 1.0; maximum/minimum learning rate 0.0006/0.00006; warmup 1M targets; target-indexed cosine schedule. Evaluation every 500 updates, checkpoints every 100 and at completion.
- Tesla T4: 15,637,086,208 bytes VRAM; one device; compute capability 7.5. Python 3.13.15, PyTorch 2.11.0+cu128, CUDA runtime 12.8, NumPy 2.1.3. Full environment evidence: `colab-session.log`, `pip-freeze.log`, startup event.
- FineWeb-Edu sample-10BT commit `fc9850dff5e2d0f8f776efe41b24a1c49556cfc5`; GPT-2 tokenizer commit `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Training SHA-256 `8d71872e4d7024801e459a6c44942907a89ae946105eb0905968746a0652c738`; validation SHA-256 `d5058ea280216a4504ac6c899028adfd1c5b236c3af9de5ff813dfc3e5ed61b1`. Both equal the previously prepared local streams. The imported manifest records two tokenizer assets; both pass their own size/hash checks. Preserve this actual manifest rather than replacing it with the local five-asset manifest.
- Config SHA-256 `e03e341c413de59808628095700cbdfc4a3aa5e520dc109f2082c0dcdcba8b4b`; dataset manifest SHA-256 `71a12f5765cf37b7f5fb8753d51fa0828915a6db313ae20aef7fa9fb2e14df8a`; both match startup and final checkpoints.
- Source manifest SHA-256 `21258f051074dddaaec074f6dd31bd0a1ab7a96e95b6703fa08db42337501193`. All 18 files in its ZIP snapshot match listed hashes. The local trainer also matches the recorded trainer hash.

## Separate capacity and smoke results

Capacity protocol: fresh process per candidate, optimizer initialization update, three repeated complete updates, reserved-memory ceiling 90% of VRAM. Candidate batches 1, 2, 4, 8, 16, 24, 28, 29 pass; 30 and 32 fail headroom without OOM. Batch 29 is the largest verified batch under this protocol, not an unrestricted hardware maximum. Search details and per-candidate commands are in `benchmarks/baseline_batch_capacity.json` and `.log`.

Smoke: 65,536 targets, three updates, final/best validation loss 10.6698841640625, elapsed 21.533655 seconds. Its summary's completed-budget flag refers to the smoke budget only. Smoke and capacity targets are excluded from 50M full-run accounting. Colab correctness log records six tests passed in 21.98 seconds.

## Full-run measurements

| Quantity | Result |
| --- | ---: |
| Committed training targets | 50,000,000 |
| Optimizer updates | 1,684 |
| Held-out validation targets per evaluation | 1,000,000 |
| Final and best validation loss, natural-log units | 5.462294847167969 |
| Final training-window loss, 21,632 targets | 5.472444974459135 |
| Elapsed training process including validation/checkpoint overhead | 1,119.090000186 seconds |
| End-to-end committed targets/second | 44,679.158952086 |
| Peak allocated CUDA memory | 10,769,822,208 bytes (10.030179 GiB) |
| Peak reserved CUDA memory | 13,706,985,472 bytes (12.765625 GiB) |

Validation trajectory: step 500 / 14,848,000 targets → 6.164262945800782; step 1000 / 29,696,000 → 5.677928981445312; step 1500 / 44,544,000 → 5.493801813232422; step 1684 / 50,000,000 → 5.462294847167969. Sampled training-window trajectory is preserved in all 169 progress events; it is not an every-update loss record.

Both final checkpoints load on CPU, have finite model tensors, and contain model, optimizer, scaler and Python/NumPy/CPU/CUDA RNG state. Latest checkpoint SHA-256: `980a25b30b13d3cd0dae11fac097e6c936172e3489c6496cfd3690bbab2550f2`; best: `ceef4f8ac101e11d823ab8a8d96f858bbca277c2b96bdf7b5fa3be0936a30697`.

## Recorded commands

Full baseline (console adds `/usr/bin/python3 -u`):

```bash
/usr/bin/python3 -u /content/Reversability/scripts/train_baseline.py --config /content/drive/MyDrive/Reversability/artifacts/configs/baseline_colab.json --device cuda
```

Capacity search parent and candidate commands are preserved verbatim in its log/report. Smoke invocation and preparation command are preserved in their console logs/manifest. Paths are original Colab paths; the imported files live under the artifact root above.

## Verification and limits

Audit commands used Python `json.loads` over metrics, `collections.Counter` over events, `hashlib.sha256` over config/data/tokenizer/source/checkpoint files, `zipfile.ZipFile` to verify the 18 source entries, and `torch.load(..., map_location="cpu", weights_only=False)` to inspect both final checkpoints. Model finiteness was checked with `torch.isfinite` and parameter count against the saved weights. All checks passed. No GPU training or GPU evaluation was launched locally.

Throughput includes evaluation and checkpoint overhead and has no separately synchronized steady-state measurement window. Process time excludes environment setup, data preparation, capacity search, smoke and any notebook/Drive-export downtime. Memory is total process CUDA allocation/reservation, not isolated activation memory. No measured monetary cost or Colab compute-unit use is available. No reversible advantage can be inferred from this baseline alone.

Stage 1 is accepted on this evidence. Next work is paper-grounded midpoint/Euler implementation and correctness gates followed by matched runs. Preserve the baseline and freeze the reversible validation-loss threshold before those runs. The required planning checkpoint remains after the matched reversible runs and before the selected maximum-batch run.
