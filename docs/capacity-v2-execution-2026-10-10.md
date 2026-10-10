# Capacity v2 execution record — 2026-10-10

Prepared and executed locally; **GPU execution remains pending** because this machine has no CUDA device. Delivered `capacity_colab_v2.ipynb` is self-contained and preserves the frozen study and previous notebook/results.

- **64 tests passed**,4 CUDA runtime tests skipped,10.56seconds; one inherited Requests dependency warning. Checks cover original suite, midpoint state/VJPs, tied-weight masked gradients, CPU checkpoint independence/exact continuation, physical50 body shape with accumulation1, rejection of accumulated/mislabeled batches, simulated GPU capacity refinement/three confirmations and runtime-error classification. Simulated capacity tests are orchestration evidence, not measured GPU results.
- **19 fresh-process tiny CPU probes passed** with identical current harness source hash. Every reference/candidate uses accumulation1. CPU checkpoint copy/write records are separate; roundtrip checks pass. CPU capacity remains explicitly unavailable.
- Full20,340,736-parameter **FP32 B2/context64** numerical gates passed for shared chunking and force reuse, two single-batch optimizer updates per method. These are bounded-context CPU checks, not full-context FP16 evidence. GPU gates use B8/context512 and the complete50257 vocabulary.
- All **five Triton kernels compiled offline for T4 sm75** with Triton3.5.1. This proves compilation only; CUDA numerical accuracy, peak memory and speed still require the notebook's runtime gates.
- Notebook's seven code cells compile; four embedded source hashes match current files; embedded installation and CLI import succeed in an isolated copy of the frozen25-source manifest. Delivered/repository notebook bytes match. All25 frozen hashes and original profiling harness are unchanged.
- Uploaded v1 ZIP CRC and all five extracted members verified. Numerical-gate failure preserved; no capacity or throughput rows were produced. Last recorded maximum remains physical33 under the earlier protocol.

V2 uses physical==effective batch, accumulation1, a fixed10% reserved margin, memory-first selection, progressive integer capacity refinement and three full-duration confirmations with CPU checkpoint staging. Only independently gated optimization combinations run. It records batch50 as a milestone and optional speed diagnostics; full50M quality remains a later phase with validation ceiling5.56229485.

The full-model CPU gate receipts were generated before the final GPU-only search/error-reporting amendment; optimized model code, numerical policy and CPU execution path are unchanged. The final19-process suite and notebook receipt use the final harness hash.

Run the notebook on Colab T4 and return its exported `*-capacity-v2.zip`. New unique Drive output folder: `Reversability/capacity-v2`. CPU probe snapshots stay separately in Drive and are not production resumes. No new GPU maximum, equal-quality win, speed improvement or savings is claimed.
