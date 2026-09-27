# Goal — reversible-training efficiency study

The Admin assignment, normalized into an executable research goal:

Determine whether reversible training provides a measured advantage in training time, activation memory, feasible physical batch size, and estimated monetary cost while preserving language-model quality. Establish where the advantage begins or disappears (the **point of inflection**) as model scale and context length increase, and use the measured small-scale evidence to build explicitly qualified projections for 70B-and-larger models. Do not assume reversibility wins; the experiment must be capable of showing a loss, tie, or win.

## Required experiment stages

**Scope lock:** Stages 1–6 below are the complete active assignment. Finish them and deliver the core report before implementing, benchmarking, or planning any optional method or scaling extension. Until the Admin accepts that report or explicitly reprioritizes the work, do not spend experiment budget or implementation time on MoE/RevFFN, Moonwalk, leapfrog, broader model/context grids, or 70B engineering.

1. **Baseline capacity and run.** On the selected Colab or local GPU, run a separately logged batch-capacity search for the approximately 20M-parameter ordinary-autograd baseline. Freeze the largest repeatedly successful physical batch under a stated memory-headroom rule, choose and record gradient accumulation/effective batch size, then train the baseline on exactly 50,000,000 FineWeb-Edu targets.
2. **Matched reversible comparison.** Implement and correctness-test at least two paper-grounded reversible variants:
   - **Midpoint:** the explicit midpoint architecture in Eq. 2.4 of *Reversing Large Language Models for Efficient Training and Fine-Tuning*.
   - **Euler:** interpret the assignment's “Euler/oiler” variant as the paper's staggered Hamiltonian update resembling symplectic Euler in Eqs. 2.8–2.9. Record this terminology mapping in results. Do not substitute forward Euler or another recurrence silently.
   - Leapfrog (Eq. 2.6) is deferred and must not be implemented during the active assignment unless the Admin explicitly reprioritizes it.
3. Train both required reversible variants on exactly 50,000,000 targets at the baseline's frozen physical and effective batch sizes. Use the same data order, tokenizer, evaluation targets, initialization mapping where possible, precision, optimizer, target-indexed schedule, and hardware. Report exact parameter-count differences before comparing outcomes.
4. Select the reversible variant only after it passes reconstruction/gradient gates. Apply a predeclared validation-loss acceptance threshold first; within acceptable variants compare final validation loss, loss trajectory, throughput, peak memory, and stability. Do not select from a visually preferred trajectory alone.
5. **Maximum-batch reversible run.** Run a separate capacity search for the selected reversible variant, then train it again for exactly 50,000,000 targets at its largest repeatedly successful physical batch. Preserve the baseline effective batch size through accumulation when feasible; otherwise disclose the changed effective batch, update count, and optimization confound.
6. Submit a report containing every full and partial run, exact configuration and commands, final and best validation losses, training-loss trajectories, valid targets/second, end-to-end time, peak allocated and reserved GPU memory, maximum verified batch sizes, checkpoint/resume history, failures, estimated compute cost, and other findings. Separate measurements from hypotheses and projections.

## Deferred scaling and decision analysis

The items in this section are not active work. Revisit them only after stages 1–6 are complete and the Admin has accepted the core report or explicitly authorized the next phase.

- Identify the point of inflection with a controlled grid over model depth/size and context length after the core 20M experiment. Include ordinary autograd and activation checkpointing as practical references. A reversible method saves memory by reconstruction but adds computation; report the region where larger batches recover that overhead and where they do not.
- Build 70B+ projections from an explicit memory decomposition (parameters, gradients, optimizer states, activations, temporary buffers, sharding/offload, and communication) and measured intermediate-scale points. The 20M/50M experiment alone is not evidence that a 70B run is faster, cheaper, or sufficiently trained.
- Treat “saving money” as a derived result: measured accelerator time multiplied by a cited price for a named service/date, plus any storage or transfer costs. Colab compute-unit use is reported separately when an exact currency mapping is unavailable.

## Sources and deferred-extension backlog

- Primary reversible-LLM paper: [Reversing Large Language Models for Efficient Training and Fine-Tuning](https://arxiv.org/pdf/2512.02056), arXiv:2512.02056. Read and record the exact equations, initialization, step size, reconstruction algorithm, and experimental controls before implementation. The brief did not include a code URL; do not claim code-level reproduction until one is supplied or located and pinned.
- **Deferred—do not start:** [Moonwalk: Inverse-Forward Differentiation](https://arxiv.org/pdf/2402.14212), arXiv:2402.14212. If authorized after the core report, gate it behind a feasibility study for Transformer attention/MLP layers because it requires efficient layer-specific right-inverse Jacobian products and the paper's reported experiments are convolutional.
- **Deferred—do not start:** [RevFFN: Memory-Efficient Full-Parameter Fine-Tuning of Mixture-of-Experts LLMs with Reversible Blocks](https://arxiv.org/pdf/2512.20920), arXiv:2512.20920. If authorized after the core report, establish a matched ordinary-autograd MoE reference and report total/active parameters, routing, load balance, and sparse-kernel overhead. RevFFN does not replace midpoint or Euler.

## Planning checkpoint

Hold one explicit planning session after the baseline and matched-batch midpoint/Euler runs are complete, and before the selected reversible maximum-batch run. Review correctness evidence, losses, memory, throughput, failures, remaining Colab budget, and the variant-selection decision. Keep the session focused on finishing stages 5–6; do not expand it into Moonwalk, leapfrog, MoE, or scaling work. Record the decision and any protocol amendment before continuing.

# Working rules

- Preserve the selected **fully autonomous** session preference. Proceed with authorized implementation and verification without asking the user to select a mode again.
- Target approximately **20M trainable parameters**. Report the exact count, architecture, and any count differences between variants before comparing results.
- Require **exactly 50,000,000 FineWeb-Edu training targets per full run**. The former English/code/math mixture is retired. Validation, correctness probes, warmup benchmarks, and batch-size searches are excluded from that budget and must be logged separately.
- Freeze dataset revisions, preprocessing, splits, ordered training targets, tokenizer assets, and evaluation protocol across comparisons. Record hashes and seeds. Do not silently substitute data or tokenizer versions.
- Distinguish physical batch size (sequences per device per microstep) from effective batch size (sequences per optimizer update across accumulation and devices). Also report valid target counts; padding does not count.
- Report measured results only with hardware, software versions, precision, configuration, exact commands, and artifact locations. Separate hypotheses, inherited findings, pending work, and measurements.
- Validate reconstructed states and gradients against an ordinary autograd reference before reversible training. Fix and document tolerances for each precision before assessing acceptance.
- Use arXiv:2512.02056 as the supplied reversible-LLM paper. Map “Euler” to its symplectic-Euler-like Hamiltonian formulation only after documenting the exact equations and backward reconstruction; do not infer an algorithm from the name alone.
- Follow the [experiment protocol](docs/experiment-plan.md), maintain [Progress.md](Progress.md), and keep speculative extensions in the [research roadmap](docs/research-roadmap.md).
- Save enough checkpoint state to resume the same target stream, optimizer schedule, and random state without counting targets twice. An interrupted partial run is not a completed full run.
