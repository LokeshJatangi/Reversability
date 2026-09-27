# Research roadmap

These are proposed follow-up studies, not implemented features or measured findings. Start with the controlled approximately 20M-parameter, exactly 50M-target runs in the [experiment plan](experiment-plan.md). Keep their manifests and results separate from later extensions.

## Activation checkpointing comparison

Add ordinary autograd with activation checkpointing as a practical reference after establishing the baseline. Compare stored activations, checkpointed blocks at several checkpoint intervals, and accepted reversible variants on the same hardware, precision, sequence length, data, and effective batch size. Measure loss, throughput, and peak memory, including recomputation and any retained boundary states. Apply the same exact target budget to every full training run.

## Numerical stability

Study reconstruction and gradient error as depth, integrator step size, precision, and sequence length vary. Test higher-precision reconstruction or periodic stored states as explicit variants if needed. Track non-finite values, error accumulation, gradient norms, and training-loss behavior. Require predefined error tolerances and identify where reversibility ceases to be reliable; successful algebraic inversion alone is insufficient acceptance evidence.

## Profiling before kernel work

Profile representative full optimizer updates only after correctness is established. Attribute time and memory to attention, feed-forward layers, loss, optimizer state, reconstruction, data delivery, and device transfers. Investigate whether chunked cross-entropy reduces peak memory and what throughput it costs. Pursue fused or custom kernels only for demonstrated bottlenecks, and repeat numerical equivalence checks after each optimization.

## Alternative integrators

After reading the supplied paper and validating Euler and midpoint, identify other integrators and coupling schemes worth testing. Record equations, required retained state, function evaluations, backward method, and stability assumptions before implementation. Compare quality and cost with controlled parameter counts and training targets; do not assume a higher-order method is faster or more memory efficient.

## Moonwalk inverse-forward differentiation

Evaluate [Moonwalk](https://arxiv.org/pdf/2402.14212) only after the required midpoint/Euler study. Begin with a feasibility analysis for Transformer attention, normalization, embedding, and SwiGLU layers: identify whether efficient right-inverse Jacobian products exist, which cotangents must be checkpointed, and whether architectural constraints would invalidate a controlled comparison. Validate every custom operator against ordinary autograd. The paper's current empirical evidence is based on convolutional networks, so do not transfer its memory or runtime claims to language models without measurement.

## MoE and RevFFN

Keep the core approximately 20M experiment dense so it isolates reversible dynamics without routing and expert-utilization confounds. After the core study, evaluate [RevFFN](https://arxiv.org/pdf/2512.20920) as a separate MoE track. First establish a matched ordinary-autograd MoE baseline, then compare RevFFN under the same target stream and hardware. Report total versus active parameters, expert/router configuration, load balance, tokens per expert, routing stability, communication and sparse-kernel overhead, peak memory, throughput, and quality. The source paper studies full fine-tuning of a pretrained Qwen MoE with 2.7B activated parameters on an H800, so its results are hypotheses—not measurements—for a 20M from-scratch model.

## DeepSeek research

Review relevant primary DeepSeek papers and released implementations in a separate research pass. Identify specific techniques that may affect attention memory, optimizer/precision behavior, or architecture, and record sources and applicability assumptions. No DeepSeek technique or performance claim has been verified in this foundation. Introduce one technique at a time after the initial comparisons so architecture changes do not obscure the integrator comparison.

## Conditions for extrapolating to 40B parameters

A 20M-parameter study cannot by itself establish feasibility, quality, or speed at 40B parameters. Any projection must identify and test these assumptions:

- Architecture scaling, including width, depth, vocabulary, sequence length, weight tying, and dense versus mixture-of-experts total and active parameter counts.
- Memory decomposition into parameters, gradients, optimizer state, activations, reconstruction state, temporary buffers, and allocator/workspace overhead. Specify precision, sharding, offload, and replication for each component.
- Distributed execution strategy, device memory and count, interconnect bandwidth, collective communication, and pipeline/data/tensor parallelism overhead.
- Reconstruction stability and gradient agreement at substantially greater depth and under the proposed distributed precision policy.
- Compute per target, hardware utilization, recomputation, communication overlap, and kernel behavior measured at intermediate scales rather than assumed to scale linearly.
- Training-budget and quality assumptions: the fixed 50M-target experiment is a controlled comparison, not evidence that 50M targets suffice to train a 40B model well.
- Cost and runtime estimates with explicit uncertainty, hardware assumptions, and intermediate-scale validation points.

Label every extrapolated number as a projection. Establish intermediate-scale measurements and a distributed memory/throughput model before presenting a 40B feasibility claim. Log evidence and unresolved assumptions in [Progress](../Progress.md).
