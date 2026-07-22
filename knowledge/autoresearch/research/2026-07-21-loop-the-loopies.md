---
type: research-note
title: Loop the Loopies full-PDF review
description: Full-PDF architecture, compute, training, ablation, and SAIR-transfer audit for arXiv:2607.16051v2.
status: reviewed-exp655-authorized-no-result
timestamp: 2026-07-21
---

# Loop the Loopies

## Frozen source and provenance

- Primary record: [arXiv:2607.16051v2](https://arxiv.org/abs/2607.16051), revised 2026-07-20.
- Local immutable PDF: [`research/papers/arxiv-2607.16051.pdf`](../../../research/papers/arxiv-2607.16051.pdf), 38 pages, 891,751 bytes, SHA-256 `1abfc7644b0253119b53b86327011471a0d017c2904d648461fc9d6a10b074ce`.
- Acquisition: the primary landing page and latest revision were visibly checked with repository-local Playwright managed Chromium. The PDF was parsed and all 38 pages were rendered; representative architecture, ablation, appendix, and reference pages were visually inspected.
- Limitation: the paper's `IQuestLab/loopie`, `IQuestLab/megatron`, and `IQuestLab/vllm` GitHub links returned 404, and the linked Hugging Face previews were inaccessible (401/404) on 2026-07-21. This review therefore establishes PDF-level, not code-level, fidelity.

Everything below labelled a paper result is a **source claim**, not reproduced SAIR evidence. No model was trained or evaluated during this intake.

## Causal mechanism

Section 2.1 and Figure 1 (pp. 4–5) distinguish two schedules over the same stored blocks:

- model-loop: `B1, ..., BL, B1, ..., BL`;
- layer-loop: `B1, B1, B2, B2, ..., BL, BL`.

Loopie fixes recurrence at `R=2`; the complete Attention/MoE block is shared across its two local visits. Its layer-loop order is the same layout that Mixture-of-Recursions calls **Sequence sharing**, so schedule evidence is cross-registered at `ST-M009.a` and `ST-M043.a`. Loopie is not the full Mixture-of-Recursions system: it has no learned recursion-depth router or associated adaptive-depth/KV policy. It also does not introduce a loop/visit embedding, visit adapter, learned or adaptive exit, entropy depth allocator, or input reinjection. Its MoE is ordinary token routing inside each block; a token may route differently after its hidden state changes, but the paper does not provide a causal routing ablation. All stored experts count toward an artifact limit.

Appendix A, Table 4 (p. 35) reports:

| Model | Stored layers | Effective applications | Width | GQA heads/groups | MoE hidden | Experts/top-k |
|---|---:|---:|---:|---:|---:|---:|
| Loopie20B-A2B | 27 | 54 | 2,304 | 72/36 | 832 | 128/8 |
| Loopie6B-A0.6B | 18 | 36 | 1,536 | 48/24 | 576 | 128/8 |

Both use RoPE, RMSNorm, SwiGLU, and a Qwen3 vocabulary. Embeddings and the output head sit outside the recurrent stack. The PDF does not fully specify weight tying, dropout, router capacity, visit-specific KV semantics, or all normalization details, so these must be disclosed choices in any implementation.

## Compute and capacity recycling

Section 2.4 (pp. 6–10) starts from a `D=2048, L=48, R=1` reference and a `D=2048, L=24, R=2` seed. The selected `D=2304, L=27, R=2` model has a paper work proxy of about `1.424x` and activation proxy `0.63x`, not equal FLOPs. Checkpointing both visits as one recurrent unit makes the activation proxy scale with stored depth rather than `R`; the authors double microbatch from 1 to 2 and halve gradient accumulation while keeping tokens per optimizer update fixed. They report 261.53 versus 189.65 nominal TFLOPS on their configuration. Hardware, token-throughput, wall-time dispersion, memory measurements, and parallelism details are insufficient for local numerical reuse.

The transferable lesson is operational: measure representative optimizer-step wall time and aggregate process CPU, then reinvest demonstrated memory/throughput savings. It is not permission to declare parameter-, FLOP-, or wall-matched arms equivalent without measurement.

## Source results and limitations

- Figure 2 (p. 5) reports that layer-loop trails model-loop early and crosses it at roughly 1.2T tokens. Exact values, seeds, and error bars are absent.
- Figure 3 and Table 1 (pp. 8–10) compare capacity-reinvested models under a stated wall budget; the only exact throughput numbers are nominal TFLOPS.
- Figure 4 (p. 10) reports downstream-mean gains of `+1.1`, `+0.6`, `+1.7`, and `+2.2` across a scaling ladder. These are source-reported bundled results, not isolated schedule effects.
- Figure 5 (p. 11) is an imperfect recurrence ablation: the replacement is ambiguous and the 5B/6B naming is inconsistent. The stated `2.14x` is best treated as sample/token efficiency, not independently measured wall speed.
- Figure 6 (pp. 12–13) compares `R=2/3/4` with ordinary depth. Ordinary depth scores higher with substantially more work; `R=2` is selected for marginal tradeoff, not universal superiority.
- There is no dense-versus-MoE, router, initialization, norm, loop-index, adaptive-depth, inference-depth, or depth-extrapolation ablation. Section 6 (p. 25) explicitly leaves inference compute for future work.

Training details (pp. 13–15; Tables 5–6, pp. 36–37) include AdamW, stable no-decay learning rates, sequence length 8,192, and MoE auxiliary loss. The narrative's `3T` initial-stage wording does not reconcile exactly with `4 x 570B + 1.263T` and the table's approximately `3.5T`; downstream SPT and RL are additional.

## Post-training is a separate mechanism

Section 4.1 (pp. 16–18; Table 8, p. 38) describes supervised post-training with target-only cross-entropy masking, global batch 1,024, sequence length 131,072, about 2T token positions/ten epochs, and a stable `1e-5` learning rate. The authors acknowledge that comprehensive ablations are missing (pp. 24–25). This becomes `ST-M044`, a conditional **analogue**: target-only continuation versus matched mixed/trajectory supervision on a compatible SAIR recurrent backbone. Literal 2T-token scale is neither required nor justified.

The later GSPO/DAPO reinforcement stages (pp. 18–19) depend on autoregressive stochastic generation, group verifiers, and prompt filtering. They are non-applicable unless a future compatible model and separate preregistration justify them; they are not bundled with `ST-M043` or `ST-M044`.

## SAIR experiment mapping

`ST-M043` has independent atomic components:

1. layer-local versus whole-stack recurrence ordering, jointly covering the MoR Sequence-sharing slot `M009.a` but not its routing slot `M009.b`;
2. operational capacity recycling after measured memory/throughput gains;
3. optional ordinary MoE route evolution, only after a dense schedule verdict.

T048 [EXP-655](../ablations/exp-655-layer-loop-compute-matched-recurrence.md) implemented the schedule isolation first. T049 then reconstructed only checksum-authenticated historical test evidence, proved a green 391/391 unchanged-source oracle, dynamically admitted the full route from measured throughput, and completed one three-root formal comparison. Its primary dense layer-loop and model-loop arms retained identical stored parameters, executed block calls, data, seeds, updates, evaluation fidelity, and byte-identical initial states. `R=1` and untied equal-compute controls isolated recurrence and extra work; capacity reinvestment remained outside the package. The 300-second limit still applies only to final submission inference qualification.

EXP-655 is now `completed-rejected-dense-layer-loop-schedule-formulation`. Both primary arms scored zero at W32 on all three 300-case roots; their W16 scores were at most 1%, the layer-loop W32 mean delta and bootstrap lower bound were both zero, maximum invalid output was 88.33%, and the minimum trained-minus-randomized drop was zero. Integrity, matched-denominator, determinism, noncommutation, and resource gates passed, so this is a decision-producing rejection rather than a broken run. It updates `ST-M043.a` and the schedule-only part of `ST-M009.a` with negative evidence for this exact dense two-block fixed-R2 formulation. It does not test `ST-M009.b`, operational capacity recycling, ordinary MoE/adaptive routing, or `ST-M044` post-training.
