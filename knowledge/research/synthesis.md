---
type: Research Synthesis
title: Modular Arithmetic Research Synthesis
description: Evidence-backed synthesis of the supplied memo, official baselines, and implementation audit.
status: active
tags: [research, learned-algorithms, modular-arithmetic]
timestamp: "2026-07-11T23:15:00+05:30"
last_updated: "2026-07-21T00:00:00+05:30"
---

# Research synthesis

## Central conclusion

Direct sequence-to-sequence modular multiplication is a weak primary bet. The official digit Transformer and DLP-inspired models clear Tier 1 but collapse rapidly on unseen primes. The most promising direction is a **compliant learned algorithm decomposition**: supervise a local transition and compose it through an explicit recurrence.

For a binary scan of `a`, the relevant transition is:

```text
t[k+1] = (2 * t[k] + bit[k] * b) mod p
```

The scan order is fixed; the cell must learn carry propagation and conditional reduction. A hard binary bottleneck ensures autonomous rollout consumes the model's own predicted state.

## Strength of evidence

- **Observed locally**: official public data, source code, scorer behavior, example artifacts, and baseline descriptions at the pinned challenge commit.
- **Source-backed**: neural algorithmic reasoning literature favors intermediate-step supervision and structured transitions; representation, sampling, and losses strongly affect arithmetic generalization.
- **Unverified external claim**: a public challenge-tagged model card reports perfect public-benchmark accuracy from a carry-aware TCN Horner recurrence. It is not an independently confirmed official ranking and does not prove hidden-seed generalization.
- **Hypothesis**: a trajectory-trained local cell can generalize across unseen primes and widths while remaining fast enough on the CPU sandbox.

## Repeated lessons

1. Exact answer accuracy is qualitatively harder than low arithmetic distance.
2. Held-out-prime validation matters more than random example splits.
3. True recurrence trajectories are a better training distribution than arbitrary states if rollout is the target.
4. Prime-value and bit-width sampling must match the benchmark and cover reduction-boundary positions.
5. Tail errors dominate long rollouts; worst-bit or margin-aware losses are worth testing after ordinary BCE works.
6. Public-seed success must be separated from secret-style generated diagnostics.
7. Weight perturbation is both a scientific control and compliance evidence.

The first POC added a concrete eighth lesson: training on composite moduli while excluding held-out prime identities materially improved prime generalization. This supports learning a generic conditional-reduction rule, but it is a development-set intervention that needs confirmation on new splits.

## Learned execution refinement

Programmable and compiled Transformer work strengthens the case that a shared neural block can represent memory access, control flow, and long iterative execution. Percepta's technical article further proposes append-only execution traces and low-dimensional attention for efficient indexed reads. These are source claims about representational and systems feasibility, not reproduced SAIR results; see the [Playwright-verified applicability note](../autoresearch/research/2026-07-18-learned-neural-execution.md).

The compliant transfer is architectural rather than algorithmic. A SAIR challenger may train persistent registers, trace writes, read/write addressing, execution transitions, and residue heads from generated trajectories. It may not import a compiled interpreter, modular solver, by-construction memory-update program, or hand-assigned execution weights. This distinction is stronger than the randomized-weight check alone because provenance determines whether weights contain learned capability or encoded software.

The resulting neural-RAM hypothesis complements rather than replaces the learned Horner decomposition. Candidate 005 supplies a strong compact-state control; a trace executor is valuable only if addressable learned memory preserves information lost at hard residue boundaries, improves exact held-out width extrapolation, and remains CPU-feasible. EXP-654 tests that claim separately from a generic latent workspace.

## Full-PDF Super Transformers reconciliation

The [normative full-PDF ledger](../autoresearch/research/2026-07-21-super-transformers-full-pdf-audit.md) now separates 44 stable parent records into 81 atomic coverage components. It corrects several abstract-level conflations: symbol equivariance is not operand swapping; target-token DRM is not latent-boundary SPRM; EXP-654 is not a Universal-Transformer memory-token test; literal ERM depends on hand-constructed verifiers; the legal local call/return model is distinct from external SRLM; and Neural Computers demonstrates open-loop video rather than reusable neural RAM. No parent method or main Super Transformer method has faithful completed SAIR evidence; policy/source-only closures are reported separately.

[Loop the Loopies](../autoresearch/research/2026-07-21-loop-the-loopies.md) contributes a promising but untested ordering hypothesis: apply each stored block twice locally rather than repeating the whole stack, and reinvest measured activation/throughput savings into capacity. Planned EXP-655 isolates the dense schedule under identical stored parameters and executed calls before any MoE factorial. Its target-only post-training is a separate conditional analogue. The paper's source claims, PDF acquisition, and blocked plan do not change Candidate 005 or authorize evaluation.

The supplied memo is retained as an immutable source under [`sources/research/`](../../sources/research/).
