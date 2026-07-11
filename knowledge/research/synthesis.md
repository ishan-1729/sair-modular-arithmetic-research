---
type: Research Synthesis
title: Modular Arithmetic Research Synthesis
description: Evidence-backed synthesis of the supplied memo, official baselines, and implementation audit.
status: active
tags: [research, learned-algorithms, modular-arithmetic]
timestamp: "2026-07-11T23:15:00+05:30"
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

The supplied memo is retained as an immutable source under [`sources/research/`](../../sources/research/).
