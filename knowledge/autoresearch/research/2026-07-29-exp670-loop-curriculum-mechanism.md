---
type: research
title: EXP-670 clipped-Poisson loop curriculum mechanism map
description: Page-anchored ST-M018.a/M026.b mapping, exact schedule, SAIR adaptations, and exclusions.
status: completed-source-map
timestamp: "2026-07-29T05:00:00+05:30"
experiment: EXP-670
---

# EXP-670 clipped-Poisson loop curriculum mechanism map

## Authenticated source

The immutable local full PDF `research/papers/arxiv-2604.07822.pdf` has SHA-256
`f9d0469a7628396daa1c0a67425927b72733a1339e0c838a27594f571f0f237b`. Text extraction
and rendered inspection covered pp. 2 and 4–10; Appendix F p. 16 supplies the exact dynamic-depth
parameters. The checkpoint SHA-256 is
`cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`.

## Claim-to-code map

| Source claim | Page | EXP-670 implementation | Boundary |
|---|---:|---|---|
| A recurrent block is reused R times | 2, Fig. 1; 4, §4 | `SharedResidualOperator` is one shared module called for every refinement | Direct structural transfer |
| Output projections are zero initialized, making each block an identity initially | 4–5, §4 | `operator.output.weight` and bias are exactly zero; every count initially reproduces proposal bytes | Direct transfer |
| Dynamic recurrence samples once per training iteration from `clip(Poisson(lambda), Rmin, Rmax)` | 5, §4 | `torch.poisson` with one frozen `torch.Generator` schedule and one scalar count per batch | Direct schedule transfer |
| Max-R=8 uses lambda=4 and minimum R=2 | 16, Appendix F | mean 4, min 2, max 8; seed 67000825; 400 frozen draws | Exact source setting |
| More inference iterations can improve extrapolation but can also cause latent overthinking | 8–9, §6.3 | fixed target-free inference count 4 plus preregistered count 2/3/4/5/8 diagnostics | Bounded diagnostic transfer |
| Adaptive halting uses KL/entropy and the hop curriculum changes training data complexity | 7, §6.1; 10 | not implemented | ST-M018.b explicitly closed |

## Frozen schedule and matched work

The native PyTorch stream has histogram `{2:99, 3:86, 4:73, 5:63, 6:34, 7:27, 8:18}`,
total 1,600 and empirical mean exactly 4.0 over 400 updates. The fixed-mean arm uses 4 each
batch. The deterministic matched-histogram arm orders the same multiset; the shuffled/count-independent
arm uses a separately seeded permutation of that multiset. Thus all trained arms execute exactly 1,600
refinement calls per example over formal training. Count is chosen once per batch, never per example or
from an answer. Calibration uses the frozen first pair 5 and 3, so its representative mean is also 4.

## SAIR adaptations and exclusions

The paper uses a decoder-only Transformer with attention/FFN blocks, NoPE in depth experiments, and
does not inject the original input at each recurrence. EXP-670 instead uses a position-free Conv1d residual
operator around the frozen EXP609 hidden proposal and explicitly reinjects both that proposal and the six
legal input channels on every refinement. This input reinjection is a declared SAIR adaptation, not a paper
claim. The mandatory operand-bit scan remains exactly the inherited dilation schedule and is performed once;
loop count affects only post-proposal refinement.

No hop curriculum, KL/entropy halt, adaptive stopping, positional embedding, language-token attention/FFN,
answer-dependent count, validity filtering, repair, modular arithmetic routine, checked/private answer, or
submission path is present. This is bounded-faithful evidence for ST-M018.a and overlapping ST-M026.b, not a
full language-model reproduction.
