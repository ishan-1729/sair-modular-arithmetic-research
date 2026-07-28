---
type: Research Note
title: EXP675 RRT source and ConvGRU mechanism map
description: Authenticated visual mapping of Relaxed Recursive Transformer equations to physical recurrent visits.
status: active
tags: [autoresearch, rrt, svd, lora]
timestamp: "2026-07-29T00:00:00+05:30"
---

# EXP675 source-complete RRT mapping

The authenticated 49-page local PDF was visually audited on pages 3-5 and
10-11. Page 3 defines CYCLE sharing. Page 4 introduces distinct LoRA modules by
physical layer/visit in Eq.3. Page 5 states that all parameters, not only LoRA,
are trained; Eq.4 applies truncated SVD to the full-teacher minus tied-weight
residual; Eq.5 reconstructs `W approximately W' + BA`. Figures 6-7 on pages
10-11 show the paper's claimed advantage of SVD over zero initialization,
rank-dependent recovery, Average initialization, and extended uptraining.

For Candidate005 the semantic analogue is a raw gate/candidate Conv1d kernel
at each physical `(cycle,dilation)` visit. The retained shared kernel is the
continued teacher's visit centroid. Rank-8 factors are physical-visit-specific,
jointly trained with the shared base, and fixed at inference. This differs from
EXP617's one global delta and from EXP651's zero-initialized frozen-base analogue.

PEFT was absent. Native `torch.linalg.svd` is both maintained and exact for the
actual raw parameter geometry, avoids an unproved wrapper semantic translation,
and permits direct deterministic CPU factor/reconstruction tests.

The paper's Transformer-only attention/FFN placement, language-model token
corpus, KD, early exit, and continuous batching are excluded. No source claim
is treated as evidence that the mechanism generalizes to SAIR; only the frozen
root experiment can provide that evidence.
