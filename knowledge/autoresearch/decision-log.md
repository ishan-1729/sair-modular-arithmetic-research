---
type: Decision Log
title: Autoresearch Decision Log
description: Dated promote, reject, and continue decisions tied to canonical experiment evidence.
status: active
tags: [autoresearch, decisions, evidence]
timestamp: "2026-07-12T02:20:00+05:30"
---

# Autoresearch decision log

## 2026-07-12

- **Continue incumbent** `candidate-001-horner-tcn9`: reproduced repository evidence establishes `highest_tier_above_90 = 2`, strong Tier-2 generalization, deterministic behavior, weight dependence, and sandbox headroom. It remains the release artifact until a challenger wins independently.
- **Start EXP-100**: Tier 3 is the first unsolved ranking boundary. Pre-register a matched shared-context width-9/width-16 smoke test before any full run. Do not consult checked Tier-3 answers for tuning.
- **Preserve negative DLP evidence**: the prior phase/DLP probe remained at chance for `p=251`; do not reopen it without a materially new mechanism.
- **EXP-100 inconclusive-revise**: width 9 validated the shared-context architecture at 91.77% transition / 56.8% rollout. Width 16 learned to 64.31% transition but only 0.52% rollout with 54.82% invalid-state rollouts. It missed the continuation gate; do not promote or consult public Tier 3.
- **Start EXP-105**: test learned width-9-to-width-16 initialization against the exact random-initialized EXP-100 arm. This targets the observed optimization/state-validity bottleneck while changing only initialization.
- **EXP-105 pass**: learned transfer improved width-16 transition from 64.31% to 87.25%, rollout from 0.52% to 24.12%, and invalid rollouts from 54.82% to 11.18% at matched steps. Advance to the preregistered 1,200-step random-versus-transfer comparison; do not evaluate public Tier 3 yet.
- **EXP-106 select transfer, revise curriculum**: at 1,200 matched steps, transfer wins 94.335%/47.07% transition/rollout versus 87.76%/23.48% random, but misses the 97%/60% full-budget gate. A 100,000-case diagnosis localizes errors to bit-1, quotient-2 reductions; test that curriculum next.
- **EXP-110 pass, advance to 2,400**: quotient-2/boundary hard mining reaches 99.095% transition and 87.41% rollout, improving matched uniform training by 4.76/40.34 points and reducing invalid rollouts to 0.72%. It clears the full-budget gate but not the 90% independent gate.
- **EXP-111 pass, authorize independent Tier 3**: the full 2,400-step curriculum reaches 99.908% heldout transition, 98.895% heldout rollout, and 0.145% invalid rollouts. All frozen gates pass.
- **EXP-112 pass, promote to integration**: the frozen checkpoint scores 499/500 across five official-generator Tier-3 seeds; every seed passes and the worst seed scores 99%. Checked public Tier-3 answers remained untouched during selection.
- **Promote `candidate-002-tier3-hybrid`**: the full independent 1,100-case official evaluation scores 31.0% overall with Tier 1/2/3 at 100%/97%/99%, advancing `highest_tier_above_90` from 2 to 3. Static analysis, preprocessing isolation, determinism, and the 300-second timeout gate pass. A post-selection checked release regression scores 31.3% overall and 100% Tier 3; randomized weights collapse Tier 3 to 2%.
