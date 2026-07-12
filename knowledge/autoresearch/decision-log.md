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
- **EXP-200 infrastructure retry**: the first width-32 attempt was OOM-killed before training because the inherited sampler exhaustively enumerated through `2^32`. Mark both MLflow runs killed and invalid; replace only enumeration with bounded deterministic, identity-disjoint sampling.
- **EXP-200 pass, advance width 32**: at matched 400-step compute and hard-mined data, width-16 transfer improves transition from 44.12% to 98.21% and rollout from 0% to 61.68%, with invalid rollouts at 2.12%. This passes every preregistered gate and authorizes a full-budget Tier-4 run; it does not authorize candidate promotion or checked-public evaluation.
- **EXP-201 narrow miss, authorize one confirmation**: 1,600 steps reach 99.648% transition, 90.465% rollout, and 0.51% invalid rollouts. Do not relax the 99.7% transition gate; the positive late slope and 0.052-point miss justify one preregistered 2,400-step run before changing the intervention.
- **EXP-202 pass, authorize independent Tier 4**: 2,400 steps from the same frozen width-16 initialization reach 99.814% transition, 94.49% rollout, and 0.15% invalid rollouts. All unchanged gates pass; freeze the checkpoint and evaluate five official-generator Tier-4 seeds without consulting checked answers.
- **EXP-203 fail, retain Candidate 002**: the frozen width-32 checkpoint scores 441/500 (88.2%) across five official-generator Tier-4 seeds, with only one seed at 90% and a worst seed of 87%. The failure is consistent with underweighting 30–32-bit primes during identity-balanced training; test an official-distribution correction on fresh identities without reusing EXP-203 seeds.
- **EXP-204 reject simple distribution correction**: paired official-like heldout rollout improves 84.60% -> 89.97%, confirming the high-bit sampling mismatch matters, but misses the frozen 94% absolute gate; transition reaches 99.666% versus 99.8%. Do not spend more steps on the same fine-tune. Test scheduled learned-state/rollout-aware training next.
- **EXP-210 reject scheduled learned-state formulation**: at matched transition-equivalent budget, the challenger improves official-like rollout 90.72% -> 91.425% and slightly improves transition, but the +0.705-point gain misses the frozen +2-point gate and 92% absolute threshold. The stable result is insufficient; proceed to the materially different width-independent architecture track.
