---
type: submission-candidate
title: Candidate 006 Tier-7 EXP701 integration
description: Strongest observed candidate and pinned-rules runtime qualifier; current upstream pre-reduction rules block submission approval.
status: runtime-qualified-compliance-blocked
date: 2026-08-12
timestamp: "2026-08-12T08:04:41Z"
tags: [candidate006, tier7, exp701, runtime-validated]
---

# Candidate 006: Candidate005 plus EXP701 Tier-7 specialist

Candidate006 is the strongest observed research artifact. It preserves Candidate005 through 128-bit moduli, routes 129-256-bit moduli to EXP701's frozen `teacher_cosine` H92 specialist, and keeps the zero fallback above 256 bits. EXP706 changes only inference execution; its manifest and all 569,873 trained parameters are byte-identical to the original Candidate006 artifact.

## Evidence

- Five matched full generator seeds score 64.7%, 64.6%, 66.1%, 66.1%, and 64.9%, beating Candidate005 by 5.2-6.6 percentage points on every seed.
- Tier 7 aggregates to 319/500 exact (63.8%).
- All 3,000 Tier 1-6 predictions match Candidate005 exactly.
- A randomized Tier-7 checkpoint scores 2/100, while the trained specialist scores 54/100 on the same seed.
- The recovered pinned-sandbox result is valid, deterministic, and scores 64.1% overall with Tier 7 at 54%.
- The EXP706 runtime-fused artifact completed all tiers in the same official 4-CPU/8-GiB sandbox in 281.6 inference seconds. Tiers 1-7 retained the baseline accuracy vector exactly; completing Tiers 8-10 added six edge-case answers and raised overall accuracy to 64.7%.
- A separate checked-public run completed in 270.86 inference seconds at 66.6% overall and 73% on Tier 7.

The exact runtime-qualified artifact is stored under [`experiments/runs/exp-706-candidate006-exact-runtime-fusion/artifact`](../../experiments/runs/exp-706-candidate006-exact-runtime-fusion/artifact). Its model, manifest, and weights SHA-256 values are `082ca41d86c758254dd0e75841c1b1d8df94bbf3280d539389782deaf4857d9d`, `dbeb95ed3b085aa2b338ec57041d343ec924ca72f4b3c7fd3094d6cd215a0c91`, and `a7d25ee4eed49c5f625bc8369447be9f51ff1be54766631d8b96ab725860553e`.

## Runtime gate clarification

The official written limit is 300 seconds. The earlier 285-second value first appeared as a repository-internal 15-second reliability margin; it is not an official competition rule. EXP706 binds its runtime decision to the actual 300-second limit and clears it twice in the pinned official container (264.2 and 281.6 seconds observed), with the slower durable run retained here.

## Current-rules blocker

The current upstream rules at commit `99cac6ef5c2f82e53105ec0ddcbb9b8d37bf6fca` explicitly prohibit deterministic reduction of full-width operands against the actual modulus. Candidate006 computes `int(a_enc) % p` and `int(b_enc) % p` before its learned networks run. Therefore the artifact is runtime-qualified against the pinned evaluator but is not approved for current-rule submission. Fixing this requires a model trained to consume raw operands; deleting those reductions from the frozen model would sacrifice its established accuracy.

No private benchmark, 85% accuracy, or private-seed generalization claim follows from this result.
