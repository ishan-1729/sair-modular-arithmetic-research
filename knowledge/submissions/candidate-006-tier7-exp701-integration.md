---
type: submission-candidate
title: Candidate 006 Tier-7 EXP701 integration
description: Strongest observed candidate, retained outside submission because its runtime promotion gate is unproven.
status: research-challenger-not-promoted
date: 2026-08-12
timestamp: "2026-08-12T08:04:41Z"
tags: [candidate006, tier7, exp701, runtime-gated]
---

# Candidate 006: Candidate005 plus EXP701 Tier-7 specialist

Candidate006 is the strongest observed artifact in this repository, but it is not the release incumbent. It preserves Candidate005 through 128-bit moduli, routes 129-256-bit moduli to EXP701's frozen `teacher_cosine` H92 specialist, and keeps the zero fallback above 256 bits.

## Evidence

- Five matched full generator seeds score 64.7%, 64.6%, 66.1%, 66.1%, and 64.9%, beating Candidate005 by 5.2-6.6 percentage points on every seed.
- Tier 7 aggregates to 319/500 exact (63.8%).
- All 3,000 Tier 1-6 predictions match Candidate005 exactly.
- A randomized Tier-7 checkpoint scores 2/100, while the trained specialist scores 54/100 on the same seed.
- The recovered pinned-sandbox result is valid, deterministic, and scores 64.1% overall with Tier 7 at 54%.

The exact artifact is stored under [`experiments/runs/exp-704-candidate005-tier7-exp701-integration/candidate006`](../../experiments/runs/exp-704-candidate005-tier7-exp701-integration/candidate006). Its model, manifest, and weights SHA-256 values are `b7c255af7b619e6f689a84e5307ce445432e4285af0cdd9f514d982ebc8c0315`, `dbeb95ed3b085aa2b338ec57041d343ec924ca72f4b3c7fd3094d6cd215a0c91`, and `a7d25ee4eed49c5f625bc8369447be9f51ff1be54766631d8b96ab725860553e`.

## Why it is not `submission/`

The preregistered promotion rule required an isolated inference receipt at or below 285 seconds. The official evaluator completed Tier 7 because its 300-second timeout is checked only between batches, but the outer wrapper timed out before publishing exact runtime metadata. A cache-only follow-up also failed to finish the 100-case Tier-7 batch within 300 local seconds. Candidate005 therefore remains the release incumbent until Candidate006 is distilled or otherwise made deployably faster without losing its predictions.

No private benchmark, 85% completion, or release claim follows from this result.
