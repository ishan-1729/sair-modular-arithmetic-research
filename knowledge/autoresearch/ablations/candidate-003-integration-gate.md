---
type: Candidate Preregistration
title: Candidate 003 Tier-4 Cellular Hybrid Integration Gate
description: Frozen full-artifact release gate for the independently validated Tier-4 cellular specialist.
status: preregistered
tags: [autoresearch, candidate, integration, Tier-4, release]
timestamp: "2026-07-12T12:10:00+05:30"
---

# Candidate 003 integration gate

Freeze the three-file submission artifact with combined weights SHA-256 `a991c389aec5f8c4e8153846b07d7d1207bdd0eaf192ff11ca036663f056a514`. It combines the retained Candidate-002 Tier-2/3 tensors with frozen cellular Tier-4 checkpoint `a07bcfc9b7f990a7a5b62001fb40ab4533c4776d9317099808bd60f56b0be241`. No training or selection occurs after this freeze.

First evaluate all 1,100 cases with independent seed `a791075e4357a6bc629b8753d1e1b05fcc9003c3038c72d9becca9cc5e4d41c2`. Require at least 40% overall, Tier 1/2/3/4 at 99%/95%/95%/90%, `highest_tier_above_90 = 4`, and inference below 240 seconds. Only after those gates pass may the checked-public release regression be read.

The exact artifact must then pass manifest/static checks, repeated determinism, a randomized-weight dependence control that collapses Tier 4, pinned-runtime evaluation, the official isolated container, artifact-size audit, and submission-boundary review. Candidate 002 remains incumbent until every gate passes. Candidate 003 is still only an intermediate checkpoint under the 85% full-overall program requirement.
