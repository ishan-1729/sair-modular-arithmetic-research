---
type: Ablation Study
title: EXP-673 DeepLoop proactive parameterization
description: Terminal topology-feasibility park before implementation, calibration, or scientific evaluation.
status: parked
tags: [autoresearch, exp-673, DeepLoop, feasibility]
timestamp: "2026-07-29T09:15:00+05:30"
---

# EXP-673 - DeepLoop proactive parameterization

EXP673 authenticated and visually reviewed the required DeepLoop and supporting LayerNorm-dynamics PDF
pages, froze the exact source equations, Candidate005 checkpoint/topology, width-dependent visits, arms,
roots, diagnostics, gates, identities, and resource contract, then applied the explicit topology
feasibility stop rule.

Candidate005 is a pretrained PreNorm/GroupNorm dilated ConvGRU with an unnormalized gated residual
update. DeepLoop is a PostNorm residual parameterization whose `alpha` changes the skip inside
normalization and whose `beta` is a one-time residual-matrix initialization gain. Candidate005 also has
14 fully gradient-visible visits at W128 and 16 at W256 but no separate H/L hierarchy or detached outer
cycle. An exact-function mapping would need a non-source bypass/zero gate, a cancellation that makes
alpha inert, or generic runtime beta/alpha correction scaling. None can causally test ST-M030.b, and the
last is the expressly excluded reactive/generic visit-scaling substitute.

Decision: park before code, focused behavioral tests, package oracle, root `20261242` calibration,
training, formal W128/W256, Jacobians, candidate timing, and roots `20261243`-`20261245`. MLflow run
`969121a5c43d4847a7c3ac5ca81f69e7` records the local terminal feasibility outcome. This is not a
scientific rejection of DeepLoop or ST-M030.a/b; a future test would require an ordinary-initialized
PostNorm recurrent architecture with a single fixed source depth, which is a different experiment.
