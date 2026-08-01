---
type: Experiment Preregistration
title: EXP679 Operational Capacity Recycling
description: Frozen M043.b reduced-call and exact-embedded-width resource-selected comparison around Candidate005.
experiment: EXP679
status: permanent-pre-root-denominator-park
tags: [autoresearch, exp679, capacity-recycling, candidate005]
timestamp: "2026-07-30T02:00:00+05:30"
---

# EXP679 capacity recycling

Before profiling, freeze W128 recurrent-call counts `{8,10}`, total hidden widths `{104,112}`, roots `20261266`-`20261269`, six arms, 400 matched updates, data order, denominators and gates. Root 20261266 profiles every grid and control route, trains/evaluates the complete calibration route, and selects without labels. Park if no exact-embedded point is no slower than guarded frozen Candidate005 and inside RSS/activation headroom.

If admitted, root 20261267 trains and caches capacity-recycled, reduced-only, wider-full, standard-full and randomized-wider arms, plus the frozen source. All trained arms receive identical examples, optimizer settings, update counts and data-order seeds. Score the W128 primary first. Reject before W256 if retention, exactness, validity, reload, extra-channel gradient/ablation dependence, randomized-weight dependence or runtime fails. W256 requires at least 10% exact and five points over reduced-only and standard-full. Roots 20261268/69 open only after every primary conjunct passes.

## Terminal verification-feasibility outcome

The exact preserved config/model/runner/test hashes are `7718c560...ae850`, `0cb923c3...3982`, `376c32eb...0583f`, and `a36c3256...1318`. Eight focused tests and the final-byte project validator were accepted from T078. T079 then completed the first full project suite at 663/663 in 560.35 pytest seconds (548.40 outer wall, 712.07 aggregate process CPU seconds, 943,808 KiB RSS) under the 1,684.13765625-second dynamic guard.

Before opening root `20261266`, freeze inspection found that the verified runner requires `evaluation.trajectory_loss_cases` and `evaluation.calibration_trajectory_loss_cases`, while neither key exists in the verified config. Adding them would change the sole fully verified config identity. T079 authorizes no post-oracle repair or recovery successor, so roots `20261266`-`20261269`, MLflow, training, calibration, capacity selection, W128, W256, and confirmations remain permanently unopened. This is a verification-feasibility park, not scientific evidence for or against M043.b or operational capacity recycling.
