---
type: Experiment Preregistration
title: EXP688 Compact Hidden-Trajectory Distillation
description: Matched hidden56 logit-distillation arms differing only by normalized selected-channel final-hidden alignment.
status: parked-calibration-runtime-headroom
tags: [autoresearch, W256, compression, hidden-distillation, Candidate005]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP688 compact hidden-trajectory distillation

EXP688 reuses EXP687's answer-independent ranked 56-of-92 channel map and byte-identical
`f9938a97...b340` pruned initialization. Both trained arms freeze the same hidden92 teacher,
repeat `[64,128,256,256]` for 2,400 planned updates, and compute eight exact losses, residue
logit MSE at weight 0.10, quotient KL at weight 0.05, and normalized selected-channel
teacher/student final-hidden MSE per update. Both arms capture and align identical hidden tensors;
only the final coefficient is 0.05 for `hidden_aligned` and 0.0 for `logit_only`.

Native PyTorch directly repeats the authenticated `recurrent_update` sequence and returns the
final recurrent tensor during training. Focused parity proves logits and quotient logits are
source-exact at W64/W128/W256. Formal inference remains EXP687's compact student-only rollout;
no teacher, capture, exact arithmetic, pruning operator, or answer selection enters inference.

## Terminal result

The rootless all-mode smoke passed in 70.37 wall / 170.62 process CPU seconds with 669,324 KiB
peak RSS. Historical lifecycle checks passed 35/35; the combined focused cone passed 53/53.
The sole package oracle passed validation and 779/779 tests in 535.35 wall / 805.17 process CPU
seconds with 940,508 KiB peak RSS.

Root 20261306 completed all five routes, exact reload, equal hidden work, warm-up, and three cold
complete label-free 1,100-case candidate routes. Cold walls were 150.457, 147.061, and 146.250
seconds. The frozen 2.0 measured-variability headroom raises the maximum to 300.913 seconds,
exceeding the 285-second gate. All other resource and integrity checks passed. EXP688 therefore
parks before root 20261307; roots 20261308/09 also remain unopened. The 16-transition/two-rollout
calibration denominators are non-scientific, so there is no W128/W256 mechanism verdict or delta.
