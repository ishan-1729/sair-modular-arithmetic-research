---
type: Experiment Preregistration
title: EXP687 Compact Distilled W256 Specialist
description: Frozen high-incident hidden56 channel transfer with matched exact-only and frozen-teacher-distilled recovery arms.
status: parked-package-verification
tags: [autoresearch, W256, compression, distillation, pruning, Candidate005]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP687 compact distilled W256 specialist

EXP687 freezes an answer-independent 56-of-92 channel map before training. Each source hidden
channel receives the sum, across stem, normalization, gate/candidate input and output axes,
local head, and both quotient-head layers, of its squared incident parameters divided by the
mean incident square for that term. Channels are ranked descending with ascending source index
as the tie-break. The same selected indices slice every compatible hidden axis. Native PyTorch
`index_select` keeps this fixed graph auditable without adding a pruning dependency or inference
runtime. The resulting compact model has 38,588 parameters versus 99,824 in the teacher.

Both trained arms start from byte-identical compact weights and repeat widths
`[64, 128, 256, 256]` for 2,400 updates. Every update has eight straight-through learner-state
calls, eight exact losses, and eight frozen-teacher calls. Both arms compute normalized residue
logit MSE and quotient KL; only the distilled arm applies weights 0.10 and 0.05, while exact-only
multiplies them by zero. The frozen teacher and exact arithmetic are absent from formal inference.

Root 20261302 was preregistered for four-update all-route calibration, warmup, and three cold
complete 1,100-case candidate-shaped timings. Root 20261303 could open only after full resource
admission, and roots 20261304/05 only after complete >=85% W256 evidence plus every causal,
retention, and runtime conjunct.

## Terminal result

The real rootless smoke completed both trained arms, all five routes at W128/W256, save/reload,
randomized/reset dependence, one candidate-route case, admission math, and all terminal
serializers in 63.71 seconds. The focused suite passed 18/18 in 7.53 seconds. The single package
oracle passed project validation (218 OKF concepts and 1,100 benchmark records) and 760 of 761
tests. Its only failure was EXP686's placeholder guard observing a pre-existing ignored
`experiments/runs/exp-686-w256-straight-through-specialist/run.json` in the reused fixture
worktree. That file is outside T092's allowlist.

The frozen one-oracle and out-of-scope stop rules therefore park EXP687 before root 20261302.
Removing the EXP686 fixture, weakening the suite, or running a second oracle is not authorized.
This is a verification park, not evidence for or against compact distillation. No complete
candidate timing, calibration, formal science, confirmation, MLflow root, protected benchmark,
candidate, submission, promotion, or release surface opened.
