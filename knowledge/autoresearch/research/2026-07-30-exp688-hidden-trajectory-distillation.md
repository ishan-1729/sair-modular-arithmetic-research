---
type: Mechanism Map
title: EXP688 Hidden-Trajectory Distillation Mechanism Map
description: Exact hidden capture, selected-channel alignment, matched-work, and inference-boundary evidence for EXP688.
status: parked-calibration-runtime-headroom
tags: [autoresearch, hidden-state, distillation, W256, compression]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP688 hidden-trajectory distillation mechanism map

The frozen teacher is Candidate005/EXP609 checkpoint `cbc4a1bc...d4bd`. EXP688 imports EXP687's
answer-independent incident-weight ranking and coherent slice unchanged. The selected-channel
map hash remains `92d31507...8cc0`, the hidden56 pruned state remains `f9938a97...b340`, and each
student has 38,588 parameters.

For each of eight recovery transitions, the student and no-gradient teacher run the exact stem,
dilation schedule, authenticated recurrent-update method, output normalization, local head, and
quotient head. The teacher's final recurrent tensor is indexed by the frozen 56 channels. Both
student and selected teacher tensors are L2-normalized over channels before MSE. Each arm computes
the same exact, residue-logit, quotient-KL, and hidden losses and the same hidden-gradient probe.
Only the preregistered hidden coefficient differs: 0.05 versus 0.0. Calibration records 32 teacher,
student, capture, alignment, and exact-loss calls per trained arm.

Hidden capture is training-only. The candidate path cold-loads only a compact student checkpoint
and calls the inherited learned rollout. It carries no teacher, hidden alignment, answers, repair,
filter, or hand-coded reduction. Three complete candidate routes were deterministic in route size
and outputs but failed guarded runtime admission at 300.913 seconds versus 285. Formal science
therefore did not open, so the internal-alignment hypothesis remains scientifically untested.
