---
type: Experiment Preregistration
title: EXP686 W256 Straight-Through Specialist
description: Matched W256 continuation test of learner-exposed hard-forward sigmoid state against teacher forcing.
status: parked-calibration-candidate-runtime-headroom
tags: [autoresearch, W256, straight-through, teacher-forcing, Candidate005]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP686 W256 straight-through specialist

EXP686 isolates exposure bias at width 256. The primary and teacher-forced arms both start
from Candidate005 checkpoint `cbc4a1bc...d4bd`, receive the same 800 batches, 16 examples
per batch, optimizer schedule, eight supervision calls and one optimizer step per update.
Only recurrent state exposure differs: the primary feeds hard-forward sigmoid predictions
back through a straight-through gradient estimator, while the control receives the exact
teacher state. Frozen-source, reset and randomized controls test inherited-weight dependence.

Formal inference uses trained `SharedDilatedConvGRU` weights only. The tensor-native W256
state cache must reproduce the legacy output and minimum-margin tensors exactly. An actual
1,100-case candidate-shaped route sends widths through the frozen incumbent at <=128, the
specialist at 129-256, and the preregistered zero fallback above 256; labels and answers are
unavailable to that timed path.

Root 20261294 alone performs complete-route calibration and resource admission. Root
20261295 may open only if every integrity/resource conjunct passes, and must reach at least
85% exact W256 rollout before roots 20261296/97 can confirm. No root reuse, repair chain,
retuning, arbitrary cap, reduced fidelity, or protected benchmark access is authorized.

## Terminal result

Rootless smoke completed every mode and route. The focused suite passed 17/17. The sole
package oracle passed validation and 743/743 tests. Root 20261294 then completed matched
training, all controls, reload, exact tensor-native parity, the full 1,100-case candidate
route, memory, expected-value, and no-fidelity-reduction checks.

Admission nevertheless failed its frozen candidate-runtime conjunct: 220.102 nominal wall
seconds becomes 398.966 seconds under measured 1.81264 headroom, above the 285-second cap.
Expected formal wall (8,101.734 seconds) and peak RSS (4,718,037,723 bytes) remain within
their ceilings. Roots 20261295-20261297 remain unopened. This is a resource park, not a
scientific verdict on learner-state straight-through training or W256 accuracy.
