---
type: Experiment Preregistration
title: EXP-639 Full-Depth Soft-Boundary Adaptation
description: Shared-weight width-64/128 macro-recovery continuation testing whether continuous intra-macro state crosses the Tier-6 boundary.
status: completed
tags: [autoresearch, tier6, cellular, macro-transition, continuous-state, straight-through, recovery]
timestamp: "2026-07-17T12:44:00+05:30"
---

# EXP-639 full-depth soft-boundary adaptation

## Question and causal surface

EXP-638 shows that two full-depth calls to the frozen EXP-609 cell can be joined by a continuous intermediate state without losing width-64 transition exactness. On training identities, the continuous macro scores 95.5% rollout versus 95.0% for two ordinary hard transitions. Can low-rate adaptation through this two-bit computation raise fresh width-128 rollout above 90% while preserving width 64 and the original parameter count?

The challenger is one shared 99,824-parameter `SharedDilatedConvGRU`; there are not separate high/low phase weights. Within each macro, the first full two-cycle call consumes the high bit. Its probabilities become the continuous state plane for the second full call and low bit. Only the second output is hard decoded at inference. During macro-trajectory training, the outer hard state uses hard-forward/sigmoid-backward straight-through credit, matching the supported EXP-608 mechanism at half as many outer state boundaries.

The matched controls are the immutable source checkpoint under two frozen inference modes: ordinary hard composition and unadapted full-depth continuous macro. They receive the exact same fresh cases as the trained challenger. Controls are not retrained because the causal question is whether adaptation changes the viable continuous initialization; completed source training is valid cached evidence. This rung changes training objective and state boundary together and does not isolate each auxiliary loss.

## Trained-parameters boundary and frozen training

Source checkpoint SHA-256 is `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`. Inference receives only legal state, right-operand, modulus, multiplier-bit, and marker planes. Trained tensor operations produce both learned phase outputs. No exact modular multiplication, division, reduction, CRT, repair, or benchmark lookup appears in any candidate-shaped arm.

Exact bigint arithmetic is confined to offline prime sampling, labels, recovery starting states, hard-example truth, and evaluation truth. For a macro step, offline supervision supplies exact high-bit intermediate and low-bit final states plus their quotient classes.

Train exactly 400 AdamW/cosine updates with seed `20260839` and schedule `[transition64, recovery64, recovery128, recovery128]`. Transition updates use batch 128 random width-64 macro transitions. Recovery updates use batch 16 and four consecutive macro steps (eight multiplier bits), starting from exact offline states at aligned even bit positions. Each macro loss is final exact-bit/worst-bit loss plus half-weight intermediate exact-bit/worst-bit loss; quotient supervision is used only on independent transition updates. Straight-through state is used only between macros. Learning rate is `5e-6` to `5e-7`, weight decay `2e-5`, and gradient clip 2.0. Save exact model, optimizer, scheduler, all sampler/start RNGs, Python/NumPy/Torch RNG, history, wall, and aggregate CPU every 25 updates.

## Fresh matched evaluation and gates

Create new width-64/128 train/test prime splits with 512/128 identities per width. Neither EXP-638 training identities nor any checked, independent-full, official, or private answer is used for selection. On the test identities, evaluate 3,000 paired macro transitions and 300 paired multiplication rollouts at widths 64 and 128 for frozen hard composition, frozen continuous initialization, and trained continuous macro. Record exactness, validity, all-step exactness, arm-only successes, disagreement, wall, and aggregate CPU. Repeat 20 trained width-128 cases with every intermediate/final logit retained.

Advance only if every gate passes:

- trained width-128 rollout is at least 90%;
- trained width-128 rollout is at least two points above the better frozen arm;
- trained width-64 rollout is at least 95%;
- trained width-128 macro-transition exactness is at least 99.8%;
- trained width-128 invalid rollout is at most 1%;
- trained continuous width-128 wall time is at most 1.15 times frozen hard composition;
- predictions, intermediate/final logits, and validity repeat exactly.

Passing authorizes a separate Tier-6 independent-seed confirmation and candidate integration gate; it is not itself candidate promotion or evidence toward the complete 85% overall goal. Failure closes this exact shared-weight, four-macro, low-rate adaptation without retrospective schedule or loss-weight tuning.

## Dynamic resource-budget protocol

Before formal execution, run five actual optimizer updates for each declared mode on training identities, including two full learned phases, exact intermediate/final objectives, four-macro unrolling, and straight-through outer state. Measure all three evaluation-shaped arms on 96 training-identity transitions and six training-identity rollouts at widths 64 and 128, plus determinism-shaped work. Project the exact 100/100/200 update counts implied by the schedule and every evaluation/control/repeat case. Record wall and aggregate CPU separately, add explicit setup/tracking/checkpoint headroom, and retain a hard runaway ceiling.

Calibration may change only measured/projection/limit fields under `resource_budget` and the measured paragraph in this section. It may not change checkpoint, seed, identities, architecture, temperature, training modes, updates, batch sizes, recovery horizon, optimizer, learning rate, losses, cases, arms, or gates. Formal execution is forbidden while calibration is pending. Reuse hash-valid partial training/evaluation blocks, save the trained checkpoint before evaluation, and never reduce fidelity to fit an arbitrary short wall cap.

The completed training-identity-only calibration measured five full optimizer updates per unique mode. Projecting the exact schedule counts gives 1,395.077 wall and 5,216.318 aggregate CPU seconds for all 400 updates. Complete three-arm evaluation-shaped profiles at both widths project 2,256.691 wall and 8,432.151 aggregate CPU seconds for all 3,000-transition, 300-rollout, and determinism cases. With the frozen 180-second setup/tracking/checkpoint allowance, expected total is 3,832 wall seconds and 234 aggregate CPU minutes. The soft budget is 5,220 wall seconds; the hard runaway ceiling is 7,680 wall seconds with a distinct four-thread upper bound of 511 aggregate CPU minutes.

External calibration timing was 122.23 wall seconds, 442.60 user CPU seconds, and 5.95 system CPU seconds at 366% CPU, with 1,468,168 KiB maximum RSS. Calibration JSON and console hashes are `6976f4b654d254a1b03a442411d7a69edbdf51d698f9000d5778754dbb7c9df5` and `7aa5444b020513c1f42571b73ab463316cbeec924358715da6dc605c952b9d98`. No held-out identity was read and no scientific field changed.

## Limitations

EXP-638's gain is one net training-identity case and may not generalize. Shared phase weights constrain specialization but preserve parameter count and causal clarity. Full-depth macro computation does not halve convolutional work; it halves only outer hard state boundaries and may gain modest Python overhead savings. Even a Tier-6 pass remains far below the binding full-overall completion target and requires independent confirmation.

## Result and decision

Rejected after the complete frozen run. At width 128, hard-frozen, initialized-continuous, and trained-continuous rollout score 248/300 (82.67%), 245/300 (81.67%), and 242/300 (80.67%). The trained arm therefore misses the 90% absolute gate and regresses two points versus the better frozen arm. Its 2,992/3,000 (99.733%) transition score is one case below the 99.8% gate. It is valid and exactly deterministic, but its isolated 1.509 hard-runtime ratio also misses. Width-64 retention passes at 288/300 (96.0%).

All 400 updates and every declared case completed. Runner timing is 2,312.41 wall seconds and 8,531.04 aggregate CPU seconds (142.18 minutes), below the calibrated 3,832-second expectation, 5,220-second soft budget, 7,680-second hard ceiling, 234-minute expected CPU work, and 511-minute CPU upper bound. GNU time separately records 2,174.28 elapsed seconds, 8,520.47 user seconds, and 24.81 system seconds; the unexplained 138.13-second wall discrepancy is retained. No fidelity was reduced and the trained checkpoint/evaluation cache are hash-preserved.

Close this shared-weight low-rate soft-boundary adaptation. Do not tune its schedule or loss weights after seeing the result. The frozen hard path remains stronger. A materially different next diagnostic may exploit the legal commutativity of multiplication to test whether operand-swapped learned trajectories provide complementary exact outputs; that hypothesis must use consumed evidence first and a fixed target-free selector.
