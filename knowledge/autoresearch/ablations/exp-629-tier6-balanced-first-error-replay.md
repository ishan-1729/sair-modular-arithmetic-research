---
type: Experiment Preregistration
title: EXP-629 Tier-6 Balanced First-Error Replay
description: Matched low-rate full-cell continuation with uniform or model-mined exact first-error replay.
status: completed
tags: [autoresearch, tier6, hard-mining, replay, autonomous-rollout]
timestamp: "2026-07-13T12:05:00+05:30"
---

# EXP-629 Tier-6 balanced first-error replay

## Question and hypothesis

The frozen EXP-609 checkpoint is about 99.8% transition-exact but only roughly 81–89% rollout-exact across fresh width-128 samples. EXP-613 showed that a separate high-rate additive corrector trained on repeated first errors is destabilizing, while earlier autonomous-state recovery showed that the full recurrent cell can improve trajectory robustness. Can low-rate continuation of the full cell, with a minority of exact pre-error transitions and majority uniform replay, remove enough of the sparse transition tail without harming ordinary dynamics?

The falsifiable hypothesis is that balanced first-error replay improves fresh width-128 rollout by at least three exact points over an update-, example-, initialization-, optimizer-, and identity-matched uniform replay control, reaches at least 90%, and retains Tier-5 behavior.

## Mining and matched comparison

The frozen source model autonomously rolls out 1,200 newly generated width-128 training cases over 2,048 training prime identities. For each failed trajectory, the challenger stores only the first erroneous transition. Its input state is still exactly the teacher state, so its target is the legal next modular state rather than an ambiguous repair from an already-corrupted state. The paired control record comes from a position 64 steps away in the same `(left, right, modulus)` trajectory and is generated from the exact offline teacher.

Both arms start from the same checkpoint and run 300 full-cell AdamW updates with batch size 64. Every fourth update is a uniform width-64 retention batch. Other batches contain 48 fresh uniform width-128 transitions and 16 replay records. The control draws the replay records from the paired ordinary positions; the challenger draws the corresponding first-error records using the same replay indices. The online uniform samplers, seeds, schedules, learning rate, weight decay, and example counts are identical. Exact arithmetic is confined to offline sampling, labels, mining, tests, and evaluation truth.

Mining must yield at least 128 unique first errors. A lower yield is a feasibility failure; it will not be repaired by repeating a tiny set more heavily or by changing the gate after observation.

## Evaluation and gates

Each frozen endpoint is evaluated on the same fresh heldout prime identities with 3,000 transitions and 300 autonomous rollouts at widths 64 and 128. Advance only if all gates pass:

- challenger width-128 rollout is at least 90%;
- paired width-128 gain over control is at least three exact percentage points;
- challenger width-64 rollout is at least 95%;
- challenger width-128 transition accuracy is at least 99.8%;
- challenger invalid width-128 rollout rate is at most 1%.

A pass authorizes a higher-fidelity fresh confirmation, not an independent, checked, official, or candidate evaluation. A close result may justify only a preregistered confirmation when sampling uncertainty and the observed paired outcomes support it; the present gate will not be retroactively relaxed.

## Dynamic resource budget

Valid cached evidence from EXP-613 uses the identical checkpoint and width-128 training distribution: mining 600 trajectories produced 96 first errors in 406.26 wall seconds and 1,579.69 aggregate CPU seconds. This projects 812.52 wall seconds for 1,200 trajectories and about 192 first errors. A new representative batch-64 full-cell profile took 43.861 wall and 166.128 aggregate CPU seconds for 20 updates. Six hundred matched arm-updates therefore project to 1,315.84 wall and 4,983.84 aggregate CPU seconds. EXP-609's identical-model 3,000-transition/300-rollout evaluation took about 196.16 wall seconds per arm, projecting 392.32 wall and 1,565.45 aggregate CPU seconds for both arms.

Including 150 wall seconds for setup and artifacts gives an expected 2,671 wall seconds and roughly 162 aggregate CPU minutes. The planning budget is 4,000 wall seconds (about 50% headroom); the hard runaway ceiling is 6,500 wall seconds, with a four-thread aggregate-CPU upper bound of 433.34 minutes. These are explicitly different units. Current host contention may increase wall time without equivalent aggregate CPU growth. Mining batches and 25-step arm states are resumable, completed checkpoints are saved before evaluation, and valid caches will be reused. Training or evaluation fidelity will not be reduced merely to fit a shorter wall cap.

## Boundaries and limitations

No checked, consumed independent, or private answer is used. The inference architecture and parameter count are unchanged, so a winning checkpoint remains trained-weight-dependent and tensor-native with the incumbent runtime shape. This experiment tests one balanced replay ratio at a preregistered low rate; it does not establish that all hard-example curricula work or that prime-identity robustness is solved.

## Infrastructure attempt

MLflow run `39cd3ebbef1c4da28d5204f7b922557c` failed before mining because the runner's 15-second `git status` provenance timeout was too short under host filesystem contention. It produced no dataset, checkpoint, or metric. The timeout alone was raised to 120 seconds before evidence generation; the hypothesis, data, training, evaluation, gates, and resource budget remain unchanged.

## Result and decision

Mining produced 153 unique first errors from 1,200 fresh training trajectories (12.75%) and passed the minimum-128 feasibility gate. The matched uniform-replay control reached 263/300 (87.67%) width-128 rollout, 2,994/3,000 (99.8%) width-128 transition, 292/300 (97.33%) width-64 rollout, and 1% invalid width-128 rollouts.

Balanced first-error replay regressed width-128 rollout to 259/300 (86.33%, −1.33 points), missed the transition gate at 2,993/3,000 (99.767%), and reduced width-64 rollout to 289/300 (96.33%). Its 0.33% invalid rate and width-64 absolute retention passed, but the primary absolute, causal-gain, and transition gates failed.

Reject and close this formulation. Exact pre-error replay remains genuinely difficult—the hard-mixed batches stayed around 78–86% exact with frequent clipped gradients—but concentrating 25% of width-128 batches on 153 model-selected transitions harms fresh autonomous behavior even when the full cell and majority uniform replay are used. Do not tune the replay ratio on this observed set. The complete run took 5,193.61 runner wall seconds and 19,440.43 aggregate CPU seconds, within the 6,500-second hard ceiling; externally measured elapsed time was 4,959 seconds under contended WSL scheduling.
