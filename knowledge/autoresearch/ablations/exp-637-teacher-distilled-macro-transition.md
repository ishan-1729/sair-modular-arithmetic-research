---
type: Experiment Preregistration
title: EXP-637 Teacher-Distilled Macro Transition
description: Matched viability test of a two-phase compressed radix-4 cellular transition with exact intermediate supervision and frozen-teacher logits.
status: preregistered
tags: [autoresearch, radix4, distillation, cellular, macro-transition, intermediate-supervision]
timestamp: "2026-07-17T11:58:00+05:30"
---

# EXP-637 teacher-distilled macro transition

## Question and causal contrast

EXP-630 established that a direct learned map from `(state, right, modulus, two-bit digit)` to the radix-4 successor is fast but not viable: after 1,600 updates it reaches only 37.5%, 22.9%, and 13.95% transition exactness at widths 16, 32, and 64, and zero exact rollout at every width. Its roughly 0.50 runtime ratio nevertheless confirms that halving the outer recurrent horizon is operationally valuable.

EXP-637 tests whether the missing ingredient is explicit two-step structure rather than the radix itself. The student contains two separately parameterized one-cycle cellular phases inside one macro call. Phase 1 consumes the high digit bit and predicts an intermediate bit-logit state. Phase 2 consumes that continuous differentiable state and the low digit bit. Only the phase-2 output is hard decoded between macro calls. Thus a width-`W` multiplication has `W/2` autonomous hard-state boundaries, while the two one-cycle phases together execute the same number of dilation sweeps as one ordinary two-cycle radix-2 transition.

The exact matched contrast is the frozen EXP-630 direct checkpoint. It used the same seed, train/test prime identities, sampler seeds, widths, 1,600 updates, batch size, optimizer family, and evaluation case generators. Completed direct training is reused by checkpoint hash rather than repeated. The frozen EXP-609 radix-2 checkpoint is both the offline logit teacher and the simple inference control. This viability rung changes macro architecture and supervision together; it does not isolate the marginal effect of logits from intermediate hard supervision. A pass would authorize that later ablation.

## Trained-parameters boundary and supervision

The two student phases are initialized from the frozen 99,824-parameter radix-2 teacher, but their parameters are untied thereafter. Phase 1 uses the teacher's increasing dilation sweep; phase 2 uses the reverse sweep. At inference, both phases receive only legal per-case bit planes, digit bits, markers, and the learned continuous intermediate plane. No exact modular multiplication, division, reduction, quotient selection, output repair, or benchmark lookup is present in the inference path.

Exact bigint arithmetic remains offline and supplies `(s1, q1, s2, q2)`, where `s1` is the exact high-bit radix-2 successor and `s2` is the exact low-bit successor. The hard objective contains final exact-bit/worst-bit/quotient loss, half-weight intermediate loss, and a half-weight teacher-forced phase-2 loss on exact `s1`. The frozen teacher also emits step-1 and exact-conditioned step-2 logits. Binary temperature distillation is applied only where the corresponding teacher prediction is exactly correct, so teacher errors are not imitated. Student phase 2 still receives the student's continuous intermediate state on the primary autonomous path, allowing final loss to train both phases end to end.

## Frozen training and evaluation

Use exactly the EXP-630 curriculum: 1,600 batch-128 updates cycling widths 8, 12, 16, 20, 24, and 32, with seed `20260826`, the same disjoint prime-identity counts and split seeds, AdamW/cosine decay, clipping, and loss coefficients. Save exact resumable state every 50 updates, including model, optimizer, scheduler, every sampler RNG, Python/NumPy/Torch RNG, history, and cumulative wall and aggregate CPU time. Save and hash the trained checkpoint before evaluation.

On the same deterministic test identities and case streams as EXP-630, evaluate 2,000 macro transitions and 200 paired rollouts at widths 16, 32, and 64 for the trained student, frozen direct control, and frozen radix-2 control. Report student intermediate/final exactness, paired arm-only successes, validity, all-steps exactness, wall and aggregate CPU, and the student/radix-2 wall ratio. Evaluate the transferred-but-untrained student on 400 transitions and 40 rollouts per width as a trained-weight-dependence control. Repeat 20 width-64 student cases exactly for predictions, intermediate/final logits, and validity.

Advance only if every gate passes:

- at least 90%, 70%, and 20% student rollout at widths 16, 32, and 64;
- at least 98% width-32 and 80% width-64 macro-transition exactness;
- at least 50 points width-64 transition gain over both frozen EXP-630 direct and transferred-but-untrained controls;
- at most 10% invalid width-64 student rollouts;
- width-64 student wall time at most 70% of the paired radix-2 control;
- exact repeated predictions, intermediate/final logits, and validity.

Passing is architecture viability only and authorizes a separately preregistered higher-width/distillation ablation. It is not candidate promotion and does not authorize checked, independent-full, official, or private evaluation. Failure closes this exact compressed two-phase formulation without retrospective loss-weight tuning.

## Dynamic resource-budget protocol

Before formal execution, run three actual full-objective optimizer updates at each training width using training identities only. Also measure all three frozen/trained-shaped evaluation arms on 96 training-identity transitions and six training-identity rollouts at each evaluation width, including initialized controls and determinism-shaped work. Record wall and aggregate CPU separately. Project all 1,600 updates, the complete frozen evaluation, setup, MLflow, checkpointing, and serialization; add explicit safety headroom and a hard runaway ceiling.

Calibration may update only `resource_budget` measured/projection/limit fields and this section's measured-budget paragraph. It may not change sources, hashes, seed, identities, architecture, loss, widths, updates, cases, controls, gates, or evaluation streams. Formal execution is forbidden while calibration status is pending. The final limits must distinguish wall from aggregate CPU, reuse the valid EXP-630 and EXP-609 checkpoints, and preserve the trained checkpoint if evaluation overruns. No training or evaluation fidelity may be reduced to fit an arbitrary short cap.

The completed training-identity-only calibration measured three full-objective batch-128 updates at every declared width. Projecting the mean width-balanced throughput gives 1,686.304 wall seconds and 6,299.677 aggregate CPU seconds for all 1,600 updates. Small complete-arm transition/rollout profiles at widths 16, 32, and 64 project 338.480 wall and 1,267.914 aggregate CPU seconds for the full evaluation, initialized controls, and determinism repeats. With the frozen 180-second setup/tracking/serialization allowance, expected total is 2,205 wall seconds and 133 aggregate CPU minutes. The soft budget is 3,000 wall seconds; the hard runaway ceiling is 4,440 wall seconds, with a distinct four-thread upper bound of 296 aggregate CPU minutes.

External calibration timing was 49.31 wall seconds, 153.30 user CPU seconds, and 3.45 system CPU seconds at 317% CPU, with 765,308 KiB maximum RSS. Calibration JSON and console hashes are `25cc3f78d9d8fa43effb3f6f014bc75202c8293ac8c6fd990232a89546ff148c` and `235ab9a21c5fdda1efd5b35672fa6c6f7a417a08e58fff75d5de79399afe95e9`. No held-out identity was read and no scientific field changed.

## Limitations

The two untied phases double parameter storage relative to the radix-2 control even though compute is based on one dilation cycle per phase. Continuous intermediate state is intentionally different from two hard radix-2 transitions and may behave differently under width extrapolation. The test isolates the complete structured-distillation formulation, not each auxiliary loss. The deterministic test streams match EXP-630 for causal comparability and therefore are selection evidence, not new independent confirmation.
