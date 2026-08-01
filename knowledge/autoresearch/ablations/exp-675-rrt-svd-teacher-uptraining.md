---
type: Experiment Preregistration
title: EXP675 RRT SVD Teacher Uptraining
description: Source-complete ConvGRU transfer of Relaxed Recursive Transformer teacher, SVD and joint-uptraining mechanics.
status: parked-calibration-runtime-headroom
tags: [autoresearch, rrt, svd, lora, recurrent]
timestamp: "2026-07-29T00:00:00+05:30"
---

# EXP675 - RRT SVD teacher and joint uptraining

## Decision

Can a source-complete RRT conversion retain at least 85% exact W128 and recover
at least 10% W256, with a five-point W256 gain over exact-zero visit, global
LoRA, and shared continuation controls, while satisfying the causal, validity,
reload, artifact, and candidate-runtime gates below?

EXP651 is terminally parked pre-science and read-only. It explicitly omitted
the untied teacher residual SVD and joint base/adaptor uptraining. Its roots and
formulation are not inputs to this experiment. EXP609 contributes only the
authenticated source checkpoint; EXP617 contributes only the observed fact
that global rank-8 recurrent LoRA gained 0.5 point.

## Authenticated source and topology freeze

- Integrated base: `c38349547f3891af5cfefe78ca413f48e793b602`.
- Paper: local `research/papers/arxiv-2410.20672.pdf`, SHA-256
  `680a597088a3642bdd1c4eef51f687556c0e10d5ebc0b9f44736241a51cfbfb3`,
  also present in `research/checksums.sha256`, `manifest.json`, and
  `verification.json`.
- Source checkpoint: EXP609 SHA-256
  `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`.
- Shared core: hidden 92, two CYCLE traversals, power-of-two dilation schedule,
  sixteen width-stable physical slots `cycle * 8 + log2(dilation)`.
- Teacher: gate and candidate raw Conv1d kernels are copied exactly into every
  physical visit before 100 legal-data updates. Non-kernel transformations stay
  copied and shared. Exact logit/quotient identity is required before training.
- Shared RRT centroid: arithmetic mean of the continued teacher kernels across
  the sixteen physical slots. Each residual is `teacher_visit - centroid`.
- Rank/placement: rank 8 for gate and candidate raw Conv1d parameter matrices,
  flattened as `[out, in*kernel]`; `torch.linalg.svd` retains the leading eight
  singular triplets, with `B=U_r Sigma_r`, `A=V_r^T`, alpha 8 and scaling 1.
- Uptraining: 200 phase-two updates jointly optimize the shared base and fixed
  visit adapters. The schedule is transition64, recovery64, recovery128,
  recovery128, with batch 128 / recovery batch 16 and the frozen EXP609
  optimizer family and loss weights.
- Roots: complete-route calibration 20261250; formal 20261251; confirmations
  20261252/20261253 only after every formal conjunct passes.

The frozen implementation identities before focused verification were config
`966bede3...982c`, model `ede1f9b3...d265`, runner `a120c1bb...a78`, and test
`f36af017...ad8e`; terminal records replace these abbreviated preregistration
values with complete committed hashes.

## Source map and SAIR adaptations

Paper p.3 Eq.2 maps CYCLE layer reuse to the two repeated dilation traversals.
Page 4 Eq.3 maps each physical depth to an independent LoRA choice. Page 5
Eq.4 maps teacher-minus-shared residuals to truncated SVD, and Eq.5 maps the
result to `shared + BA`. Page 5 explicitly requires all model parameters to be
trained after initialization. Pages 10-11 support SVD over zero initialization,
higher-rank recovery, Average initialization for relaxed models, and extended
uptraining.

The immutable pages 3, 4, 5, 10 and 11 were rendered at 2x with local PyMuPDF
because Poppler is absent in WSL, and visually inspected under
`tmp/pdfs/exp675/`. Equations, figures, captions, legends, axes, and table rows
were legible with no clipping. The adaptation is bounded to the raw learned
Conv1d recurrent kernels because the incumbent has no Transformer attention or
FFN matrices. RMS/group normalization, biases, stem and heads remain shared;
continuous batching, early exit and KD are excluded as independent mechanisms.

PEFT is not installed in the authenticated environment. More importantly, no
available wrapper can prove direct placement on raw `nn.Parameter` Conv1d
kernels, supplied residual-SVD factors, license provenance, and deterministic
CPU parity without fetching. Native maintained PyTorch 2.5.1 CPU
`torch.linalg.svd` (BSD-3-Clause) is therefore the faithful library choice.

## Arms and matching

The required arms are full SVD-teacher/joint RRT, exact-zero visit adapter,
global LoRA with rank 128 (parameter-matched to 16 rank-8 visit banks), continued
untied teacher, shared continuation, equal-norm random subspaces, and
slot-shuffled random subspaces. Full RRT is additionally evaluated with its
adapters zeroed, slots reversed, and adapter tensors randomized.

Every target sees the same legal phase stream and optimizer where meaningful.
The teacher/full/random/shuffled conversion paths use phase one to create their
teacher source and phase two to uptrain the converted target. Exact-zero,
global, shared, and untied controls train across both matched phases. Conversion
requires an extra teacher artifact and SVD operation; that unmatched offline
work is measured and disclosed, never charged as inference work.

## Gate and sequential route

Train and cache every arm before evaluation. Evaluate full RRT on the complete
3,000-transition / 300-rollout W128 denominator first. Reject immediately if it
is below 85% or has any invalid residue. Otherwise complete W64, controls,
interventions, and W256. Require zero invalids; W64 no more than one point below
shared continuation; causal shared/adaptor updates and changed predictions from
zero/permutation/randomization; SVD reconstruction advantage of at least 0.01
relative-error point over equal-norm random subspaces; exact checkpoint reload
and separation; W256 at least 10% and at least five points above exact-zero,
global, and shared; compression no more than one point below the untied teacher;
artifact below 20 GiB; guarded 300-case candidate inference at most 285 seconds.
Both confirmation roots must reproduce every conjunct before any later gate.

## Dynamic resources and safety

The one final package oracle starts from T070's observed 575.53-second run and
1,246-second expected / 2,492-second hard envelope, revised only from focused
throughput. Calibration executes the teacher, SVD, all training arms, all width
evaluations, interventions, and persistence at root 20261250, separately
recording wall, process-tree CPU, cgroup CPU and process-tree RSS. It derives
formal ETA plus expected, soft and hard guards with 1.35 throughput headroom.
No fidelity is shortened to fit a static cap.

Formal inference accepts only public features and uses learned shared base plus
fixed visit adapters. It excludes the teacher, exact answer or validity filters,
answer-dependent routing, repair, target canvases, programmed multiplication,
division, reduction or CRT, lookup, private/checked data, and external calls.

## Terminal result

The one hermetic package oracle passed validation (196 concepts, 1,100 records)
and all 628 tests in 553.90 seconds. After one bounded compatibility repair for
the historical EXP300 soft-state helper, focused tests again passed 9/9; no
second full pytest was run.

Root 20261250 completed all declared route classes in 119.775 wall, 388.350
process-tree CPU and 497.676 cgroup CPU seconds with 1,023,324,160-byte peak
process-tree RSS. It proved exact copied-teacher identity, 0.6372 mean SVD
relative reconstruction error versus 1.4133 for equal-norm random subspaces,
joint shared/adaptor movement, persistence, and a 9,796,600-byte artifact set.
Those tiny calibration denominators are not scientific accuracy evidence.

The guarded candidate projection was 1,863.361 seconds against the frozen
285-second ceiling, so the calibration stop rule parks the package and leaves
roots 20261251-20261253 unopened. The runner also instantiated the calibration
global control at rank 8 instead of the intended parameter-matched rank 128;
control-matching evidence is therefore unavailable. This defect does not change
the independent runtime rejection, but it is a binding limitation. There is no
scientific RRT/ST-M005 verdict and no candidate, benchmark, submission or
promotion change.
