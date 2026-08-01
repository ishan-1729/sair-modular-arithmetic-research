---
type: experiment-preregistration
title: EXP-659 symbol-equivariant recurrent-reasoner falsifier
description: Frozen matched test of an explicit SE-RRM symbol axis for learned modular transitions.
status: terminal-parked-before-formal-verification
timestamp: 2026-07-22
---

# EXP-659: symbol-equivariant recurrence

## Source mechanism and bounded-faithful boundary

The authoritative local source is `research/papers/arxiv-2603.02193.pdf`, SHA-256
`5b7dfdc6474bb1060d7277677d7e17f8b7905b7a1fc4b2323a515ac1180ee058`.
Pages 5–6, §2.3, Equations 9–11 define an explicit `D × I × K` state with
position attention, then symbol attention, then a token-wise MLP, all with
residual RMS normalization. Proposition 2.3 on page 6 gives symbol-axis
permutation equivariance when task-type embeddings are absent. Pages 7–8
describe fixed recurrent inference and the domain-dependent decision to break
equivariance for Maze. Page 9 states the 2M-parameter and increased-compute
limitations. Appendix C, pages 16–18, reports matched hyperparameters, repeated
blocks, learning curves, and the enforced-versus-broken embedding ablation.

EXP-659 is bounded-faithful to the alternating position/symbol operator and
shared recurrent application. It is an analogue at the arithmetic boundary:
zero and one have anchored semantics, and legal prime-modulus inputs are not
closed under binary symbol complementation. The model therefore uses declared
learned input/output semantic anchors around an equivariant core. It does not
claim end-to-end task symmetry. An exhaustive small-domain closure test and a
separate exact core residual are mandatory.

The position-only arm removes the symbol axis and symbol attention. The
broken-tying arm keeps the same symbol-axis layout but inserts symbol-indexed
affine parameters inside the recurrent core. The latter must causally destroy
the swap-equivariance residual. All three arms use ordinary PyTorch 2.5.1 CPU
operators; no specialized external library is needed for the paper mechanism.

## Frozen data, training, and evaluation

- Formal seed: `20261177`; every arm uses the same fresh prime identities,
  generated transitions, curriculum, evaluation identities, denominators, and
  optimizer schedule. No EXP-654 answers or weights are reused.
- Curriculum per arm: 800 updates at width 8, 1,200 at width 12, and 2,000 at
  width 16, for exactly 4,000 updates. Batch size is 64. AdamW uses initial LR
  0.001, cosine decay to 0.00005, weight decay 0.00002, and gradient clipping
  at 2.0. No loss-based early stop or post-hoc extension is allowed.
- Dense validation is recorded every 200 updates and at every stage boundary.
  Stage checkpoints are retained after updates 800, 2,000, and 4,000.
- Exact autonomous rollout denominators per arm are 300 at W16, 200 at W24,
  and 200 at W32. Exact transition diagnostics are 300 at each width.
- Mandatory controls are identity and swap permutations of the two-symbol core,
  a broken-tying intervention, exhaustive small legal-domain task-symmetry
  applicability, 64-case deterministic replay, 100-case trained-versus-fresh-
  randomized dependence, static leakage checks, and four W256 runtime/RSS
  rollouts. Evaluation may stop only at a preregistered attainable-count bound;
  this implementation elects to run every declared denominator.

## Frozen scientific gates

Advance requires every condition: the symbol-equivariant arm reaches at least
80% W16 exact rollout and 5% W32 exact rollout; its W32 result exceeds the best
matched control by at least five percentage points; the core identity/swap
maximum residual is at most `1e-6`; broken tying produces residual at least
`1e-4`; every rollout invalid-residue rate is at most 2%; deterministic replay
is exact; trained W16 exact accuracy exceeds fresh randomized weights by at
least 20 points; parameter ratios remain within `[0.75, 1.25]`; static leakage
and source-boundary checks pass; and formal wall, aggregate CPU, and RSS remain
within the dynamically admitted limits. A miss rejects this exact formulation.
A pass selects only a successor experiment and is not candidate, checked,
independent, private, Tier-7, 85%, or goal-completion evidence.

## Resource admission

Calibration must execute setup, 100 matched training updates for every arm
across all three stages, each evaluation/control class, checkpoint persistence,
deterministic replay, randomization, both permutations, and one W256 rollout.
It measures monotonic wall, aggregate process user+system CPU, and peak RSS by
component. Formal expected cost is independently recomputed from measured
throughput and exact workload ratios. Soft and hard limits are respectively
`1.25×` and `1.75×` expected wall/CPU; the RSS ceiling is `1.25×` measured peak.
No evaluation fidelity may be reduced to obtain admission.

Calibration completed before formal training on the exact runner/model bytes
`b66f37a8…` / `7c95f160…`. The complete representative route measured 18.398 s
monotonic wall, 47.20 aggregate process CPU seconds, 66.292 cgroup CPU seconds,
and 437,198,848 bytes peak RSS. Component throughput projects 435 s wall,
1,552 process CPU seconds, and 2,111 cgroup CPU seconds for the formal route.
The resolved 1.25× soft limits are 544 / 1,940 / 2,638 seconds; the 1.75× hard
limits are 762 / 2,715 / 3,693 seconds; and the 1.25× RSS ceiling is 546,498,560
bytes. The calibration evidence SHA-256 is
`e1e17f1756e530c4ea7fbaef5465464b7d5ae242411d7b0d358d861399a04b16`.
Admission preserves the exact frozen training and evaluation workloads.

## Compliance and limitations

Python big-integer arithmetic is restricted to offline sample generation,
labels, exact evaluation, and property tests. Inference receives only bit
tensors and returns learned logits. Checked, independent, official/private, and
submission artifacts are out of scope. A one-seed advance would require a new
preregistered confirmation; a reject does not close other SE-RRM formulations.

## Terminal T046 outcome

The exact source package is preserved at commit
`6974c91fc600b6d9edc19edf80f7dab56ef4870c`. Its focused suite passed 13/13,
and resource-only calibration completed without a short wall cap or fidelity
reduction. The sole authorized rejected-verification recovery nevertheless
ended at 235 passed / 7 failed across the package-boundary focused suite: one
EXP648 copied-fixture permission conflict, three EXP649 frozen official-source
hash conflicts, and three EXP658 mutually conflicting frozen-test lifecycle
assertions. Under the T046 terminal rule, no further repair or formal execution
was authorized.

EXP659 is therefore parked before formal training at a verification boundary.
No formal checkpoint, exact scientific metric, MLflow formal run, causal result,
checked/independent/official/private result, promotion, or completion evidence
exists. This outcome does not reject the symbol-equivariant mechanism; reopening
this exact package requires fresh PM authorization and resolution of the
cross-package verification boundary.
