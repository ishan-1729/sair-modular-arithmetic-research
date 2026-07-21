---
type: Experiment Preregistration
title: EXP-657 coupled redundant learned state
description: Parameter-matched width-16/32 falsifier for two learned persistent residue-belief planes with learned syndrome, correction, and fusion.
status: rejected-before-L2-resource-gate
tags: [autoresearch, redundant-state, natural-errors, recurrent, tier7]
timestamp: "2026-07-21T21:00:00+00:00"
last_updated: "2026-07-22T00:30:00+00:00"
---

# EXP-657 coupled redundant learned state

## Question and causal variable

The width-256 evidence says that a transition model can be about 99.7% exact
under teacher forcing while autonomous rollouts remain only 22–32% exact. The
experiment asks whether two separately learned persistent residue-belief planes
provide complementary error paths that a learned syndrome and learned fusion
decoder can exploit. The cheapest discriminating surface is width 16/32, not a
Tier-7 training run.

Both arms instantiate exactly the same PyTorch module, parameter shapes,
initialization seed, optimizer, learning-rate schedule, generated pair order,
update-mode order, and forward-step operators. Each executes two separately
parameterized learned ConvGRU branches, a learned syndrome encoder, two learned
corrections, a learned fusion gate, and a learned decoder. The only causal
variable is persistent feedback:

- `single-state-control` feeds the learned fused belief into both branch inputs
  at the next Horner step. It therefore has one persistent state but retains the
  challenger's complete parameter and compute geometry.
- `coupled-redundant-challenger` feeds each branch's separately learned,
  syndrome-corrected belief into that branch at the next step.

The two branches never alias parameters. The control is stronger than a narrow
one-branch baseline: extra parameter capacity, fusion capacity, and per-step
compute are held constant, so a gain must come from maintaining separate learned
state across steps.

## Trained-parameters boundary

`model.py` accepts only floating tensor state and five generic context planes:
multiplicand bits, modulus bits, the current operand-bit plane, and endpoint
markers. It has no integer decoding, exact residue, parity, carry, division,
reduction, or correction routine. Every persistent plane, syndrome feature,
correction, gate, and decoded bit logit is produced by ordinary initialized,
trained parameters.

Python integer arithmetic is restricted to offline generated inputs, labels,
tests, diagnostics, and evaluation truth. Fixed bit-plane representation and the
known zero rollout boundary are legal representation structure, not an answer
procedure. No submission file is changed in this package.

## Natural-error-only recovery

Recovery updates may use only errors the current arm actually makes while
autonomously scanning a freshly generated training pair. For each pair, the
miner stops at the first naturally observed wrong fused state and retains the
two detached learned belief planes immediately before that transition, its legal
context, and the offline target. The replayed gradient update teaches that
specific transition. It never flips a bit, injects noise, samples an arbitrary
state perturbation, or synthesizes generic corruption.

The two arms see the same generated-pair order and the same declared number of
updates and selected recovery examples. First-error positions may differ because
they are model observations; candidate scan work is measured and disclosed.
Failure to find the declared natural cases at either width stops the package.

## Staged run protocol

### L0 — static and focused gates

Before any calibration, focused tests must prove:

1. inference-graph absence of modulo, floor division, shifts, bitwise arithmetic,
   integer decoding, and exact correction calls;
2. exact control/challenger parameter signatures and counts;
3. non-aliasing branch parameters and correct tensor shapes;
4. deterministic repeated outputs;
5. equality of first-step control/challenger logits followed by different
   persistent feedback on later steps;
6. non-inert channel-A and channel-B ablation paths at initialization;
7. gradient reachability for both channels, syndrome, corrections, fusion, and
   decoder;
8. natural first-error mining without random or generic corruption;
9. exact attainable-bound arithmetic and a fail-closed formal authorization.

L0 contains no scientific claim.

### Consumed failed calibration root 20261170

Root `20261170` launched once after L0 passed and is permanently
`consumed-failed`. It stopped during the first natural-error recovery update
because tensors constructed inside `torch.inference_mode()` were passed directly
to autograd. The parent observed exit code 1 after `9.60841893900033` wall /
`8.918745000000001` process user-plus-system aggregate CPU seconds with
`481116160` bytes peak RSS. It released no scientific accuracy or selection
metric and opened no formal root.

Its four artifacts are immutable: `attempt-state.json`
`0f00042606d29e9554f40eaf61abc28ebe35e39e5e5c4000ff342b4259a748dc`,
`completion.json`
`a2d3c5e598b19b4e1f366dc25fd206a25bf87e1ae955cfa44e475e6af9f8311f`,
empty `worker-stdout.txt`
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
and `worker-stderr.txt`
`72f5af7d49b1397fd98fd96e594a2feaa93c692480a14323975360a001203ca6`.
Root `20261170` is never retried. Unopened roots `20261171` and `20261172`
are superseded-never-authorized.

The only recovery change is tensor ownership. Immediately after the
inference-mode miner returns, its two state planes, context, and target are each
cloned into fresh ordinary tensors. L0 requires `Tensor.is_inference()` to be
false for all four and executes a real supervised backward pass through the
unchanged model/loss. Model geometry, generated data, labels, optimizer work,
loss, and evaluation fidelity do not change.

### L1 — root 20261173 resource-only calibration

Root `20261173` is calibration-only and may run once after recovered L0 passes. It performs
actual setup and identity-disjoint split construction; eight optimizer updates
for every arm including clean and natural-first-error work at both widths; every
planned evaluation component; channel ablations; collapse/error-overlap tensor
work; width-256 inference-shape work; checkpoint serialization, reload, hashing;
MLflow publication; and child-process exit observation.

Calibration releases resource work counts, cryptographic digests, the boolean
availability of at least the preregistered number of natural errors, wall time,
process user-plus-system aggregate CPU time, and peak RSS. It releases no loss,
correct count, error count, exact-answer accuracy, recovery fraction, ablation
score, selection metric, promotion result, or benchmark answer. Calibration
weights are explicitly non-scientific.

For each formal component, wall and CPU projections use the maximum measured
matching per-unit rate, never an average favorable rate. Formal setup is the
unassigned observed outer time plus explicit headroom. The soft budget is
observed setup plus `1.50 * projected variable work` plus at least 15 wall
seconds and 5 aggregate CPU seconds of setup/publication headroom. The hard
runaway ceiling is `1.50 * soft`; the RSS guard is `1.25 * measured peak`.
Neither ceiling permits reduced training or evaluation fidelity. There is no
arbitrary short calibration wall cap.

### L2 — reserved formal roots 20261174 and 20261175

Formal work is prohibited in this package. A later package requires a separate
committed authorization binding the calibration receipt, config, runner, model,
root, and `launch: true`. The formal matched falsifier trains both arms for 600
updates per seed with batch 32 and one natural-first-error recovery update every
third step. All declared updates run; training is not stopped on an accuracy
proxy.

For each seed and arm, evaluation uses exactly 4,096 held-out transitions and
512 autonomous rollouts at each of widths 16 and 32. Held-out recovery scans a
fresh stream until exactly 256 natural one-step errors of the matched control are
collected, subject to the declared 16,384-candidate availability ceiling. The
recovery numerator is challenger-exact cases among those exact 256 control
errors.

Width-32 causal ablations each use exactly 512 autonomous rollouts. `drop_a`
zeros channel A's learned branch contribution before syndrome, correction, and
fusion; `drop_b` does the symmetric intervention. Normal and ablated paths use
the same cases. Collapse diagnostics use exactly `4096 * 32 = 131072` channel
bit pairs; case-error Jaccard uses the union of channel-A and channel-B erroneous
cases as its denominator and requires at least 32 union cases.

Evaluation may stop early only when the exact best attainable final numerator,
assuming every remaining case is favorable and every adverse paired count is
zero, cannot reach a required gate. Checkpoints and completed denominators are
preserved on any such futility stop. This changes runtime, never the claim's
denominator or fidelity.

### L3 — confirmation and scaling

L3 is prohibited until L2 passes every gate and a separate phase Judge approves
fresh roots. L2 promotion does not modify Candidate 005, authorize the checked
benchmark, or claim Tier-7 success.

## Exact L2 promotion gates

All gates are conjunctive:

- recover at least `128/256 = 50%` held-out natural control errors;
- normal challenger minus `drop_a` exact rollout is at least `0.02` over 512
  paired width-32 cases;
- normal challenger minus `drop_b` exact rollout is at least `0.02` over the
  same denominator;
- challenger minus control width-32 exact rollout is at least `0.03` over 512
  paired cases;
- challenger minus control width-16 exact rollout is at least `-0.01` over 512
  paired cases;
- challenger minus control clean width-32 exact transition is at least
  `-0.0005` over 4,096 paired transitions;
- challenger exact rollout numerator is nonzero at both trained widths;
- invalid residues are at most 1%; deterministic replay, checkpoint reload, and
  source integrity are exact;
- a credible calibrated 100-case width-256 projection is at most 240 wall
  seconds and 240 process user-plus-system aggregate CPU seconds.

Channel collapse is a stop only when all three bit-level conditions hold over
the exact 131,072-bit denominator: mean absolute probability difference at most
0.01, binary disagreement at most 0.001, and centered cosine similarity at least
0.999. Independently, case-level channel-error Jaccard above 0.90 stops once its
union denominator is at least 32. Either inert causal ablation, zero rollout at a
trained width, transition regression beyond 0.05 percentage point, invalidity,
integrity failure, runtime failure, or an impossible-to-cross attainable bound
rejects the L2 formulation.

## Current evidence boundary

EXP-657 root `20261170` is consumed-failed non-scientific history with the exact
resources and hashes above. Recovery root `20261173` is now consumed-complete,
non-scientific, sealed, and non-retryable. It completed all fixed representative
components for both 17,832-parameter arms with exact zero parameter-count
difference, separate learned channel parameters, no cache reuse, no fidelity
reduction, natural first-error availability for both arms, and no generic or
random synthetic corruption. Post-receipt L0 passed 13/13 focused tests.

The parent-observed child-exit boundary is `19.261643565999293` monotonic wall
seconds, `39.670866` process user-plus-system aggregate CPU seconds, and
`487632896` bytes peak RSS. Maximum matching component rates project the complete
declared L2 work at `315.67735660561084` wall / `1135.1608660000004` aggregate
CPU seconds. With separate setup/publication headroom and the 1.50 throughput
multiplier, soft budgets are `483.03303216141694` / `1703.3008660000005`;
dynamic hard runaway ceilings are `724.5495482421254` /
`2554.9512990000007`; the RSS guard is `609541120` bytes. No fixed short cap or
fidelity reduction was used.

The measured width-256 calibration component processed exactly two cases / 512
bit-steps in `4.599589136999384` monotonic wall seconds and
`17.060000000000002` process user-plus-system aggregate CPU seconds. The frozen
projection formula `(measured / 2 cases) * 100 cases * 1.50` yields
`344.9691852749538` wall / `1279.5000000000002` aggregate CPU seconds. These are
projected 100-case resources, not a measured 100-case run. They are respectively
1.437 and 5.331 times the conjunctive 240/240 gate.

T014 therefore rejects this exact geometry before L2. Because runtime is a
binding conjunctive promotion gate, no possible unmeasured L2 accuracy can make
the current formulation promotable. This is a resource-gate decision only: it
does not claim any scientific accuracy, recovery, causal-ablation, channel-
collapse, or mechanism result.

Canonical calibration receipt SHA-256 is
`f8471f45acd0eb8cae508c0436b17f6d669ff123ba2a09beefee99f0ab698616`;
completion SHA-256 is
`261d65af75acf434ac7d50300cce981018965d9a4a62c0904cdf48b87a9c40dc`;
MLflow run `baea166d0eee497ba0f9bbb3ef871913` is FINISHED and resource-only.
Execution-bound config/runner/model hashes are respectively
`bd6f116096e52e0ce834661c46e2f2a92671d456c74c7ed98e2b75bb869a2d8f`,
`55581c1e267a273128a1a505f87fadfdb79b17f2cf6b814efc85d7514165ef8c`,
and `305e6ae4102bd37b88162e029d83f5c6313e7327ea2447e7f5c0c3e203f27239`.

Formal roots `20261174` and `20261175` are closed for this exact geometry while
remaining unopened and unauthorized; no authorization file exists. No
loss, accuracy, correct/error numerator, recovery rate, causal-ablation score,
selection metric, checked/independent/private answer, candidate update, or
submission change was released. EXP-648 root `20261162` remains byte-preserved,
consumed, non-scientific, and parked; EXP-656 remains durably rejected on its
20% route gate. Candidate 005 remains the 59.3% independent / 59.5% checked
incumbent.
