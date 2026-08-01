# T003 — Current SAIR repository and bottleneck audit

## Scope and evidence status

This was a read-only audit of the repository after the archived session ended. No
tests or experiments were rerun because the named trace already contains recent
focused, full-suite, and validator results, while the immediate question is what
scientific state actually exists now. File contents, Git state, registered
metrics, and process state below are observed. Performance or accuracy effects
of proposed changes are hypotheses until measured.

## Post-trace repository truth

- The branch is `agent/submission-ready-horner-tcn9`; `HEAD` and its upstream are
  both `91113dc` (`exp648: authorize root162 readiness calibration`).
- The worktree is materially dirty before this GoalBuddy board: six tracked
  files are modified and nine EXP-648 readiness-calibration files are untracked.
  The modified files are `experiments/registry.yaml`, the EXP-648 ablation,
  autoresearch backlog/decision-log/state, and the EXP-648 scientific test.
- No EXP-648, pytest, or MLflow process was running when inspected. The archived
  run is therefore not merely waiting on a live scientific screen.
- Root `20261162` is sealed as a completed **non-scientific resource
  calibration**. It reports 2,709.37 wall seconds, 10,192.56 aggregate CPU
  seconds, and 2.761 GB peak RSS for one warmup plus three repeats. It projects
  an actual screen at 4,137.61 wall seconds and 15,845.33 aggregate CPU seconds,
  with 5,208.39/7,812.59-second soft/hard wall limits. No accuracy metric was
  produced and no screen `judge-authorization.json` exists.
- The nine untracked root-162 screen artifacts total roughly 1.51 MB; the two
  largest JSON files are approximately 730 KB each. They are evidence, but are
  not durable under this repository's rules until reconciled and committed.
- `knowledge/autoresearch/state.yaml` correctly marks the worktree dirty, but
  its `formal_source_commit` is `5519b5...`, not current `HEAD`; the field is
  historical EXP-649 provenance and is easy to misread as the current source.

## Incumbent evidence

The incumbent remains
`candidate-005-tier6-conditional-commutative-integration`, checkpoint SHA-256
`2f6ac319...`, with 470,049 parameters and a 1,915,885-byte artifact.

Observed exact-answer metrics from the registered held-out artifacts are:

| Surface | Overall | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7–T10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Independent 1,100-case set | 0.593 | 0.39 | 1.00 | 0.99 | 0.97 | 0.97 | 0.94 | 0.98 | 0.02 each |
| Checked 1,100-case set | 0.595 | 0.39 | 1.00 | 0.99 | 0.97 | 0.97 | 0.94 | 0.98 | 0.02 each |

The five independent Tier-6 seeds score 0.98, 0.96, 0.95, 0.99, and 0.99
(487/500 combined). Only one complete independent all-tier seed exists. The
submission returns its initialized zero for moduli wider than 128 bits, so the
2% T7–T10 values are structural zero coincidences, not learned arithmetic.
Private evaluation remains unavailable and is not local completion evidence.

## Scientific bottleneck

The remaining problem is not broad arithmetic quality. It is stable autonomous
rollout at width 256 and above:

- Prior width-256 one-step transition results were reported near 99.70–99.75%,
  while autonomous rollouts were only about 22–32% exact. A per-step accuracy
  that sounds high is insufficient over 256 dependent transitions. Under a
  deliberately simplified independent-error model, 90% whole-rollout success
  requires per-step accuracy of `0.90^(1/256) = 99.9589%`; the observed gap is
  therefore of the right magnitude to explain catastrophic trajectory drift.
- EXP-647 consistency regularization regressed, EXP-654 neural workspace/RAM
  produced zero exact beyond its smallest smoke, and EXP-649 was stopped by an
  input-only runtime projection more than an order of magnitude above its gate.
  The research-super-transformer ledger contains 44 parents and 81 components,
  but no faithful completed main method. More catalog expansion has low expected
  information value until a concrete width-256 mechanism is tested.
- EXP-648's K=8 endpoint-loss hypothesis attacks exposure bias and correlated
  drift, but remains scientifically untested after a very large provenance and
  readiness-control investment.

## Measured control-plane footprint

The current EXP-648 implementation comprises 2,883 lines of calibration runner,
9,527 lines of scientific runner, 525 lines of resolved configuration, 3,754
lines of scientific tests, a 265-line ablation, and a 311-line autoresearch state
file. This does not prove those lines are wrong; it does show that the control
surface is much larger than the hypothesis (`endpoint_loss_weight` 0.0 versus
0.1) and aligns with the trace's low late-stage information throughput.

The intended gate stack is sound in principle: focused EXP-specific tests,
project validator, then the full pytest suite. In the archived trace, the latest
reported full suite passed 282 tests in about 204 seconds and the focused
EXP-648 suite passed 162 tests in about 79 seconds. Re-running all of these after
every evidence-only edit is unnecessary; gate selection should be based on the
changed dependency cone, with full gates reserved for a package boundary.

## Inference-route audit

`submission/model.py` exposes a separate systems bottleneck for 64- and 128-bit
specialists:

1. `_bigint_features` extracts every bit from Python integers with nested Python
   comprehensions on every Horner step.
2. It reconstructs the multiplicand, modulus, LSB, and MSB planes even though
   they are constant across the entire rollout.
3. `_decode_bigint_logits` transfers the thresholded tensor to nested Python
   lists and reconstructs Python big integers after every step, only for the next
   step to expand them back into bits.
4. `_SharedDilatedConvGRU32.forward` also rebuilds the dilation schedule, though
   this is small beside convolution and representation traffic.

A tensor-native rollout can keep the predicted state as a thresholded bit plane,
precompute the static planes once, update only the multiplier-bit plane, and
decode to Python integers once at the output boundary. This changes neither
weights nor learned arithmetic and can be tested for bit-identical logits,
predictions, margins, and malformed-state handling. Its speedup is not yet
measured. It is nevertheless a high-confidence, low-scientific-risk enabler for
testing width 256 under the sandbox budget.

## Candidate optimization surfaces

1. **Route/data-layout vertical slice.** Prototype tensor-native cached feature
   planes outside the incumbent submission first; prove bit identity and measure
   wall/CPU/RSS at widths 64, 128, and a width-256 research checkpoint. Promote
   into `submission/model.py` only after the candidate is re-evaluated and its
   artifact/provenance are regenerated.
2. **Decision-safe experiment gates.** For an N-case exact-accuracy selection
   gate with k correct after m cases, the attainable final interval is
   `[k/N, (k+N-m)/N]`. Stop a selection-only run only when this interval can no
   longer cross its preregistered threshold; still execute complete evaluation
   when a persisted full distribution is required.
3. **Width-256 trajectory mechanism.** Compare EXP-648's endpoint objective
   against a small deduplicated portfolio of symmetry, redundant-state/error
   correction, and local quotient/carry transducer hypotheses. Each needs a
   cheap transition/short-rollout falsifier before a full screen.
4. **Evidence reconciliation.** Preserve root-162 as non-scientific calibration
   evidence, but do not let completion audits consume another long tranche.
   Whether to launch its roughly 69-minute screen or redirect that compute must
   be a single Judge decision after comparing the mathematical portfolio.

## Gate and ownership constraints for the next package

- Treat the existing six modified and nine untracked EXP-648 files as
  pre-existing work. A Worker must either explicitly own and reconcile them or
  avoid them; it must never overwrite them opportunistically.
- Do not edit `sources/` or `data/external/`.
- Do not place exact modular multiplication, long division,
  Barrett/Montgomery reduction, or CRT recombination in an inference route.
- Keep observed runtime improvements separate from exact-answer accuracy, and
  keep public/checked/independent/private evidence separately labeled.
- The first optimized package should deliver a measured scientific-throughput
  delta, not another governance-only layer.

