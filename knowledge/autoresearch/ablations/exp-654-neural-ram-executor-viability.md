---
type: Experiment Preregistration
title: EXP-654 learned neural-RAM executor viability
description: Matched width-16/32 viability test of trained addressable memory against decoded and non-addressable recurrent controls.
status: completed
tags: [autoresearch, neural-execution, neural-ram, memory, attention, width-transfer]
timestamp: "2026-07-18T06:20:00+05:30"
last_updated: "2026-07-18T08:10:54+05:30"
---

# EXP-654 learned neural-RAM executor viability

## Why this experiment is next

EXP-643 rejects unstable or destructive recurrent depth as Candidate 005's primary width-256 failure cause. Its frozen cell remains about 99.7% exact on ordinary width-256 transitions but reaches only 22–32% autonomous rollout, with zero overthinking and contractive random-direction gain proxies. A representational change is therefore better supported than residual depth scaling or spectral regularization.

EXP-654 tests the allowed mechanism synthesized in `knowledge/autoresearch/research/2026-07-18-learned-neural-execution.md`: learn persistent registers, read/write addressing, an append-only latent trace, a shared transition, and residue readout from ordinary initialization. It borrows no compiled executor, instruction set, arithmetic update table, modular program, or hand-assigned attention pattern.

## Frozen arms

All arms train from scratch on the same generated prefix trajectories, ordered prime identities, stage widths, optimizer steps, batch size, and exact prefix labels.

1. `decoded_control`: the established shared dilated ConvGRU with hidden size 58 and two dilation cycles. Training receives exact offline prefix states; autonomous evaluation hard-decodes and reinjects each predicted prefix.
2. `decoded_wider`: the identical decoded architecture with hidden size 80. Its parameter count is matched to the memory arms and tests whether capacity alone is sufficient.
3. `nonaddressable_workspace`: the EXP-620 persistent latent Neural GPU with hidden size 88 and two microsteps. It carries a spatial latent canvas across bits but has no addressable registers or trace.
4. `neural_ram`: a persistent 48-channel spatial canvas, 48-dimensional GRU controller, eight learned 32-dimensional registers, append-only 32-dimensional trace, and four-head PyTorch `nn.MultiheadAttention` read. Learned soft register allocation performs erase/add writes. Generic normalized/Fourier step features provide addresses; a conditioned learned ConvGRU updates the output canvas twice per input bit.

The addressable model has 74,865 parameters, the wider decoded control 76,244, and the non-addressable control 74,801. This close match isolates addressability better than adding state capacity. The smaller 41,242-parameter decoded arm remains the simple Candidate-005-style reference.

PyTorch's maintained `GRUCell`, `MultiheadAttention`, convolution, normalization, and optimizer implementations are used directly. A specialized external memory library would not add a more canonical primitive at this scale and would add an unmeasured CPU dependency; avoiding one here is an engineering choice, not a restriction against libraries.

## Initialization and compliance provenance

Every arm uses PyTorch ordinary Kaiming/Xavier/default initialization under the frozen seed. The runner saves and hashes an initial safetensors state before the first update and a final checkpoint afterward. Neural-RAM registers and slot addresses use Xavier initialization; none are assigned semantics or arithmetic values. Every answer-producing tensor is optimized from generated labels. A complete randomized-checkpoint control and addressable-memory ablations are mandatory even if exact accuracy is low.

Exact big integers are permitted only in offline pair/prefix generation and evaluation. Model inputs contain multiplicand/modulus bit planes, boundary planes, streamed multiplier bits, and generic position features. No expected answer, benchmark identity, exact modular reduction, hand-coded carry, or solver state enters inference.

## Training and untouched evaluation

The formal training stream is seed `20261055`; calibration uses distinct seed `20261056`. Generate 32 training and 16 held-out prime identities per width from the top four bits of each width band. Train 800 updates at width 8, 1,200 at width 12, and 2,000 at width 16, with batch size 64, AdamW, cosine decay, all-prefix BCE, worst-bit loss, and an additional final-prefix term. Each arm sees the same deterministic pair sequence.

Evaluate exact autonomous rollout on 300 held-out width-16 cases and 200 each at widths 24 and 32. Report all-prefix exactness, final exactness, invalid residues, first-error position, initialization/final hashes, learning curves, parameters, wall time, aggregate CPU, and peak RSS. Checked-public and prior independent answers remain unread.

For the neural-RAM arm, also evaluate width-16 and width-32 rollouts after: resetting registers/trace before each bit; reversing memory content while retaining its address positions; and replacing learned read/write allocation with uniform addressing. Log normalized read/write entropy, maximum write probability, register-update RMS, and trace RMS. Randomize the complete trained architecture on 100 width-16 cases. Measure four width-256 forward cases and project 100-case incremental Tier-7 wall time from direct measured throughput, not an assumed complexity law.

## Frozen decision rule

Advance only if every condition passes:

- neural RAM reaches at least 80% exact width-16 rollout and 5% exact width-32 rollout;
- its width-32 exact rollout exceeds the best of all three controls by at least five percentage points;
- the larger of its width-16 reset, addressed-content permutation, and uniform-addressing drops is at least ten points;
- mean register-update RMS is at least 0.005 and mean maximum write probability at least 0.145, above uniform allocation `1/8 = 0.125`;
- complete randomized weights reduce width-16 exact rollout by at least twenty points;
- invalid rate is at most 2% at every reported width;
- its parameter ratio to the wider decoded control stays between 0.90 and 1.10; and
- directly projected incremental 100-case Tier-7 work is at most 100 wall seconds, leaving headroom beside Candidate 005's 156-second isolated runtime.

If trained-width exactness is zero, memory is inactive, provenance is ambiguous, or direct runtime misses, stop this formulation before higher-width training. A passing single-seed viability result authorizes a separately preregistered independent training-seed confirmation; it is not candidate or completion evidence.

## Resource-budget protocol

The formal resource budget is unset until a complete calibration profiles all four arms for 10/10/20 updates at widths 8/12/16 using the real batch size and optimizer, all three evaluation widths, all memory ablations, randomized dependence, and a direct width-256 timing case. Scale each measured component to the frozen formal counts, apply 1.50 throughput headroom plus explicit setup/serialization allowance, and record wall and aggregate process CPU separately. Formal mode must refuse pending or dirty execution and preserve each completed arm by a hash-keyed cache.

The calibration's 1,800 wall / 7,200 aggregate-CPU-second limits are runaway guards for a representative 160-arm-update workload plus evaluation, not a formal short cap. Formal fidelity will not be reduced to fit an arbitrary duration.

### Pre-calibration correction

The first calibration launch reached no optimizer update or evaluation: split construction requested 96 unique primes in the width-8 band, which contains only 48. The process was terminated during input construction. The identity count is corrected to the already established EXP-620 value of 32 training plus 16 held-out primes, preserving all model, update, example, evaluation, seed, and decision-rule fidelity. The runner now checks the exact band capacity with SymPy `primepi` and fails explicitly rather than looping when a future request is infeasible. No calibration or formal result was consumed before this correction.

The corrected calibration then completed all four 40-update training profiles, all normal evaluations, all three memory ablations at both widths, and randomized dependence before the direct width-256 timing call. That call failed because the new guard attempted exact `primepi` counting at a 256-bit bound, which SymPy does not support. Preserve the completed partial at SHA-256 `e7da9c6987c1c25a61968c8238f8111cedf23e34e6201943d1ddbe9bc1f5ebab`; its four final checkpoint hashes are recorded inside it. Limit exact capacity counting to widths at most 32, where the guard is needed and tractable, and hash-reuse the completed partial so the next calibration invocation runs only the missing direct timing probe and budget calculation. This changes no model, optimizer, input, evaluation, or diagnostic computation.

### Resolved budget before formal execution

The hash-verified resume completed the missing direct timing and budget calculation from clean commit `cb7f8ea3fa5e7b0e580742aba7ce3166827332f4`. The final calibration artifact has SHA-256 `571fabcb986a47f3b2c8ed34b7acb3cfceb0e606d0e4fdf164a64f915748b7d1`; its runner SHA-256 is `8e09a99915720c3386ec6bd4567e05583d16415ba2e5ed4c94552e88aa8d4a12`. The resume itself consumed 4.8687 wall and 2.8 aggregate CPU seconds with 382,930,944 peak RSS; completed component timings remain embedded in the reused partial and final calibration artifact.

Scaling each arm's measured training and evaluation components, all memory/dependence diagnostics, and the direct runtime probe to the frozen formal work projects 6,632 wall seconds and 26,503 aggregate CPU seconds. The resolved soft allowance is 10,279 wall and 40,284 aggregate CPU seconds: 1.50 times the component projection plus 332 wall and 531 CPU seconds of setup/serialization allowance. Hard runaway ceilings are 15,419 wall and 60,426 aggregate CPU seconds, another 50% beyond soft. No update, example, case, arm, ablation, or width is reduced.

One directly measured width-256 neural-RAM case takes 0.3201 wall and 1.29 aggregate CPU seconds at calibration weights, projecting 32.01 wall seconds for 100 Tier-7 cases. This is only a throughput calibration; the formal trained checkpoint receives its own four-case direct measurement and must still pass the 100-second gate.

## Formal result

Reject the current formulation before scale-up. All four arms completed the frozen 4,000-update curriculum and every evaluation/ablation. The decoded control reaches 85.33% exact at trained width 16, while the parameter-matched wider decoded arm reaches 94.33%; both score zero exact at transfer widths 24 and 32. The non-addressable workspace and neural-RAM arms score zero exact at widths 16, 24, and 32, including zero final-prefix exactness on their last width-16 training batches.

The neural-RAM registers move, but allocation is nearly uniform: width-16 normalized write entropy is 0.9996, maximum write probability is 0.1336 versus the 0.125 uniform baseline, and register-delta RMS is 0.0173. Reset, permutation, and uniform-allocation ablations change no exact outcomes. Trained and fully randomized neural-RAM checkpoints both score zero exact on the dependence diagnostic. Parameter matching and direct runtime pass; trained-width, extrapolation, causal-memory, nonuniform-addressing, validity, and weight-dependence gates fail.

The full run consumes 5,220.24 wall and 20,716.37 aggregate CPU seconds with 1,183,764,480 peak RSS. This is below the 6,632/26,503 calibrated projection and within the 10,279/40,284 soft and 15,419/60,426 hard budgets. No fidelity was reduced. See the [result analysis](../analysis/exp-654-neural-ram-executor-viability.md), [canonical run record](../../../experiments/runs/exp-654-neural-ram-executor-viability/run.json), and [raw formal summary](../../../experiments/runs/exp-654-neural-ram-executor-viability/formal-summary.json).
