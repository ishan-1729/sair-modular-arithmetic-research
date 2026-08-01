---
type: Mechanism Research
title: EXP672 LT2 gated hybrid mixer source map
description: Frozen PDF, official-code, adaptation, control, root, and resource contract for the Candidate005 LT2 transfer.
status: parked-package-verification
tags: [autoresearch, exp672, lt2, linear-attention, sparse-attention, gating]
timestamp: "2026-07-29T00:00:00+05:30"
---

# EXP672 LT2 gated hybrid mixer

## Authenticated sources

The immutable local PDF is `research/papers/arxiv-2605.20670.pdf`, SHA-256
`84f7f4de355928e5006e8aa85c958fb656752158e3f19bc5d7dc919922013d9b`. Pages 3-5 and
8-11 were rendered at 2x with PyMuPDF 1.26.3 and visually inspected. Page 3 Eq. 2-3 defines a
genuinely shared block reused for `T` loops, permits linear or sparse mixers, and gives the
zero-initialized per-channel loop residual. Page 4 Eq. 4-6 gives the recurrent linear-state and
`T*w` sparse receptive-field arguments. Pages 5 and 8 distinguish depth-level fixed hybridization
from changing the mixer by loop; the fixed depth-level hybrid is the source-backed default. Pages
9-11 motivate the post-SDPA elementwise gate, sink/activation diagnostics, and linear+sparse hybrid.

Repository-local Playwright 1.61.1 with managed Chromium 149.0.7827.55, new headless, fresh
nonpersistent contexts, GET/HEAD-only routing, and clean browser/context teardown observed the public
official repository, BSD-3-Clause license, README, file history, and implementation. The source pin is
official commit `adc2ba2f2704ff82744700679d1d0a090314df6c` (initial public implementation commit).
At that pin, bounded browser-response hashes are transformer
`cb90a87723ba751a05abd7fee1457c53a54aec964d5159fcc54e94618e4e61bb` (114,923 bytes), license
`8ef78c2d6e787b103a0548e73ba2cfe68bc469b5b0e6f5e7db649f1312f4f2a6` (1,474 bytes), and README
`c1a120b205fec970351869e9ca30987b9ae4bc93d568a2e7dd60e4fb450b7739` (12,659 bytes).
GitHub telemetry POSTs were blocked and an unrelated messaging endpoint returned 401; source pages
remained accessible. Evidence is under `tmp/playwright/exp672/` and all contexts closed.

## Official parity and bounded CPU transfer

The official `transformer.py` reuses one physical layer list for `loop_count`, supports fixed
interleaved linear/full and sparse patterns, creates per-loop per-dimension `residual_weight` as exact
zeros, and applies it on every loop. Its `GatedAttention` applies a query-dependent elementwise gate
after SDPA and before the output projection; `w_gate` is zero-initialized, hence its initial sigmoid is
0.5. Official GDN/DSA fast paths depend on FLA, xFormers, Triton/CUDA, or FlashAttention and are not a
portable CPU dependency for this challenge.

EXP672 therefore freezes the smallest CPU source-compatible transfer: a shared four-loop block whose
hybrid mixer is the sum of (1) native feature-map linear attention with recurrent prefix state and (2)
native fixed-window sparse attention. The sparse SDPA output is multiplied by the official-shaped
zero-weight sigmoid gate before projection. Every loop then uses a zero-initialized per-channel
residual gate. A separate zero-initialized per-channel output residual gate maps the learned loop state
back around the frozen Candidate005 proposal, making every gated arm byte-exact to the proposal at
initialization. This outer proposal gate is a SAIR function-preserving adaptation, not a claim that the
paper's language-model output is proposal-identical.

For normalized input `x`, the linear branch uses positive feature maps and prefix state
`S_t = sum_{i<=t} phi(k_i) v_i^T`, with output `phi(q_t)^T S_t / (phi(q_t)^T sum_{i<=t} phi(k_i)+eps)`.
The sparse branch uses learned query/key/value projections over a fixed local window and softmax only
inside that window. With shared hybrid block `H`, loop state is
`z_tilde_tau = z_(tau-1) + H(z_(tau-1), proposal, legal_inputs)` and
`z_tau = z_tilde_tau + tanh(rho_tau) * z_(tau-1)`. Final hidden state is
`proposal + tanh(gamma) * (z_T - proposal)`. `rho` and `gamma` initialize to exact zero.

Excluded source features are language tokens/embeddings, positional/RoPE behavior, autoregressive KV
caches, GPU-only kernels, DSA top-k selection kernels, stochastic mixer schedules, ensembles, and
language losses. No exact arithmetic, answer/correctness/validity signal, target canvas, repair,
lookup, identity feature, external call, or adaptive route enters inference.

## Frozen experiment

The source checkpoint is the regular single-link 400,584-byte EXP609 fixture at
`.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors`,
SHA-256 `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`.
Roots are calibration `20261238`, viability `20261239`, and confirmations `20261240/20261241`.
All trained arms use 400 updates, the same legal examples, widths, optimizer, four shared calls, and
frozen proposal bytes where feasible. Arms are full hybrid+gates, gate-only frozen-proposal loop,
mixer with fixed scalar gate, parameter/compute-matched shared convolution, frozen proposal, plus
reset and randomized interventions derived from the full checkpoint. Parameter/FLOP mismatches,
caches, wall/process-tree CPU/cgroup CPU/RSS, and variability are reported rather than hidden.

Root 20261238 measures every route and persistence/reload/intervention path and derives formal ETA,
soft/hard guards, and candidate runtime. It is calibration only. Root 20261239 opens only after every
admission check; it evaluates complete W128 first and stops when a conjunct cannot recover. W256 opens
only after the W128, validity, retention, dependence, reload, separation, and runtime conjunction can
still pass. Confirmations open only after a complete viability pass. No checked, independent,
official/private, candidate, submission, or promotion evaluation is authorized.

## Terminal calibration boundary

T068 normalized the inherited EXP671 schema records without changing their scientific evidence and
recovered a green 612-test package oracle after one bounded PyTorch inter-op-thread idempotence fix.
Root `20261238` then completed all four trained arms and seven routes. It measured 239.649 wall,
929.166 process-tree CPU, 995.840 cgroup CPU seconds, and 660,582,400-byte peak RSS. The complete
formal route projects to 16,386.738 seconds, with 32,773.476-second ETA, 40,967-second soft guard, and
65,547-second hard guard under 2.0 variability headroom.

The candidate projection is 183.084 seconds nominal and 366.169 seconds guarded. The latter exceeds
the frozen 285-second guarded ceiling, so root `20261239` and all formal W128/W256/confirmation work
remain unopened. The raw calibration payload's admission flag checks nominal runtime only and does not
override T068's binding guarded-runtime conjunct. This is a resource park; ST-M025 remains open.
