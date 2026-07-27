---
type: normative-research-ledger
title: Super Transformers full-PDF implementation and evidence audit
description: Page-anchored atomic mechanism ledger, implementation requirements, project evidence, and terminal gaps for ST-M001 through ST-M044.
status: binding-research-governance
timestamp: 2026-07-21
---

# Super Transformers full-PDF audit

## Normative status and counting

This ledger supersedes abstract-level summaries when defining an implementation or claiming coverage. It contains **44 stable parent records** and **81 atomic coverage components**. Parent identifiers `ST-M001`–`ST-M042` retain their historical semantic owners; corrections split compound parents into explicit components rather than renumbering or reassigning them. A parent is coverage-complete only when every applicable component has an independent faithful, bounded-faithful, analogue, supporting-only, excluded, or non-applicable verdict with the terminal criterion satisfied.

Current project truth is **0 scientifically reproduced parent methods and 0 faithfully completed main-paper methods**. Six parents—`ST-M033`, `ST-M034`, `ST-M036`, `ST-M038`, `ST-M039`, and `ST-M040`—are coverage-complete only through explicit exclusion, non-applicability, or supporting-source classification; that is policy/source coverage, not a scientific reproduction. Candidate 005 remains a cellular incumbent at 59.5% checked-public / 59.3% independent exact overall accuracy. EXP-633 is not HRM/MAKER; EXP-634 is not LoopLM/LT²; EXP-643 is diagnostic only; EXP-654 is not Universal-Transformer memory; EXP-649 reports resources only; EXP-648 reports corpus/readiness resources only; EXP-650–653 are plans. A narrow analogue negative never closes a paper-faithful component.

For every legal implementation, exact autonomous answer accuracy and causal weight/state dependence are required. Mature authoritative libraries are preferred when they materially implement the mechanism; pin version/license/source, prove tiny-fixture parity, retain a deterministic CPU submission path, and prove CUDA-to-CPU parity if CUDA is used offline. “Library: native” means ordinary PyTorch is sufficient, not permission to hand-code an available specialized kernel poorly.

## Atomic classification and current status

`Class` describes how SAIR can cover the source mechanism; `status` describes current project evidence. `Terminal-policy` and `terminal-source` close only the stated non-runnable/source component. They are never accuracy evidence. `Exact formulation rejected; family open` preserves a durable negative without over-closing the broader analogue.

| Component | Class | Current status |
|---|---|---|
| M001.a ACT | bounded-faithful | open; no faithful evidence |
| M002.a UT | bounded-faithful | open; EXP-634 adjacent only |
| M002.b DEQ | bounded-faithful | open; no solver evidence |
| M003.a constructed executor | excluded | terminal-policy |
| M003.b learned executor | analogue | open; EXP-654 adjacent only |
| M004.a LLA | bounded-faithful | open; EXP-634 adjacent only |
| M004.b length schedule | bounded-faithful | open |
| M005.a RRT | bounded-faithful | open; EXP-617 non-faithful |
| M006.a Huginn | bounded-faithful | open; EXP-634 adjacent only |
| M006.b latent-thought regularizer | bounded-faithful | open |
| M007.a H/L recurrence | bounded-faithful | open |
| M007.b gradient/deep supervision | bounded-faithful | open |
| M007.c deep-Q ACT | bounded-faithful | open |
| M008.a head/body/tail recurrence | bounded-faithful | open |
| M008.b C/D recurrence forms | bounded-faithful | open |
| M009.a sharing layout | bounded-faithful | open; cross-covered with M043.a |
| M009.b recursion-depth routing | bounded-faithful | conditional after dense viability |
| M010.a TRM | bounded-faithful | open |
| M010.b Tab-TRM | supporting-only | terminal-source application evidence |
| M011.a Stage-I allocation | bounded-faithful | open |
| M011.b Stage-II exit gate | bounded-faithful | open |
| M012.a TRM audit | supporting-only | terminal-source diagnostic |
| M012.b URM | analogue | open with generic identities only |
| M013.a LoopViT pixel path | non-applicable | terminal-policy |
| M013.b LoopViT step/exit transfer | analogue | open |
| M013.c LoopFormer | bounded-faithful | open; EXP-648 non-scientific |
| M014.a Mamba-2/TRM | bounded-faithful | open; EXP-650 planned only |
| M015.a symbol equivariance | bounded-faithful | open |
| M015.b operand commutativity | analogue | exact EXP-647 formulation rejected; broader analogue open |
| M016.a RSM | bounded-faithful | open |
| M016.b RIM | bounded-faithful | open |
| M017.a autoregressive ladder | bounded-faithful | conditional on viable autoregressive model |
| M018.a loop-count curriculum | bounded-faithful | open |
| M018.b hop/halting curriculum | bounded-faithful | open |
| M019.a DRM | bounded-faithful | EXP-663 terminal resource park before science; EXP-610/648 non-faithful |
| M019.b SPRM | bounded-faithful | open |
| M020.a UT memory tokens | bounded-faithful | open; EXP-654 non-faithful |
| M021.a Attractor Models | bounded-faithful | open |
| M021.b EqR | bounded-faithful | open |
| M021.c HRM attractor audit | supporting-only | terminal-source diagnostic |
| M022.a GRAM variational hierarchy | bounded-faithful | open |
| M022.b GRAM selection | bounded-faithful | open |
| M023.a PTRM stochastic recursion | bounded-faithful | open |
| M023.b PTRM Q selection | bounded-faithful | open |
| M024.a H2L3 hierarchy | bounded-faithful | open |
| M024.b MagicNorm/credit transfer | analogue | open |
| M024.c language-loss/gating transfer | analogue | open |
| M024.d auto-guidance transfer | analogue | open |
| M025.a LT² mixer | bounded-faithful | open |
| M025.b LT² gates | bounded-faithful | open |
| M026.a JSR regularization | bounded-faithful | open; EXP-643 diagnostic only |
| M026.b random loop counts | bounded-faithful | open |
| M027.a FPRM parameterization | bounded-faithful | open |
| M027.b convergence halting | bounded-faithful | open |
| M028.a LOTUS latent layout | bounded-faithful | open |
| M028.b LOTUS dual supervision | bounded-faithful | open |
| M029.a rule-verifier ERM | excluded | terminal-policy |
| M029.b learned energy | analogue | open |
| M030.a LayerNorm analysis | supporting-only | terminal-source; diagnostics recur per architecture |
| M030.b DeepLoop | bounded-faithful | open |
| M031.a T²MLR | analogue | conditional on viable autoregressive model |
| M032.a learned fast indexed execution | analogue | open; Percepta web source, EXP-654 narrow negative |
| M033.a compiled deterministic executor | excluded | terminal-policy; Percepta web source |
| M034.a Neural Computers video prototype | non-applicable | terminal-policy |
| M034.b Neural Computers roadmap | supporting-only | terminal-source roadmap |
| M035.a LimiX transfer | analogue | conditional low-priority |
| M036.a RLM | excluded | terminal-policy |
| M037.a local call/return model | bounded-faithful | open |
| M037.b external SRLM | excluded | terminal-policy |
| M037.c SRLM selector transfer | analogue | open |
| M038.a lambda-RLM | excluded | terminal-policy |
| M039.a recursive agent harness | excluded | terminal-policy |
| M040.a The Duck | excluded | terminal-policy; no paper PDF |
| M041.a MAKER agents | excluded | terminal-policy |
| M041.b stepwise voting transfer | analogue | open |
| M042.a learned neural RAM/reusable state | analogue | exact EXP-654 formulation rejected; family open |
| M043.a layer-local schedule | bounded-faithful | open; same sharing layout as M009.a |
| M043.b capacity recycling | bounded-faithful | open |
| M043.c ordinary-MoE recurrence | bounded-faithful | conditional after dense viability |
| M044.a target-only SPT | analogue | conditional on compatible backbone |
| M044.b GSPO/DAPO transfer | analogue | conditional on compatible stochastic autoregressive model |

## Foundations and recurrent depth

### ST-M001 — Adaptive Computation Time (1 component)

**PDF:** `1603.08983`, pp. 3–4 §2.1 Eq. 5–9 and p. 5 Eq. 10–13. **Atomic mechanism:** sigmoid halting mass, `1-epsilon` stopping, remainder-weighted state/output, maximum steps, and ponder cost. **Library:** native; compare any mature ACT utility before custom code. **Evidence/verdict:** no faithful SAIR test; open, bounded-faithful. **Required controls/terminal gap:** sweep tau/epsilon/max steps/halt bias and ponder weight against fixed/equal-average depth; intervene on halt scores. Terminal only after exact-answer, compute, and causal halting verdicts.

### ST-M002 — Universal Transformer and DEQ (2 components)

- **M002.a UT:** `1807.03819`, pp. 2–3 §2.1 Eq. 5; shared attention with position/time embeddings and per-position ACT. Native Transformer implementation; no faithful evidence. Test shared versus untied depth, time/position ablations, ACT versus fixed depth.
- **M002.b DEQ:** `1909.01377`, pp. 4–5 §3.1.1–3 and p. 15 App. E; actual root solve plus implicit backward. Prefer a maintained DEQ/root-solver/autograd implementation with tolerance parity. No faithful evidence. Terminal requires convergence, implicit-gradient parity, exact-answer transfer, and solver-failure behavior.

### ST-M003 — Looped Transformer programmable-computer boundary (2 components)

- **M003.a literal executor:** `2301.13196`, p. 1, p. 3, p. 15 Lemma 4, pp. 7–19; hard-coded weights, binary pointers/program counter, program/scratchpad tokens, read/write/branch, SUBLEQ/FLEQ and calculator programs. Literal use is excluded by the trained-parameters rule.
- **M003.b learned executor analogue:** ordinary-initialized, label-trained addressable state, command partitions, program counter, read/write and learned branching. Native attention is adequate. EXP-654 soft RAM is adjacent and does not close it. Terminal requires provenance audit, pointer/state interventions, weight randomization, and width transfer without encoded arithmetic instructions.

### ST-M004 — Looped learning and length generalization (2 components)

- **M004.a LLA:** `2311.12424`, p. 5 §4.1/Fig. 3 and pp. 5–6; input reinjection plus moving endpoint/window loss. Native. EXP-634 is a negative adjacent loop, not faithful.
- **M004.b length schedule:** `2409.15647`, pp. 5–7 §4.1–4.3 Eq. 2/Table 1, p. 9 §6.3, p. 10 §7; NoPE, first-order prediction, known step schedules, endpoint lengths, and per-sample confidence. No invented visit-embedding shuffle. Open. Terminal after each component is isolated against fixed/equal-work recurrence across unseen widths and visits.

### ST-M005 — Relaxed Recursive Transformers (1 component)

**PDF:** `2410.20672`, p. 3 §2.2, pp. 4–5 §2.3 Eq. 3–5, pp. 10–11. **Mechanism:** CYCLE shared base plus physical-layer-by-visit LoRA, SVD residual teacher initialization, then uptraining base and adapters. **Library:** PEFT is preferred if its exact adapter placement/parity is demonstrated; otherwise small native adapters. EXP-617 global LoRA is not RRT. **Verdict:** open. Terminal after visit-specific adapter use and matched base/shared/untied controls; do not assume frozen base or global merge.

### ST-M006 — Recurrent-depth latent reasoning (2 components)

- **M006.a Huginn:** `2502.05171`, pp. 3–4 §3.1–3.3 Eq. 1–2 and pp. 10–11; prelude/core/coda, random latent, input reinjection, concatenation, sandwich RMSNorm, heavy-tailed loop sampling, TBPTT.
- **M006.b latent-thought regularization:** `2502.17416`, pp. 2–4 and p. 10 §4 Eq. 3–4; vanilla shared `k x L` loop plus cosine regularizer toward untied layers.

Native Transformer blocks suffice. EXP-634/EXP-643 are adjacent. Both remain open; terminal requires trained and extrapolated depth curves, component ablations, latent reset/randomization, and exact autonomous transfer.

### ST-M007 — HRM (3 components)

**PDF:** `2506.21734`, p. 4 §2.2 and pp. 6–9 §2.3–2.5/Fig. 5. **Components:** (a) interdependent unequal-frequency H/L recurrence with carried, not periodically reset, state; (b) detached deep supervision and one-step gradient; (c) deep-Q ACT. Native. EXP-620/633 are not HRM. Open. Required controls include equal-frequency, state reset/detach/carry, full versus one-step gradients, no deep supervision, and ACT versus fixed depth. Parent closes only after all three receive independent exact-answer and causal verdicts.

### ST-M008 — AbbIE (2 components)

**PDF:** `2507.08567`, p. 3 §2.1/Fig. 2 and pp. 3–4 §2.2 Eq. 3–5/Fig. 3. **Components:** (a) repeated layer stacks with head/body/tail—not input chunks; (b) `C B(h)` versus residual `D B(h)+h`, trained at recurrence 2 and tested at 1/2/4/8. Native. No evidence. Terminal requires recurrence extrapolation, head/body/tail and residual controls, and weight/state interventions.

### ST-M009 — Mixture-of-Recursions (2 components)

**PDF:** `2507.10524`, p. 3 §2.1/Table 1 and pp. 4–5 §2.2.1 Eq. 2.1–2.2/Fig. 2/Table 2. **Components:** (a) Cycle/Sequence/Middle sharing layouts; (b) expert-choice hierarchical top-k versus token-choice one-shot depth routing, including load, KV-cache, and CPU effects. Loopies' `B1,B1,B2,B2,...` schedule is MoR's Sequence-sharing layout, so M009.a and M043.a share one schedule-evidence slot; Loopies does not implement M009.b. Prefer official/maintained router kernels only after a dense recurrent winner; CPU parity is mandatory. Terminal requires route-causal exactness and savings versus fixed/equal-average/random routing; ordinary MoE expert routing alone is not the full MoR system.

### ST-M010 — TRM family (2 components)

- **M010.a TRM:** `2510.04871`, pp. 5–7 Fig. 3/§4, p. 5 Table 1, p. 8 Table 3, p. 11; `n` latent `z` updates then answer `y`, first `T-1` blocks forward-only, final backprop, detached deep supervision, Q halt, EMA.
- **M010.b Tab-TRM application:** `2601.07675`, pp. 6–8 §2.2/2.4; distinct full-BPTT FNN without deep supervision or ACT. It is application evidence, not the same training contract.

Native recurrent modules suffice. Persistent-latent experiments are adjacent only. EXP-652 now supplies a bounded-faithful M010.a implementation and viability result: answer/latent state, recursion/Q/randomized interventions, raw/EMA reload, and matched controls complete, but all arms score 0% exact at W16/W24/W32 and the formulation is rejected before gated component confirmations. Deep-supervision/Q/EMA causal attribution therefore remains unresolved. Tab-TRM remains separately classified and unreproduced.

### ST-M011 — LoopLM (2 components)

**PDF:** `2510.25741`, pp. 4–8 §3.1–3.2 Eq. 1–6, pp. 9–11 and 16–19. **Components:** (a) Stage I expected exit loss plus entropy/KL-to-uniform using survival/remainder probabilities; (b) Stage II freezes the LM and trains the gate with detached marginal loss and CDF-threshold exit. Native. EXP-634/643 are not LoopLM. Open. Terminal requires both training stages, fixed/random/equal-work exits, gate interventions, and exact-answer evidence; extra loops may degrade.

### ST-M012 — Identity audit and URM (2 components)

- **M012.a TRM audit:** `2512.11847`, p. 4 §3.1 and pp. 5–7 tables; checkpoint diagnostic and puzzle IDs, not generic positional features. Supporting-only; never encode evaluation identity.
- **M012.b URM:** `2512.14693`, pp. 3–5 §3.1–4.1, p. 5 Table 1, pp. 6–10; UT plus ConvSwiGLU/depthwise-k2, TBPTL with first 2/8 steps forward-only, outer ACT, EMA, and puzzle embedding. Native. Open analogue only with generic role/position identities and leakage tests.

### ST-M013 — LoopViT and LoopFormer (3 components)

- **M013.a LoopViT pixel path:** `2602.02156`, pp. 3–5 §3.1–3.4 Eq. 2/11; hybrid attention plus image ConvGLU. Pixel/image replication is non-applicable.
- **M013.b LoopViT transferable controls:** step embeddings, entropy exit, and task-specific test-time training from the same source are analogue-only and remain open.
- **M013.c LoopFormer:** `2602.11451`, pp. 4–6 §3.1–3.3/Alg. 1 and pp. 8–10; normalized time/step RMSNorm/residual gates, full/short losses, stop-gradient hidden consistency, user budget—no learned exit.

Native. EXP-648 endpoint consistency is not either method. EXP-661 now supplies a bounded-faithful M013.c viability result with time/step, shortcut, full/short-loss, hidden-consistency, fixed-budget, randomized, reload, curve, and matched-control evidence. All seven arms score 0% exact at W16/W24/W32 for budgets 1/2/4/8/16, so the exact small endpoint formulation is rejected and confirmation roots remain unopened. The source paper, LoopViT controls, and independent component-confirmation verdicts remain unresolved.

### ST-M014 — Mamba-2/TRM hybrid (1 component)

**PDF:** `2602.12078`, pp. 3–4 §3.1–3.3, p. 5 Tables 1–2, pp. 5–6 Table 3, p. 9 §7. **Mechanism:** Mamba2→Mamba2→Attention→MLP or MLP-t, postnorm RMSNorm, with reverse/bidirectional controls; pure causal Mamba is excluded from the faithful arm. **Library:** authoritative pinned Mamba-2 library required, with tiny-fixture, CPU, and device parity. EXP-650 proved maintained Transformers `5.14.1` CPU forward/backward/reload feasibility and implemented the bounded-faithful pipeline plus matched controls, but T053 parked before root `20261186` because the repaired clean-worktree full oracle remained red on 34 inherited EXP648/649/658 ignored-evidence failures. This is a verification-boundary park, not a scientific reproduction or rejection; exact transfer and wall/CPU/RSS remain open only under fresh authorization.

## Recursive reasoning offshoots

### ST-M015 — SE-RRM and operand-symmetry analogue (2 components)

- **M015.a faithful symbol equivariance:** `2603.02193`, pp. 5–6 §2.3 Eq. 9–11/Prop. 2.3; explicit symbol/color axis with alternating position/symbol attention.
- **M015.b operand commutativity:** a legal SAIR analogue only, not the paper's mechanism. EXP-647 is a completed/rejected full-fidelity matched endpoint-consistency formulation; it closes that frozen formulation but not the broader operand analogue and never closes SE-RRM.

Native equivariant tensor operations suffice. Terminal requires exact equivariance residuals, symbol-permutation interventions, broken-tying controls, leakage audit, and separate analogue wording.

### ST-M016 — RSM and RIM (2 components)

- **M016.a RSM:** `2603.15641`, pp. 5–7 §II.B–D; two states, one shared operator, terminal loss, mostly one-step gradient, separate H/L curriculum, transition stability.
- **M016.b RIM:** `2603.05234`, pp. 3–5 §3–4.1, p. 7 Table 3, p. 8 Table 4; Solver/Generator/Reweighter; RIMA gates candidate/prior and RIMFormer attends state history.

Native. EXP-633 is neither. Terminal requires each decomposition, training contract, component randomization/reset, and matched TRM/cellular controls.

### ST-M017 — Autoregressive TRM (1 component)

**PDF:** `2603.08082`, pp. 4–6 §§3–5, p. 5 Table 1, p. 8 §5.3/6. **Mechanism:** full ladder of Dense/tied/step/UT-ACT/Dual-UT/Dual-Nested/AR variants under identical stream, mask, KV, and block-pass conditions, with reinitialized latent. Native. Conditional on a viable autoregressive model; no evidence. Terminal is a matched exact-answer/error-compounding verdict or documented non-applicability after recursive AR viability closes.

### ST-M018 — Loop-count and hop curriculum (2 components)

**PDF:** `2604.07822`, p. 2 Fig. 1, pp. 4–5 §4, pp. 6–10. **Components:** (a) zero-initialized attention/FFN outputs, NoPE, per-batch clipped-Poisson loop counts; (b) separate hop curriculum with KL+entropy halting, trained for roughly 1.3M updates in the source. Native. EXP-643 is diagnostic only. Terminal requires loop and hop generalization surfaces plus zero-init/curriculum/halt ablations; local budget is measured, not copied.

### ST-M019 — DRM versus SPRM (2 components)

- **M019.a DRM:** `2604.18839`, pp. 6–8 §3.2/4.2–4.3 Eq. 6–9 and pp. 8–10; cosine corruption of target output tokens, answer-canvas unrolling, clean CE, iterative remasking.
- **M019.b SPRM:** Gaussian corruption only at TBPTT boundaries in latent state. It is not DRM target-token denoising.

Native. EXP-610 and natural-endpoint EXP-648 are legal analogues, not faithful DRM. Offline target corruption is permitted for training; inference must operate only on model state without labels, teacher answers, or repair code. Terminal records DRM/SPRM separately with corruption/remasking, boundary-noise, and submission-artifact controls.

EXP-663 implements the bounded-faithful M019.a target-token path with cosine corruption and iterative uniform remasking, but its sole parallel calibration repair crossed the frozen wall guard before a complete equivalent receipt. Viability and confirmation roots remain unopened, so M019.a has a terminal resource/expected-value park rather than a scientific verdict.

EXP-664 implements bounded-faithful M019.b state-scaled Gaussian corruption only at detached TBPTT hidden-state boundaries, with an exact noise-free source call graph and matched clean/wrong-boundary/unscaled controls. Full calibration admitted root `20261207`; the full-denominator W128-first stage then reached only 43.67% exact rollout (99.23% one-step exact, 0.67% rollout invalid) and triggered its frozen futility gate. This closes the exact bounded SAIR adaptation negatively while leaving later widths, component comparisons, confirmations, the general SPRM claim, and the paper's language-model result unresolved.

### ST-M020 — Universal-Transformer memory tokens (1 component)

**PDF:** `2604.21999`, pp. 2–4 §2.1–3, pp. 4–6 Tables 2–5, p. 8 Tables 7–8, p. 9 Fig. 7. **Mechanism:** pre-norm UT with prepended sample-independent learned memory tokens, type/step embeddings, RoPE, QK/DerfNorm, ACT, and deep-start halt bias. Native. EXP-654's soft RAM/workspace is not faithful. EXP-662 implemented the bounded transition adaptation and produced a diagnostic all-zero 0/4/8/16/32/64 sweep, but its final bytes lack a green full oracle (429 passed / 7 errors, then targeted 18/18), so it is parked and does not close this record scientifically. No same-formulation recovery is authorized.

### ST-M021 — Attractor, EqR, and HRM diagnostic (3 components)

- **M021.a Attractor Models:** `2605.12466`, pp. 4–6 §3.1–3.2 Eq. 1–3, p. 9 §4.2, pp. 11–12; proposal plus attractor, Anderson acceleration, implicit/phantom gradients. Prefer maintained root/Anderson utilities.
- **M021.b Equilibrium Reasoners:** `2605.21488`, pp. 3–9 §§3–5/Alg. 1/Eq. 2; finite TRM, random initialization, noise/damping, depth×breadth, residual selection. Residual selection is unsafe until calibrated against exactness.
- **M021.c HRM attractor audit:** `2601.10679`, pp. 3–8; supporting diagnostic of spurious/incorrect attractors.

EXP-643/649 are not these methods. Terminal requires convergence-signature calibration, state perturbations, solver/gradient/noise/damping controls, and exact-answer selection evidence.

### ST-M022 — GRAM (2 components)

**PDF:** `2605.19376`, pp. 3–5, 9–10, 15. **Components:** (a) high/low hierarchy with target-conditioned posterior versus inference prior, Gaussian mean/variance only on high state, KL/variational training, deep supervision/truncated gradients; (b) LPRM versus majority selection. Native distributions suffice. Required controls: stochastic-guidance, stochasticity-only, guidance-only, direct prediction, and random-initialization noise. EXP-649 is resource-only and unrelated. Open.

### ST-M023 — PTRM (2 components)

**PDF:** `2605.19943`, pp. 2, 4–5, 8, 13. **Components:** (a) Gaussian noise at every deep-recursion/supervision step of a pretrained TRM; (b) K rollouts selected by its existing correctness Q head. Native. Test sigma/K curves, deterministic/sigma-zero, Q/mode/majority/pass@K and selector interventions. EXP-649 is not PTRM. Resource budgets are architecture-specific and dynamic; oracle breadth cannot promote.

### ST-M024 — HRM-Text transfer (4 components)

**PDF:** `2605.20613`, pp. 3–8 and 23–26. **Components:** (a) H2L3 hierarchy; (b) MagicNorm plus warmup deep-credit assignment; (c) response-only completion loss, PrefixLM, and sigmoid attention gating; (d) zero-cost auto-guidance. Native. Large language pretraining is non-applicable, but transferable components remain open and cannot be closed by generic HRM. Test each without importing evaluation labels.

### ST-M025 — LT² (2 components)

**PDF:** `2605.20670`, pp. 3–5 and 8–11. **Components:** (a) genuinely shared loop with linear/sparse and hybrid mixers; (b) zero-initialized per-channel loop residual gates and SDPA output gate mitigating attention sinks. Prefer official `chili-lab/LT2` code after license/version/parity review; implementation unavailability is not a terminal path. No evidence. Terminal after component ablations and exact scaling/runtime curves.

### ST-M026 — STARS (2 components)

**PDF:** `2605.26733`, pp. 4–8. **Components:** (a) Jacobian Spectral Radius Regularization using a one-step power-iteration/JVP squared-norm proxy—not exact spectral radius; (b) random loop-count sampling. Use native JVP/autograd. EXP-643 does not test STARS. Terminal requires each component's causal effect, stability diagnostics, exact transfer, and compute cost.

### ST-M027 — Fixed-Point Reasoners (2 components)

**PDF:** `2606.18206`, pp. 5–14, 25, 27. **Components:** (a) PreNorm, learned layerwise residual scaling, iterationwise input reinjection, TBPTT and deep supervision; (b) damped convergence with patience/geometric decay and normalized infinity-norm residual halting. Native; use vetted convergence utilities if needed. Can be tested de novo, not only after a stable winner. Terminal requires residual-to-exactness calibration and component interventions.

### ST-M028 — LOTUS (2 components)

**PDF:** `2606.31779`, pp. 3–13 and 25. **Components:** (a) K padded blocks of c latent positions, shared loop with input injection/prefix KV cache; (b) direct per-step latent supervision through the same LM head plus separate final-answer loss. Native. Exact traces are training-only and forbidden at inference. Test no-latent, padding/no-loop, direct/auxiliary decoder, both-loss ablations, and K/c/R sweeps. No evidence.

### ST-M029 — Energy-guided recursive model boundary (2 components)

- **M029.a literal ERM:** `2607.10128`, pp. 2–6 §3.4/Table 1; hand-constructed Hopfield/task memories and rule verifiers—Sudoku permutations, PPBench rules, and Maze BFS shortest path. A modular-arithmetic literal verifier would likely be prohibited; excluded.
- **M029.b learned-energy analogue:** energy learned only from legal training labels, optionally with parallel tempering. Use a maintained sampler if selected, with CPU/determinism parity. Open analogue; test energy calibration, random/disabled energy, selection regret, exactness, and dynamically measured resources. EXP-649 supplies no scientific evidence.

### ST-M030 — LayerNorm dynamics and DeepLoop (2 components)

- **M030.a LayerNorm dynamics:** `2607.10681`, pp. 2 and 4–18; constrained carry, PreNorm gain control, nonnormal full-Jacobian stability, and nonlinear-block memory.
- **M030.b DeepLoop:** `2607.13491`, pp. 4–13 and 23; PostNorm scaling `alpha=sqrt(2N)`, initialization `beta=(8N)^(-1/2)`, visit-alignment exponent near `1/2`, and separate gradient-visible/H/L depths. It is proactive parameterization, not merely repair after instability.

Native. EXP-643 does not close either. Terminal requires full-Jacobian/activation diagnostics and exact matched training, with components separated.

### ST-M031 — T²MLR (1 component)

**PDF:** `2607.15178`, pp. 3–11. **Mechanism:** autoregressive KV-cache recurrence from `l_end` at token `t` into before `l_start` at `t+1`, exact zero-initialized gated fusion, recurrent cache update, temporal-parallel Jacobi training, sequential inference. Native. Candidate 005's bit scan is not T²MLR. Conditional analogue until an autoregressive SAIR decoder is justified; terminal requires cache/fusion causality and train/inference parity.

## Learned computers, specialists, and external systems

### ST-M032 — Learned fast indexed execution (1 component)

The user-elevated source is Percepta's web article [Can LLMs Be Computers?](https://www.percepta.ai/blog/can-llms-be-computers), not a research-paper PDF. Its transferable claim is fast low-dimensional/indexed attention over execution state. Status is analogue-only: any SAIR implementation must use ordinary initialization and learn pointers, addressing, control state, and read/write behavior from legal training labels. Native attention or a mature differentiable-memory library may be used after parity review. EXP-654 closes only its exact soft-RAM formulation, not this broader learned indexed-execution record. Terminal requires learned addressing/control, provenance, randomized/causal ablations, exact transfer, and no programmed initialization or arithmetic circuit. Giannou et al. `2301.13196` remains the related formal constructed-computer source under M003, not a substitute full-PDF citation for Percepta.

### ST-M033 — Compiled deterministic Transformer executor (1 component)

Percepta's web article also claims a compiled C-to-token executor with constructed execution semantics. No formal Percepta paper PDF is in the corpus, so this is a Playwright-verified web-source boundary, not a full-PDF reproduction. Literal compiled pointers, program counter, program/scratchpad payloads, read/write operations, or calculator instructions are excluded by the trained-parameters rule. The related formal construction in `2301.13196` (pp. 1–4, 8–27) supports the compliance analysis but does not change source identity. This parent is terminal-policy excluded; it does not close M003.b or M032.

### ST-M034 — Neural Computers (2 components)

- **M034.a demonstrated system:** `2604.06425`, pp. 2–4, 7–10, 22–25; a Wan2.1 open-loop video-diffusion prototype over interface frames. The paper reports correct answers in roughly half of arithmetic captions in the training data and cautions that reprompting may provide hints or outsource work to an external text system; it does not establish that these occurred as measured reprompt results. Pixel/video replication is non-applicable.
- **M034.b roadmap:** unbounded effective memory plus install/reuse and run/update governance are future proposals, not demonstrated RAM or routine-reuse mechanisms. They remain supporting-only inspiration with no project result. Direct Universal-Transformer memory belongs to M020; the legal learned-RAM analogue retains its historical owner M042.

### ST-M035 — LimiX (1 component)

**PDF:** `2509.03505`, pp. 4–8. Cell embeddings, discriminative feature encoding, alternating feature/sample attention, CCMM episodic masks with mask-density conditioning, and synthetic SCM-style pretraining define the method. Prefer official implementation if adopted. Low-priority analogue only; evaluation labels may never appear as context rows. Terminal requires matched synthetic pretraining and unseen-prime/width exactness, not a generic tabular probe.

### ST-M036 — RLM (1 component)

**PDF:** `2512.24601`, pp. 2–9. Symbolic prompt handle, persistent REPL, generated program subcalls, metadata-only feedback, and final-variable output are external execution and excluded from submission inference. Depth can hurt and costs have outliers. Internal learned state externalization is only an analogy and needs a separate record/plan.

### ST-M037 — Local recursion versus SRLM (3 components)

- **M037.a local call/return model:** `2603.02112`, pp. 3–4 §2; one local sequence generator, reserved call/return tokens, pure stack, active-frame isolation, and child-answer return. Potentially legal bounded-faithful internal model using offline traces; no REPL/tools/Python. Native. Open pending runtime/artifact review.
- **M037.b external SRLM:** `2603.15653`, pp. 4–5 §2.2; an external REPL generates K programs. The literal branch is excluded.
- **M037.c SRLM selector transfer:** plurality answer followed by verbalized-confidence×trace-length selection within that plurality is an internal analogue only. It remains open and must be tested without REPL/code execution or answer labels. Terminal accounting must preserve all three classifications.

### ST-M038 — lambda-RLM (1 component)

**PDF:** `2603.20105`, pp. 5–8. Fixed task-specific combinators, symbolic split/filter/reduce/composition, typed fixed-point executor, and bounded neural leaves are literal external symbolic execution and excluded. Failure-closed control and resource accounting are infrastructure lessons only, not model evidence.

### ST-M039 — Recursive Agent Harnesses (1 component)

**PDF:** `2606.13643`, pp. 3–5. Parent-generated Python/async spawning, full tool-using subagents, isolated contexts, shared output files, JSON calls for 1–5 tasks, default depth 3: external agent execution, excluded. The source lacks adequate depth/path/entry and exact token/wall ablations. No submission analogue is implied.

### ST-M040 — The Duck (1 component)

No full research PDF exists in the supplied corpus; evidence is an ARC Prize blog, Kaggle material, and video. Interactive external execution is excluded. Do not make a “full-PDF verified” claim or promote from web-reported ARC performance.

### ST-M041 — MAKER (2 components)

- **M041.a main system:** `2511.09030`, pp. 4–12; maximal decomposition (`m=1`), first-to-ahead-by-k sequential voting, and red-flag resampling of long/malformed outputs. Literal API multiagent inference is excluded.
- **M041.b multiplication analogue:** Appendix F, pp. 28–29; 5x5/6x6 multiplication with fixed depth 5 and recursive voting at decomposition, atomic solution, and composition stages. A legal learned analogue must preserve per-step error correction, not vote only over final trajectories.

Native internal ensembles suffice. Test no decomposition/voting, atomic-only versus decomposition/composition voting, first-to-k versus first-to-ahead, red-flag ablation, k curves, correlated errors and collisions. No faithful evidence.

### ST-M042 — Learned neural RAM and reusable runtime state (1 component)

This stable record retains the legal analogue historically tested by EXP-654: ordinary-initialized persistent registers or trace state, learned addressing/read/write, controlled updates, and causal reuse, informed by learned-execution sources and the Neural Computers roadmap without treating that roadmap as demonstrated RAM. EXP-654 completed and rejected its exact eight-slot soft-addressing and non-addressable-workspace formulation at zero trained-width exactness, nearly uniform writes, and zero causal-memory effect. That durable negative closes only the frozen formulation; the broader analogue remains open only for a materially different optimization or memory design. Direct sample-independent Universal-Transformer memory tokens belong to M020, not M042.

## Loopies additions

### ST-M043 — Layer-local recurrence and operational capacity recycling (3 components)

**PDF:** `2607.16051v2`, §2.1/Fig. 1 pp. 4–5, §2.4 pp. 6–10/Table 1, Appendix Table 4 p. 35. **Components:** (a) fixed `R=2` layer-loop order versus model-loop order; this is the Sequence-sharing layout already catalogued by MoR M009.a, so one matched schedule result jointly covers both evidence slots; (b) operational activation-memory/throughput measurement followed by capacity reinvestment; (c) optional ordinary MoE route evolution, only after dense viability. Native dense blocks first; use a mature MoE library only for the later factorial, count every stored expert, and prove CPU/device parity. Loopies has no learned recursion-depth router (M009.b), exit, loop embedding, or visit adapter. **Evidence:** no SAIR implementation/result; EXP-655 is planned and blocked. Terminal requires the dense schedule controls, then a separate capacity verdict; MoE cannot rescue/obscure the primary verdict.

### ST-M044 — Target-only SPT analogue and conditional RL boundary (2 components)

- **M044.a SPT analogue:** `2607.16051v2`, §4.1 pp. 16–18/Table 8 p. 38; target-only CE masking. The source's approximately 2T token positions and ten epochs are not a local requirement and the paper acknowledges inadequate ablation. On a compatible pretrained recurrent SAIR backbone, compare target-only continuation with matched mixed/trajectory supervision. Native. Conditional, analogue-only, no evidence.
- **M044.b GSPO/DAPO:** §4.2 pp. 18–19; autoregressive group sampling, verifier-normalized advantage, clipping and prompt filtering. Non-applicable unless a compatible stochastic autoregressive model and separate legal plan exist. It is not M043 evidence.

## Main-paper mapping

The thirteen user-elevated entries map as follows: HRM→M007+M001 (M012/M021 supporting); TRM→M010; PTRM→M023; RLM→M036 with M037 split; lambda-RLM→M038; The Duck→M040; LoopLM→M011 plus generic M004/M006; GRAM→M022; LimiX→M035; MAKER→M041 including Appendix F; Percepta's web-only Transformers as Computers entry→M032/M033, with formal `2301.13196` retained separately at M003; Neural Computers→M034, with UT memory→M020 and the learned-RAM analogue→M042; Loopies→M043 and conditional M044. M029 is neither PTRM nor GRAM.

## Resource governance

Historical EXP-649's fixed 285-second decision and observed calibration remain historical evidence for that exact frozen formulation. Its formerly generalized `225`-second rule is invalid for GRAM, PTRM, learned-energy analogues, local recursive models, MAKER analogues, or any other architecture. Each new experiment must measure representative setup, training, checkpoint, and full-fidelity evaluation throughput; derive projected/soft/hard budgets with variability headroom; account for monotonic wall and aggregate process user-plus-system CPU separately; preserve recoverable checkpoints; and reuse only hash-valid caches with explicit charge. The 300-second constraint applies only to final submission inference qualification.
