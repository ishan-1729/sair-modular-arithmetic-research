---
type: research-coverage-contract
title: Super Transformers mechanism coverage contract
description: Exhaustive SAIR applicability, experimental evidence, and terminal-coverage rules for the two supplied Super Transformers source documents.
status: queued
timestamp: 2026-07-21
tags: [autoresearch, recurrent-depth, neural-computers, agentic-systems, coverage]
---

# Super Transformers mechanism coverage contract

## Binding scope and interpretation

This contract reconciles `super_transformers.md` and the user-supplied `super_transformers_main_papers.md` into exactly 42 stable mechanism records. A source claim is not project evidence. “Faithful” means the paper's causal mechanism is implemented without replacing it with a convenient proxy; “bounded-faithful” preserves that causal mechanism at a smaller, dynamically calibrated SAIR rung; “analogue” tests only a stated transferable principle; “exclusion” is terminal only when the literal mechanism is illegal, non-applicable to a self-contained CPU submission, or dominated by a documented compliant abstraction. A negative result closes only the exact tested formulation.

All runnable records are queued behind EXP-648. Readiness root `20261160` is consumed-incomplete, non-scientific, and permanently denied. Recovery closure audits returned `[APPROVE, APPROVE]` and final T648 Judge returned `APPROVE`; root `20261161` is authorized-pending-authentication-child but remains unopened and unconsumed. That approval covers only the approved-base and sole-file authorization-child publication lifecycle, not execution or a scientific surface. This document authorizes no Super Transformers experiment, dependency installation, benchmark access, or phase transition. Candidate 005 remains the 59.5% checked-public / 59.3% independent incumbent and is not completion-eligible.

Every runnable test uses exact autonomous answer accuracy as its primary metric, with exact transition accuracy, first-error position, held-out-prime and width transfer, invalid-output rate, wall time, process user-plus-system aggregate CPU time, peak RSS, artifact size, determinism, and trained-weight dependence. Promotion requires frozen full-fidelity evidence and later Judge review; oracle breadth is diagnostic and can never promote. Mature authoritative external libraries are preferred over hand implementations, version-pinned, license/provenance recorded, and parity-tested against a simple reference on tiny fixtures. CPU is the binding submission path; CUDA may accelerate offline training only when the CPU inference implementation and cross-device parity are separately demonstrated.

## Source reconciliation and deduplication

The twelve named entries in `super_transformers_main_papers.md` remain explicitly mapped; this table is coverage bookkeeping, not a claim that any method has been tested.

| Main-paper entry | Stable mechanism mapping | Coverage class |
|---|---|---|
| Hierarchical Reasoning Model (HRM) | ST-M007, ST-M012, ST-M021 | bounded-faithful plus diagnostics |
| Tiny Recursive Model (TRM) | ST-M010, ST-M012, ST-M021 | bounded-faithful plus diagnostics |
| Probabilistic Tiny Recursive Model (PTRM) | ST-M023, with ST-M029 selection controls | bounded-faithful, runtime-qualified |
| Recursive Language Model (RLM) | ST-M036, ST-M037 | literal exclusion plus internal analogue |
| lambda-RLM | ST-M038 | literal exclusion; resource-accounting lesson only |
| Tufa Labs, "The Duck" | ST-M040 | literal exclusion |
| Looped Transformer / Looped Language Model (LoopLM) | ST-M004, ST-M006, ST-M011, ST-M018 | bounded-faithful |
| Generative Recursive Reasoning (GRAM) | ST-M022, with ST-M029 selection controls | bounded-faithful, runtime-qualified |
| LimiX | ST-M035 | comparison analogue |
| MAKER | ST-M041 | literal exclusion plus selection analogue |
| Transformers as Computers | ST-M003, ST-M032, ST-M033 | learned analogue plus literal exclusion |
| Neural Computers | ST-M034, ST-M042 | literal non-applicability plus learned-memory analogue |

Three additional items in `super_transformers.md` are deliberately not new mechanism IDs. [Tab-TRM](https://arxiv.org/abs/2601.07675) is application evidence for TRM and is deduplicated into ST-M010 rather than counted as a new architecture. The [incorrect-attractor analysis of HRM](https://arxiv.org/abs/2601.10679) is diagnostic evidence mapped to ST-M021's attractor/false-attractor tests. The [ARC Prize technical report](https://arxiv.org/abs/2601.10904) is supporting benchmark/context evidence only, not an independent mechanism or SAIR evaluation result.

## Mechanism ledger

Each record supplies: primary source(s); causal core; SAIR applicability and compliance; coverage status and existing evidence; stable queue mapping and prerequisites; matched controls; metrics and causal/randomized tests; dynamic resource rung; and promotion, stop, and terminal-coverage rules.

### ST-M001 — Adaptive inner computation

- **Sources/core:** [Adaptive Computation Time](https://arxiv.org/abs/1603.08983); learned halting allocates input-dependent recurrent steps.
- **Applicability/compliance/status/evidence:** legal and applicable only inside the mandatory learned operand scan; bounded-faithful, queued as `ST-Q08 / EXP-653`. Candidate 005 has fixed depth and a separate conditional orientation, not ACT coverage.
- **Design:** prerequisite is a stable fixed-depth winner. Compare learned halting, fixed maximum depth, equal-average-compute random halting, and unchanged winner. Measure the common metrics plus halt calibration and compute distribution; force maximum/minimum or shuffle halt scores to establish causality.
- **Resource/decision/terminal:** profile fixed-depth inference, then set a rung from setup + measured per-step throughput with 25% headroom and a hard CPU/RSS ceiling. Promote only with non-inferior exact accuracy on every fresh seed and material isolated CPU savings. Stop on accuracy loss or no savings. Terminal when the faithful head and causal controls pass or fail at full candidate-shaped fidelity.

### ST-M002 — Universal recurrent depth and equilibrium

- **Sources/core:** [Universal Transformer](https://arxiv.org/abs/1807.03819) and [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377); reuse a transition across depth, with DEQ solving for an implicit equilibrium rather than merely adding loops.
- **Applicability/compliance/status/evidence:** legal, high applicability, bounded-faithful `ST-Q01`; Candidate 005 is a cellular shared transition but not a Transformer or implicit-solver test. EXP-643 supplies only recurrent-dynamics diagnostics.
- **Design:** compare a proper looped attention block, an implicit-equilibrium variant, an untied equal-parameter stack, the cellular control, and a wider control. Require trained-width viability before 2x/4x length transfer. Reset, truncate, add visits, and perturb the equilibrium solver; randomize trained transitions.
- **Resource/decision/terminal:** calibrate width-16/32 throughput, then promote to 64/128/256 only after exact viability. Advance on reproducible transfer gain at matched compute without runtime failure; stop each formulation separately. Terminal requires faithful explicit-loop and implicit-equilibrium verdicts, not one proxy verdict for both.

### ST-M003 — Programmable looped Transformer

- **Sources/core:** [Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13196); shared attention repeatedly updates program/state tokens.
- **Applicability/compliance/status/evidence:** an ordinary-initialized learned token executor is legal only as an `analogue`; the paper's literal hand-programmed construction is excluded because fixed program tokens/weights implementing modular arithmetic would cross the trained-parameters boundary. EXP-654 tested soft neural RAM, not the legal learned-token analogue.
- **Design:** ordinary initialization and learned generic state tokens versus a non-addressable looped model, wider cellular model, and randomized transition. Measure exact rollout, visit transfer, trace/state use, and runtime; permute/reset state tokens and ablate attention routes.
- **Resource/decision/terminal:** start at widths 16/32 with measured token-loop cost and 25% headroom. Promote only after trained-width exactness and causal state use, then width transfer. Stop on zero exact viability or provenance ambiguity. Terminal when the learned analogue receives a full matched verdict and the literal hand-programmed construction remains excluded; do not describe the analogue as a faithful reproduction.

### ST-M004 — Learned looping for algorithms and length generalization

- **Sources/core:** [Learning Learning Algorithms](https://arxiv.org/abs/2311.12424) and [Looped Transformers for Length Generalization](https://arxiv.org/abs/2409.15647); train the loop itself and increase loop count with problem length.
- **Applicability/compliance/status/evidence:** legal, directly applicable, bounded-faithful `ST-Q01`. Current width-shared cellular recurrence is adjacent evidence only.
- **Design:** learned looped Transformer versus unlooped/untied/equal-compute cellular controls; train widths 16/32 and evaluate 64+ with preregistered loop scaling. Intervene by freezing train-time loops, extrapolating loops, shuffling visit embeddings, and randomizing weights.
- **Resource/decision/terminal:** measured per-loop wall/CPU determines rungs; never shorten evaluation to fit a cap. Promote on exact width transfer above controls with deterministic CPU headroom. Stop if added loops fail or harm. Terminal after loop-count scaling and length extrapolation are causally separated.

### ST-M005 — Relaxed tying with visit-specific adapters

- **Sources/core:** [Relaxed Recursive Transformers](https://arxiv.org/abs/2410.20672); small layer/visit-specific LoRA relaxes exact sharing.
- **Applicability/compliance/status/evidence:** legal, applicable, bounded-faithful `ST-Q06 / EXP-651`. EXP-617 tested global merged LoRA only and gained 0.5 point.
- **Design:** exact-initialized visit/dilation adapters versus shared core, global LoRA, fully untied and parameter-matched wider controls. Measure exactness, activation, merge parity, parameters and CPU; zero, permute, swap, and randomize adapters.
- **Resource/decision/terminal:** profile adapter overhead before a matched training rung. Promote only for reproducible gain beyond EXP-617 without width retention or CPU regression. Stop on inactive adapters or sub-threshold gain. Terminal after faithful visit-specific adaptation, not global LoRA, is decided.

### ST-M006 — Recurrent-depth latent reasoning

- **Sources/core:** [Huginn](https://arxiv.org/abs/2502.05171) and [Reasoning with Latent Thoughts](https://arxiv.org/abs/2502.17416); reuse a latent block for additional test-time depth and study the resulting computational power.
- **Applicability/compliance/status/evidence:** legal, applicable, bounded-faithful `ST-Q01`. EXP-634 direct attention and EXP-643 cellular-depth diagnostics do not reproduce the recurrent Transformer.
- **Design:** genuine shared latent Transformer against untied, cellular, and wider controls; train with depth variation and test beyond train depth. Measure common metrics plus accuracy-by-visit; truncate, extend, reset latent state, and randomize the block.
- **Resource/decision/terminal:** calibrate depth-throughput at widths 16/32; advance only after exact viability, then full width transfer. Stop if extra depth is neutral/harmful or CPU-infeasible. Terminal requires trained and extrapolated depth curves with causal latent-state evidence.

### ST-M007 — HRM multi-timescale recurrence

- **Sources/core:** [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734); coupled slow high-level and fast low-level modules update at unequal frequencies.
- **Applicability/compliance/status/evidence:** legal, applicable, bounded-faithful `ST-Q02 / EXP-652-HRM`. EXP-620/633 are adjacent hierarchical negatives, not faithful HRM.
- **Design:** two-timescale answer/latent model versus single-timescale, equal-frequency, reset-slow-state, wider, and cellular controls. Measure exactness/transfer plus module-state predictiveness; swap modules, equalize schedules, reset each state, and randomize each module.
- **Resource/decision/terminal:** width-16/32 viability is mandatory before scale. Promote only if hierarchy beats matched simple recurrence and transfers. Stop at zero/near-zero trained-width rollout. Terminal after unequal-frequency causality and both-module weight dependence are shown or rejected.

### ST-M008 — AbbIE blockwise iterative encoding

- **Sources/core:** [AbbIE](https://arxiv.org/abs/2507.08567); recurrently encode blocks while carrying state across sequence chunks.
- **Applicability/compliance/status/evidence:** legal but moderate applicability because SAIR already has a natural bit scan; bounded-faithful `ST-Q09`. No direct evidence.
- **Design:** block-iterative encoder versus bitwise cellular, equal-context Transformer, and no-carried-state controls. Measure exactness by block/width and state use; reorder blocks, reset boundary state, vary block size, and randomize weights.
- **Resource/decision/terminal:** profile block sizes at 16/32 and allocate measured complete rungs. Advance only if exact transfer or CPU efficiency beats the bit-scan control. Stop on boundary-state irrelevance. Terminal after at least two block sizes and causal carry tests.

### ST-M009 — Mixture of recursions

- **Sources/core:** [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524); route tokens through different recursive depths.
- **Applicability/compliance/status/evidence:** legal, conditionally applicable after a stable looped winner, bounded-faithful `ST-Q08`. No faithful test.
- **Design:** learned token/position routing versus fixed depth, equal-average random routing, global ACT, and full-depth control. Measure common metrics, route entropy and per-position compute; permute routes, force shallow/deep paths, and randomize router/core.
- **Resource/decision/terminal:** derive budget from route and core throughput, with full-depth hard ceiling. Promote only with seedwise non-inferior exactness and CPU savings. Stop if routing merely drops work and accuracy. Terminal after routing—not just fewer loops—is causally established.

### ST-M010 — TRM answer-plus-latent recursion

- **Sources/core:** [Tiny Recursive Model](https://arxiv.org/abs/2510.04871); one small shared network repeatedly refines an explicit answer canvas and latent state. [Tab-TRM](https://arxiv.org/abs/2601.07675) is deduplicated application evidence for this same mechanism, not a thirteenth main-paper mechanism.
- **Applicability/compliance/status/evidence:** legal, highly applicable, bounded-faithful `ST-Q02 / EXP-652-TRM`. Existing persistent-latent and direct-attention tests are not faithful.
- **Design:** explicit answer+latent TRM versus answer-only, latent-reset, cellular and wider controls at matched work. Measure exact answer by recursion/width, first error, and Q/confidence if used; reset/swap canvases, truncate/extend recursion, randomize core/head.
- **Resource/decision/terminal:** width-16/32 viability before any Tier-7 run. Promote on meaningful trained-width exactness plus superior transfer. Stop if zero exact rollout. Terminal requires a faithful small TRM with all state interventions.

### ST-M011 — LoopLM latent pretraining and learned depth allocation

- **Sources/core:** [Ouro/LoopLM](https://arxiv.org/abs/2510.25741); looped latent computation plus entropy-regularized learned depth allocation.
- **Applicability/compliance/status/evidence:** legal; bounded-faithful `ST-Q03`. EXP-634/643 lack LoopLM pretraining and entropy allocation.
- **Design:** looped model with entropy depth objective versus fixed depth, no entropy, equal-compute random depth, and untied controls. Measure exactness, depth distribution, entropy, CPU and transfer; clamp/shuffle depth, remove entropy, randomize block/router.
- **Resource/decision/terminal:** calibrate both training and variable inference work at small widths. Promote after exact viability and a depth-allocation gain, not language-model loss. Stop if learned depth is unused or harmful. Terminal after the allocation mechanism receives a causal matched verdict.

### ST-M012 — Identity conditioning and universal reasoning

- **Sources/core:** [Independent TRM audit](https://arxiv.org/abs/2512.11847) and [Universal Reasoning Model](https://arxiv.org/abs/2512.14693); isolate identity conditioning, recurrent inductive bias, and universal/shared reasoning components.
- **Applicability/compliance/status/evidence:** structural identity features are legal only when generic and benchmark-agnostic; bounded-faithful audit `ST-Q02`. No matched SAIR audit.
- **Design:** TRM/URM with and without generic position/role identity, matched parameters and recursion. Measure exactness across unseen primes/widths; permute identifiers, replace with constants, randomize embeddings/core.
- **Resource/decision/terminal:** piggyback on viable EXP-652 checkpoints with separately charged evaluation. Promote no model from identifiers alone unless fresh-seed transfer and leakage checks pass. Stop on identity lookup behavior. Terminal when the claimed conditioning gain is isolated and audited.

### ST-M013 — Visual looped reasoning and shortcut modulation

- **Sources/core:** [LoopViT](https://arxiv.org/abs/2602.02156) and [LoopFormer](https://arxiv.org/abs/2602.11451); repeated hybrid blocks, dynamic exit, variable trajectories, and time/step-conditioned shortcut modulation.
- **Applicability/compliance/status/evidence:** visual encoding is analogue-only; shortcut modulation is bounded-faithful `ST-Q05`. EXP-648 tests only natural endpoint consistency.
- **Design:** separate A/B tests for shortcut/time conditioning and dynamic exit against unchanged looped controls; no pixel conversion. Measure exactness-by-visit and stability; zero/shuffle time and step signals, remove shortcut, force exits.
- **Resource/decision/terminal:** calibrate each isolated mechanism; do not bundle them. Promote only if one improves full exact rollout without retention/runtime harm. Stop exact formulations independently. Terminal when shortcut modulation and exit each have matched causal verdicts.

### ST-M014 — Mamba-2/TRM hybrid

- **Sources/core:** [Tiny Recursive Reasoning with Mamba-2 Attention Hybrid](https://arxiv.org/abs/2602.12078); replace part of TRM with selective state-space sequence mixing.
- **Applicability/compliance/status/evidence:** legal and applicable, bounded-faithful `ST-Q07 / EXP-650`. No library-backed test exists.
- **Design:** pinned authoritative Mamba-2/SSM library implementation versus ConvGRU, attention TRM, and parameter/compute-matched simple recurrence; tiny-fixture parity required. Measure exactness, transfer, wall/CPU/RSS; zero/selectivity ablations and randomized SSM/core tests.
- **Resource/decision/terminal:** first benchmark library CPU kernels and candidate-shaped projection, allowing CUDA only for training with CPU parity. Promote after trained-width viability and better exactness per wall second. Stop on unsupported CPU path or zero transfer. Terminal after proper-library evidence, never a hand-coded approximation.

### ST-M015 — Structural symbol/operand equivariance

- **Sources/core:** [SE-RRM](https://arxiv.org/abs/2603.02193); architecture enforces legal permutation equivariance.
- **Applicability/compliance/status/evidence:** structural operand equivariance is legal only for generic commutativity/role symmetries, with no answers, benchmark IDs, or encoded arithmetic circuit; bounded-faithful `ST-Q10`. EXP-647 closes only its endpoint-loss formulation.
- **Design:** equivariant recurrent layers versus augmentation-only, consistency-loss, unchanged cellular and randomized symmetry controls. Measure exactness and equivariance residual on untouched identities; swap operands, permute generic role channels, break tying, randomize trained layers.
- **Resource/decision/terminal:** small-width parity then full matched rung with measured overhead. Promote on fresh width-256 exact gain and single-pass CPU fit. Stop on no gain or leakage risk. Terminal after an architectural equivariance test distinct from EXP-647.

### ST-M016 — Recursive Stem and Inference Machines

- **Sources/core:** [Recursive Stem Model](https://arxiv.org/abs/2603.15641) and [Recursive Inference Machines](https://arxiv.org/abs/2603.05234); separate input stem, recurrent inference core, answer/latent state, and training contract.
- **Applicability/compliance/status/evidence:** legal, applicable, bounded-faithful `ST-Q02`. No faithful implementation; EXP-633 is only adjacent.
- **Design:** shared inference core with explicit stem/head versus TRM, no-stem, reset-latent, cellular and wider controls. Measure exactness/transfer/state dependence; swap stems/latents, bypass recurrence, and randomize components.
- **Resource/decision/terminal:** use EXP-652 viability rungs and measured component costs. Advance only if decomposition improves exact rollout beyond TRM/control. Stop at nonviability. Terminal after RSM/RIM-specific stem/core separation receives a causal verdict.

### ST-M017 — Autoregressive recursive negative control

- **Sources/core:** [Tiny Autoregressive Recursive Models](https://arxiv.org/abs/2603.08082); controlled conversion tests whether recursion helps autoregressive generation.
- **Applicability/compliance/status/evidence:** legal but low applicability because SAIR is fixed-output algorithmic inference; bounded-faithful negative-control `ST-Q09`. No direct test.
- **Design:** only after a viable recursive Transformer, compare autoregressive, parallel answer-canvas, and cellular decoding at matched compute. Measure exactness, error compounding and CPU; teacher-force outputs, truncate recursion, randomize core.
- **Resource/decision/terminal:** reuse viable checkpoints and calibrate decoding overhead. Never promote on token loss. Terminal either a matched exact-answer verdict or documented non-applicability if no viable recursive Transformer exists after ST-Q01/02 closes.

### ST-M018 — Looping, generalization, and visit curriculum

- **Sources/core:** [Loop, Think, & Generalize](https://arxiv.org/abs/2604.07822); train across loop counts so additional thinking generalizes.
- **Applicability/compliance/status/evidence:** legal, applicable, bounded-faithful `ST-Q03`. EXP-643 is diagnosis, not a loop curriculum.
- **Design:** variable-visit curriculum versus fixed visits and equal-work mixed-depth controls. Measure exactness across train/unseen visit counts and widths; hold visits constant, extrapolate visits, shuffle visit conditioning, randomize weights.
- **Resource/decision/terminal:** profile each visit count and budget the full mixture. Promote on reproducible unseen-depth and width gain. Stop if extra loops hurt or conditioning is unused. Terminal after a complete visit-generalization surface and causal intervention.

### ST-M019 — Denoising recursion

- **Sources/core:** [Denoising Recursion Models](https://arxiv.org/abs/2604.18839); iteratively reverse defined target corruption.
- **Applicability/compliance/status/evidence:** literal arbitrary residue corruption is ambiguous; compliant natural-trajectory analogue `ST-Q00 / EXP-648` is active. EXP-610 closed generic corruption, not every denoising formulation.
- **Design:** EXP-648 natural K=8 endpoints versus lambda-zero matched control, with successful-window retention. Measure exact rollout/first error/transition and causal endpoint-loss removal; no synthetic target-state corruption.
- **Resource/decision/terminal:** binding EXP-648 dynamic calibration and guards apply. Promote only through its preregistered screen/confirmation gates. Stop the exact natural-endpoint formulation on a gate miss. Terminal denoising coverage records literal ambiguity, EXP-610 rejection, and EXP-648 analogue verdict without claiming faithful DRM reproduction.

### ST-M020 — Explicit learned workspace

- **Sources/core:** [Universal Transformers Need Memory](https://arxiv.org/abs/2604.21999); memory tokens trade state capacity against recurrent depth.
- **Applicability/compliance/status/evidence:** legal and applicable; one bounded-faithful formulation is closed by `ST-Q04 / EXP-646/654` at zero trained-width exactness.
- **Design:** reopen only with materially different optimization-grounded memory, compared with wider hidden state, no-memory, and addressable memory. Measure exactness, transfer, capacity use, CPU; reset/permutate/zero memory and randomize weights.
- **Resource/decision/terminal:** any reopen starts at width 16 after a new preregistration and measured throughput. Stop on zero viability or causal irrelevance. Terminal now for the exact eight-slot/non-addressable formulation; family terminal only after another materially different justified design or a documented mechanism-exhaustion Judge.

### ST-M021 — Learned attractors and equilibrium reasoners

- **Sources/core:** [Attractor Models](https://arxiv.org/abs/2605.12466) and [Equilibrium Reasoners](https://arxiv.org/abs/2605.21488); learn task-conditioned solution attractors and scale depth/breadth toward them. The [incorrect-attractor HRM analysis](https://arxiv.org/abs/2601.10679) supplies supporting false-attractor diagnostics, not a separate mechanism.
- **Applicability/compliance/status/evidence:** legal, applicable, bounded-faithful `ST-Q05`. EXP-643 found no Candidate-005 stability signature, so a reopen requires a new measured signature or a materially different model.
- **Design:** attractor/equilibrium objective versus ordinary recurrence, fixed depth, and equal-compute controls. Measure convergence, exactness and basin stability; perturb initialization/state, extend iterations, swap inputs, randomize energy/core.
- **Resource/decision/terminal:** first demonstrate a discriminating convergence signature, then calibrate repetitions dynamically. Promote only if convergence predicts and improves exact fresh-seed answers. Stop without signature. Terminal after faithful attractor diagnostics and causal objective test, not endpoint proximity alone.

### ST-M022 — GRAM generative trajectory breadth

- **Sources/core:** [Generative Recursive Reasoning](https://arxiv.org/abs/2605.19376); amortized variational training creates stochastic latent trajectories and parallel hypotheses.
- **Applicability/compliance/status/evidence:** legal if input-only at inference and selected without labels; bounded-faithful `ST-Q11`. EXP-649 measured a different frozen noise portfolio's runtime only.
- **Design:** trained GRAM versus deterministic recurrence, sigma-zero, equal-cost random perturbation, and randomized selector; report selected exactness, diversity, oracle union, regret, validity and CPU. Intervene on latent noise, posterior/prior, trajectory count, and weights.
- **Resource/decision/terminal:** before formal inputs, require three complete candidate-shaped input-only repeats on new roots; `1.20 * max(wall) + 5 <= 225` seconds, separate CPU/RSS accounting, no reduced fidelity. Promote only target-free selected accuracy; oracle cannot promote. Stop on runtime or noncomplementarity. Terminal after faithful variational breadth and selector tests.

### ST-M023 — PTRM stochastic recursive search

- **Sources/core:** [Probabilistic TRM](https://arxiv.org/abs/2605.19943); inject Gaussian noise at deep recursion and select parallel outcomes with a learned Q head.
- **Applicability/compliance/status/evidence:** legal under target-free selection; bounded-faithful `ST-Q11`. Frozen EXP-649's candidate-shaped path is runtime-blocked and breadth-unanswered, so only a materially different formulation on new roots may reopen.
- **Design:** PTRM with trained Q versus sigma-zero, deterministic TRM, random/Q-shuffled selection, and equal-cost depth. Measure selected exactness, oracle union diagnostic, regret, diversity, validity, runtime; disable noise, swap Q, randomize head/core.
- **Resource/decision/terminal:** the same three-repeat input-only gate and `1.20 * max(wall) + 5 <= 225` wall rule apply with separate CPU. Promote never from oracle. Stop on gate miss or selector noncausality. Terminal after a genuinely different PTRM implementation receives faithful breadth and selection evidence.

### ST-M024 — HRM-Text hierarchy transfer

- **Sources/core:** [HRM-Text](https://arxiv.org/abs/2605.20613); apply slow/fast hierarchical recurrence to sequence pretraining.
- **Applicability/compliance/status/evidence:** language pretraining is non-applicable literally; hierarchy transfer is analogue-only `ST-Q02`. The dedicated HRM multi-timescale record owns the SAIR test.
- **Design:** no language-model replication. Compare the same slow/fast schedule inside structured arithmetic to single-timescale controls; measure exactness and module causality through resets/swaps/randomization.
- **Resource/decision/terminal:** shares EXP-652-HRM rungs, with no extra large-corpus budget. Terminal when HRM hierarchy is tested for SAIR and language-specific pretraining is documented non-applicable; no separate giant model is warranted.

### ST-M025 — Linear-time looped Transformer

- **Sources/core:** [LT²](https://arxiv.org/abs/2605.20670); linear-time sequence mixing inside repeated Transformer depth.
- **Applicability/compliance/status/evidence:** legal, applicable at wide inputs, bounded-faithful `ST-Q07 / EXP-650-extension`. No direct test.
- **Design:** pinned authoritative implementation when available versus Mamba-2 hybrid, attention loop, and ConvGRU at matched parameters/work. Measure exactness, scaling curves, CPU/RSS; ablate loop sharing and randomize mixer/core.
- **Resource/decision/terminal:** benchmark kernel crossover widths before training; dynamically size rungs. Promote only if exact transfer and candidate-shaped CPU both improve. Stop if linear asymptotics do not help at SAIR widths. Terminal after library-backed parity and full scaling evidence or documented implementation unavailability.

### ST-M026 — STARS spectral stability

- **Sources/core:** [STARS](https://arxiv.org/abs/2605.26733); stabilize recurrent dynamics so accuracy does not collapse with extra loops.
- **Applicability/compliance/status/evidence:** legal, conditional; `ST-Q05`. EXP-645 is closed because EXP-643's spectral proxy was contractive and unrelated to failure.
- **Design:** reopen only after a new measured instability signature; then compare spectral regularization with unchanged and equal-regularization controls. Measure exactness-by-visit, spectral/Jacobian proxies, norms; remove penalty, extend visits, randomize weights.
- **Resource/decision/terminal:** signature collection uses cached checkpoints; training rung is dynamically calibrated only if triggered. Stop without replicated signature or causal gain. Terminal currently conditional-closed for Candidate 005; a new architecture needs its own diagnosis.

### ST-M027 — Fixed-point reasoning and convergence halting

- **Sources/core:** [Fixed-Point Reasoners](https://arxiv.org/abs/2606.18206); stable looped computation with convergence-based adaptive depth.
- **Applicability/compliance/status/evidence:** legal, applicable after a stable looped winner, bounded-faithful `ST-Q05/08`. Candidate 005 has no supported fixed-point signature.
- **Design:** fixed-point training plus convergence halt versus fixed-depth, ACT, DEQ and ordinary looped controls. Measure exactness, residual, iteration count and CPU; perturb state, alter tolerance only on preregistered validation, force depth, randomize core.
- **Resource/decision/terminal:** profile solver iterations and set complete candidate-shaped bounds. Promote only if every seed retains exactness and saves CPU or gains transfer. Stop on false convergence. Terminal requires calibrated convergence causality and untouched-threshold evaluation.

### ST-M028 — LOTUS latent/explicit trace supervision

- **Sources/core:** [LOTUS](https://arxiv.org/abs/2606.31779); align parallel latent positions to explicit reasoning steps.
- **Applicability/compliance/status/evidence:** offline exact teacher traces are legal for training only; no traces, expected answers, or teacher code may enter submission inference. Bounded-faithful `ST-Q12`; no direct test.
- **Design:** LOTUS-style trace supervision versus endpoint-only, transition-only, equal-label-budget and shuffled-trace controls. Measure autonomous exactness/first error/transfer; shift/shuffle traces, mask positions, randomize trained weights, and audit inference package.
- **Resource/decision/terminal:** generate/hash traces offline, then calibrate training throughput and full evaluation separately. Promote only on autonomous fresh-seed exact gain with zero inference-time teacher dependence. Stop on shuffled equivalence or leakage. Terminal after faithful parallel trace alignment and a clean submission audit.

### ST-M029 — Energy-guided recursive selection

- **Sources/core:** [Energy-guided Recursive Model](https://arxiv.org/abs/2607.10128); learned energy ranks/guides recurrent candidates.
- **Applicability/compliance/status/evidence:** legal if energy is trained without evaluation labels and materially selects model outputs; bounded-faithful `ST-Q11`. Margin routing in EXP-640–642 is adjacent, not an energy model.
- **Design:** learned energy versus Q head, margin selector, random selector, sigma-zero and deterministic controls. Measure selected exactness, oracle diagnostic, ranking regret/calibration and CPU; shuffle energies, swap heads, randomize energy/core.
- **Resource/decision/terminal:** first pass the same 225-second three-repeat input-only gate for the proposed candidate count. Promote only selected, never oracle, accuracy. Stop on selector noncausality or runtime. Terminal after independent energy training and untouched target-free selection evidence.

### ST-M030 — Normalization and visit-aware depth scaling

- **Sources/core:** [LayerNorm as Implicit Gain Control](https://arxiv.org/abs/2607.10681) and [DeepLoop](https://arxiv.org/abs/2607.13491); normalization/residual scaling must account for repeated parameter visits.
- **Applicability/compliance/status/evidence:** legal, conditional `ST-Q05`. EXP-644 is closed for Candidate 005 because EXP-643 found no matching instability signature.
- **Design:** when triggered, test LayerNorm/gain and visit-aware residual scaling separately against exact-init unchanged, generic scaling, and equal-parameter controls. Measure exactness-by-visit, norms and spectral proxies; remove/shuffle visit scaling and randomize core.
- **Resource/decision/terminal:** cached signature first, calibrated matched training second. Stop absent replicated signature or causal exact gain. Terminal for Candidate 005 is conditional-closed; every materially new looped architecture requires diagnosis before this mechanism is declared covered.

### ST-M031 — Temporal middle-layer recurrence

- **Sources/core:** [T²MLR](https://arxiv.org/abs/2607.15178); recurrence is confined to temporal middle layers rather than the whole stack.
- **Applicability/compliance/status/evidence:** legal but `analogue` for the present parallel fixed-output SAIR decoder because T²MLR changes recurrence across autoregressive time, not repeated depth for one fixed output. It becomes bounded-faithful only if a separately justified autoregressive candidate decoder exists. No direct evidence.
- **Design:** if an autoregressive decoder is justified, compare temporal middle-layer recurrence with whole-block temporal recurrence, untied depth, and matched decoding controls. Otherwise test only the stated middle-layer state-transfer analogue and do not conflate it with recurrent-depth evidence. Measure exactness/transfer/CPU and layer-state use; bypass/reorder recurrent middle layers, truncate temporal carry, randomize recurrent/nonrecurrent parts.
- **Resource/decision/terminal:** start only after a viable decoder establishes the relevant recurrence axis, profile per-layer cost, then size rungs. Advance on better exactness per CPU without retention loss. Stop if temporal carry is noncausal. Terminal via faithful autoregressive evidence or a Judge-accepted analogue/non-applicability verdict when no justified autoregressive decoder exists.

### ST-M032 — Learned fast indexed execution

- **Sources/core:** Percepta's [Transformers as Computers](https://www.percepta.ai/blog/can-llms-be-computers) suggests geometric/low-dimensional attention for fast memory lookup; only the learned-memory primitive transfers.
- **Applicability/compliance/status/evidence:** blog claim, analogue-only `ST-Q04`; EXP-654's soft-addressing design is closed at zero exactness and uniform writes.
- **Design:** any reopen must be materially different and ordinary-initialized, with pinned efficient attention primitives, versus simple recurrence, wider state, non-addressable memory and randomized addressing. Measure exactness, collisions, scaling and causal memory use.
- **Resource/decision/terminal:** tiny parity and CPU scaling before training. Stop on uniform/unused memory or infeasible scaling. Terminal for the tested EXP-654 primitive; broader claim needs a distinct optimization-grounded design and public reproducible implementation evidence.

### ST-M033 — Compiled deterministic Transformer executor

- **Sources/core:** Percepta's compiled C-to-token executor claim; constructed weights/program tokens execute supplied code.
- **Applicability/compliance/status/evidence:** literal mechanism is prohibited for SAIR inference because a compiled modular multiplication/reduction program, hand-assigned semantics, or nearly complete circuit would supply the answer. Status `exclusion`, stable `ST-X01`; no experiment is legal.
- **Design:** no runner. Compliance controls are static artifact inspection, parameter-provenance audit, randomized-weight collapse, and confirmation that no instruction/program payload encodes arithmetic. Metrics are violations found, not accuracy.
- **Resource/decision/terminal:** no compute rung. Terminal coverage is this reasoned exclusion plus audit of every learned-execution candidate; a future genuinely learned ordinary-initialized implementation belongs to the programmable-loop or learned-indexed-execution records, not this row.

### ST-M034 — Neural Computer pixel/video runtime

- **Sources/core:** [Neural Computers](https://arxiv.org/abs/2604.06425); a video model learns CLI/GUI frames, I/O alignment, actions, and persistent runtime state from traces.
- **Applicability/compliance/status/evidence:** the literal pixel/video runtime is non-applicable to tensor-native modular arithmetic and CPU submission; status `analogue/exclusion`, `ST-X02`. EXP-654 tested persistent learned memory, not pixels.
- **Design:** no pixel replication. Transfer only persistent state, learned read/write, routine reuse, controlled updates, and symbolic-stability diagnostics through the explicit-workspace and learned-neural-RAM records. Exactness and causal memory metrics apply there.
- **Resource/decision/terminal:** no video-model compute rung. Terminal when literal non-applicability is recorded and every transferable primitive has a mapped legal test; current memory formulation is rejected, not the entire Neural Computer aspiration.

### ST-M035 — LimiX compact structured-data foundation model

- **Sources/core:** [LimiX](https://arxiv.org/abs/2509.03505); masked joint-distribution pretraining enables context-conditioned structured-data prediction.
- **Applicability/compliance/status/evidence:** legal but comparison-only and low direct applicability; `analogue`, `ST-Q13`. It is not recursive depth or learned execution.
- **Design:** only run if a compact arithmetic-specific masked-variable hypothesis is preregistered; compare with equal-parameter supervised cellular and Transformer controls. Measure autonomous exactness/transfer and adaptation cost; mask/shuffle context and randomize weights.
- **Resource/decision/terminal:** small-width viability with measured pretraining cost. Stop if task-specific training dominates or context is noncausal. Terminal via a matched compact structured-model probe or documented low-value deferral after higher-priority families close.

### ST-M036 — RLM external REPL recursion

- **Sources/core:** [Recursive Language Models](https://arxiv.org/abs/2512.24601); externalize context, let an LLM write REPL code, decompose, and recursively call models.
- **Applicability/compliance/status/evidence:** external model calls, arbitrary Python, hosted weights, and REPL execution are incompatible with the self-contained trained-model CPU submission and risk implementing the forbidden solver. Status `exclusion`, `ST-X03`.
- **Design:** no submission experiment. The legal transferable principles—state externalization and learned selection—map to the workspace, stochastic-trajectory, probabilistic-recursion, and energy-selection records. Compliance audits reject subprocess/network/tool calls and arithmetic code.
- **Resource/decision/terminal:** no rung. Terminal coverage is the explicit systems-level exclusion plus mapped internal learned analogues; RLM benchmark gains are not SAIR evidence.

### ST-M037 — RLM search, uncertainty, and long-horizon recursion

- **Sources/core:** [Recursive Models for Long-Horizon Reasoning](https://arxiv.org/abs/2603.02112) and [RLM Meets Uncertainty](https://arxiv.org/abs/2603.15653); recursive decomposition plus search/self-consistency/confidence, with evidence that recursion alone can hurt.
- **Applicability/compliance/status/evidence:** literal external recursion/search is non-applicable; uncertainty-based target-free selection is analogue-only `ST-X04 -> ST-Q11`.
- **Design:** no agents or REPL. Test confidence/energy selection only among legal learned trajectories, with random/shuffled selector controls, selected exactness, oracle diagnostic and regret; intervene on confidence and trajectory diversity.
- **Resource/decision/terminal:** ST-Q11's 225-second input-only gate applies. Terminal when external recursion is excluded and the uncertainty-selection analogue is faithfully assessed without claiming RLM reproduction.

### ST-M038 — λ-RLM typed functional runtime

- **Sources/core:** [The Y-Combinator for LLMs](https://arxiv.org/abs/2603.20105); typed verified combinators provide recursive control, termination, and cost bounds around bounded neural leaves.
- **Applicability/compliance/status/evidence:** a deterministic functional decomposition of modular arithmetic would be a hand-coded solver; external leaf calls also violate self-contained inference. Status `exclusion`, `ST-X05`.
- **Design:** no inference implementation. Transfer only formal resource accounting and fail-closed bounded control to experiment infrastructure, never answer computation. Audit submission for combinator/program payloads.
- **Resource/decision/terminal:** no scientific rung. Terminal coverage is legal exclusion with infrastructure-only adoption; better experiment accounting cannot be counted as model accuracy evidence.

### ST-M039 — Recursive Agent Harnesses

- **Sources/core:** [Recursive Agent Harnesses](https://arxiv.org/abs/2606.13643); the recursive unit is a full tool-using agent with filesystem, code execution, planning, and subagents.
- **Applicability/compliance/status/evidence:** external agents/tools are non-applicable and prohibited in submission inference; status `exclusion`, `ST-X06`.
- **Design:** no experiment. Research orchestration may use agents outside inference but cannot generate benchmark answers or become candidate evidence. Static/runtime audits reject tools, network and child processes.
- **Resource/decision/terminal:** no rung. Terminal when the architectural exclusion and separation between research workflow and frozen artifact are audited.

### ST-M040 — The Duck interactive code agent

- **Sources/core:** ARC Prize's [Milestone #1 report](https://arcprize.org/blog/arc-prize-2026-milestone-1) and the supplied Duck notebook/write-up; a large local LLM observes an environment, writes/runs Python in a live REPL, acts, and repeats.
- **Applicability/compliance/status/evidence:** environment interaction, 27B-class weights, live Python, and improvised helper code are incompatible with SAIR's deterministic self-contained CPU inference and could directly calculate answers. Status `exclusion`, `ST-X07`.
- **Design:** no runner. Audit submission for eval/exec, interpreters, tools, hidden stateful environment calls, or solver code. No ARC score transfers to SAIR.
- **Resource/decision/terminal:** no rung. Terminal coverage is explicit non-applicability and security/compliance audit; its generic “light harness” lesson does not justify an inference mechanism.

### ST-M041 — MAKER extreme decomposition and voting

- **Sources/core:** [MAKER](https://arxiv.org/abs/2511.09030); decompose into microagent steps and vote repeatedly to suppress compounded errors.
- **Applicability/compliance/status/evidence:** literal multiagent orchestration is non-applicable; voting over expensive learned trajectories is only a diagnostic unless a target-free selector fits runtime. Status `exclusion/analogue`, `ST-X08 -> ST-Q11`.
- **Design:** no microagents. Legal analogue compares learned selector with fixed vote/random selector under identical candidate sets; report selected exactness, oracle diagnostic, disagreement and CPU. Disable/shuffle selector and randomize models.
- **Resource/decision/terminal:** ST-Q11 resource gate applies to the analogue. Oracle or majority-vote breadth alone cannot promote. Terminal when multiagent mechanism is excluded and legal error-correction selection has a verdict.

### ST-M042 — Learned neural RAM and reusable runtime state

- **Sources/core:** [Neural Computers](https://arxiv.org/abs/2604.06425), [Universal Transformers Need Memory](https://arxiv.org/abs/2604.21999), and learned-execution sources; train persistent registers, read/write addressing, trace memory, controlled updates, and reuse from ordinary initialization.
- **Applicability/compliance/status/evidence:** legal and highly relevant, bounded-faithful `ST-Q04 / EXP-654`. The tested eight-slot soft-addressing and non-addressable workspace formulations are closed: zero exact at widths 16/24/32, nearly uniform writes, and zero memory-ablation/dependence effect.
- **Design:** reopen only with a materially different memory/optimization mechanism; compare decoded cellular, parameter-matched wider state, non-addressable and address-disabled controls. Measure exactness/transfer, collisions, update reuse, CPU; reset, permute, zero, uniformize memory and randomize all trained components.
- **Resource/decision/terminal:** width-16 viability first with dynamic measured budget, then 24/32 and candidate projection. Stop on nonviability or causal irrelevance. Terminal for current formulation; family terminal requires a justified distinct design verdict or mechanism-exhaustion Judge.

## Queue order and completion accounting

`ST-Q00` is EXP-648 and remains the sole active task. After its phase is resolved, the staged order is binding: (1) structural equivariance; (2) shared recurrent depth; (3) TRM/HRM/RIM/RSM answer-plus-latent recurrence; (4) separate, unbundled LoopFormer, DRM, and LOTUS tests; (5) visit/dilation adapters (EXP-651); (6) library-backed Mamba/linear operators (EXP-650); (7) a materially new learned-memory/execution formulation; (8) stability/fixed-point mechanisms only after a qualifying replicated signature; (9) stochastic breadth and energy selection only after the candidate-shaped runtime gate; (10) adaptive compute (EXP-653) only after a stable winner; and (11) autoregressive/T²MLR work only if a justified autoregressive decoder exists. LoopLM/visit curricula and other shared-loop controls belong inside the shared-depth stage; they are not permission to bundle LoopFormer, DRM, or LOTUS. Compact structured-model comparison remains a low-value conditional after the mechanism-bearing stages. EXP-652 retains answer-plus-latent viability, and EXP-650 through EXP-653 retain their established meanings.

A mechanism is terminally covered only when its record's stated faithful or bounded-faithful test has complete durable evidence, or its explicit legal/non-applicability exclusion and every transferable analogue are documented. Conditional mechanisms remain open until their prerequisite appears or a Judge rules the prerequisite exhausted. Coverage counts at creation: 42 total; 0 family-complete; 8 literal exclusions with mapped legal boundaries; 3 exact tested formulations closed (EXP-646/654 memory/workspace, EXP-647 consistency, and the frozen EXP-649 runtime formulation); all remaining faithful mechanisms queued or conditional. These counts do not imply scientific progress or candidate promotion.
