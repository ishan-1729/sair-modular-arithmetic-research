---
type: experiment-preregistration
title: EXP704 Candidate005 plus EXP701 Tier-7 integration
description: Frozen Candidate005 retention through Tier 6 with an EXP701 teacher-cosine Tier-7 specialist.
status: frozen-before-execution
date: 2026-08-12
timestamp: "2026-08-12T06:20:00Z"
tags: [autoresearch, candidate006, tier7, exp701, integration]
---

# EXP704 Candidate005 + EXP701 Tier-7 integration

## Decision card

- **Hypothesis:** Candidate005 can retain its exact scored Tiers 1–6 behavior while EXP701's frozen `teacher_cosine` H92/W256 specialist converts the Tier-7 zero fallback into useful exact answers, producing a full-candidate score above Candidate005 without exceeding the official 300-second limit.
- **Measured failure:** Candidate005 scores 593/1000 non-Tier-0 cases on its independent full set, including only 2/100 at Tier 7. EXP703 completed the EXP701 factorial and identified `teacher_cosine` as the winner at 361/600 (60.1667%) W256 rollout exact accuracy.
- **Baseline identity:** The immutable Candidate005 artifact is `submission/{model.py,manifest.json,weights.safetensors}` with SHA-256 values `e3d4aa9ff3e22dde516d261b099826622fa37d0688fae47c07ff95fe876e9477`, `ee49f91b8a1a4aa2cde418ebec59ea1890d93fda6dc72338989c0aa3308e7c83`, and `2f6ac3199d0fd9e16afae165c2d4a16919e6c8bf4545363a6f12d077279d27ca`.
- **Changed variable:** Add exactly one frozen `tier7.*` state dict and route only moduli with bit length 129–256 to it. Candidate005 handles all moduli up to 128 bits unchanged; moduli above 256 bits keep the deterministic zero fallback. No training, retuning, answer-dependent routing, or arithmetic repair is allowed.
- **Specialist identity:** EXP701 `teacher_cosine` checkpoint SHA-256 `b2ef6691ce98195a857401692cbf94335ebeebd22d0e4077a2ee30d6669f715e`, trained under formal root `20261414` from scientific source commit `0027f66197a00ea9392e04cc7060c899500aa091` and terminal archive `dcd500925e9c57ce7daf4af78e068747c91fc71c84df04a795d1e50e456a42fc`.
- **Public route:** Pairwise operand reductions explicitly sanctioned by the challenge feed the frozen learned recurrence. Only the learned recurrence emits the final residue; inference contains no hand-coded multiplication, long division, Barrett/Montgomery reduction, CRT, validity selector, or label access.
- **Smallest falsifier:** On a fresh frozen official-generator seed, any scored Tiers 1–6 prediction differs from Candidate005, Tier 7 fails to exceed Candidate005, a randomized Tier-7 weight control does not collapse, or the exact artifact cannot finish under the 300-second official sandbox limit.
- **Primary oracle:** Full exact-answer accuracy from the pinned official generator and evaluator on untouched seeds. Tier-level exact accuracy and output validity are secondary diagnostics.
- **Promotion gate:** Every fresh full set must be complete and valid; Tiers 0–6 predictions must be byte-identical to Candidate005; aggregate Tier-7 accuracy must exceed 50%; every full-set overall score must exceed 59.3%; trained weights must beat the randomized control by at least 40 Tier-7 percentage points; predictions must be deterministic; static/preprocess/artifact checks must pass; and the isolated official run must finish within 285 seconds for promotion (300 seconds remains the hard validity limit).
- **Stop rule:** Stop with a durable negative result after the first full fresh seed if lower-tier equivalence, validity, Tier-7 improvement, or the hard runtime limit fails. Otherwise run all five preregistered fresh seeds, then one isolated official sandbox run. Do not read checked answers before the candidate and policy are frozen.
- **Retention gates:** Preserve Candidate005 exactly on scored Tiers 1–6; retain zero fallback above Tier 7; retain the exact three-file submission shape; retain deterministic CPU FP32 semantics and the challenge's trained-parameter boundary. Tier 0 is a non-ranking diagnostic tier and its width-256 rows necessarily exercise the same public bit-length route as Tier 7 because tier IDs are not inference inputs.
- **Cost model:** No training and no paid provider. One direct full-set screen is expected to take roughly 245 seconds on the local CPU; five fresh sets plus controls and one sandbox confirmation are budgeted at 2,400 wall seconds, with a 3,000-second hard lifecycle ceiling and 300-second cleanup reserve.
- **Evidence outputs:** Built and randomized artifact receipts, canary, per-seed full results, lower-tier parity, randomized control, determinism, isolated sandbox receipt, terminal summary, environment, run record, metrics, registry entry, and knowledge decision.
- **Allowed files:** `configs/exp_704*`, `experiments/exp_704*`, `tests/test_exp704*`, `tools/build_candidate006*`, this preregistration, EXP704 run/evaluation records, registry/scoreboard/decision-log indexes, and local `.scratch/exp704*` lifecycle inputs/outputs. Existing experiment and `submission/` bytes are read-only.

## Frozen seeds

The five independent full-set seeds were derived from the literal labels `exp704-candidate006-fresh-official-seed-{1..5}-frozen-20260812` before any EXP704 prediction was computed:

1. `699033186f6a205c27e2ef809fc354a7346c4116defa60c3da38b5d0c7cefe78`
2. `a835ab129af0192dbaeef0fddb1989997e08d6ee62831fac2506f01ce7fa6b05`
3. `cfcff31430bf3b5b188d316f678a73e3e6a8357753e29115b143ce10ade6706c`
4. `e817269c8b139c09efdb47ecc8fddcc5a58bb2ec4f7ae3f024d3d2a2b3e34a78`
5. `ab409f17232461e65074d82fac3cf5f290fb159ea0d05984d0f6abb6b566a9fa`

The randomized Tier-7 control seed is `5430431669808135325`. Seed 1 is also the frozen isolated-sandbox seed; it is not a checked or previously used Candidate005 seed.

## Interpretation boundary

EXP704 tests a frozen integration, not a new architecture or a new training claim. A positive result may promote a new intermediate candidate but cannot establish private-seed performance, 85% completion, or superiority above Tier 7. A negative result does not invalidate EXP701's factorial result; it only rejects this candidate-shaped transfer under the frozen routing and runtime policy.

## Post-seed-1 contract correction

The initial local lifecycle package `e6da6e48feb6cc3b9942af346b6855dac1975d68a7283f61785f0f7645c04c50` and corrected-bytecode package `6430d36beccea114d528921e5eb43d9b08b077f52ba2e045330d2e9aab802d36` mistakenly implemented the retention list as Tiers 0–6, despite the chosen option, scored denominator, and public routing contract specifying Tiers 1–6. Seed 1 then produced a formal receipt SHA-256 `d71509b1213d5d2dc38938c64713e1f3e522444da3b4e7b7e663867efa486276`: Tiers 1–6 were byte-identical, Tier 7 improved from 2/100 to 54/100, and the candidate improved from 59.5% to 64.7%, but the lifecycle stopped solely because non-ranking Tier 0's width-256 cases used the declared bit-length route.

Before any checked answer, randomized control, determinism result, later seed, or sandbox result was read, the retention list was corrected to the intended scored Tiers 1–6. Model bytes, specialist bytes, five seeds, accuracy gates, randomized-weight gate, runtime gates, and zero fallback above 256 bits remain unchanged. The exact seed-1 receipt is reused by hash rather than recomputed; this amendment is an orchestration-contract correction, not post-result model selection.
