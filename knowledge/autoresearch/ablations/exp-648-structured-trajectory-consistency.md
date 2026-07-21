---
type: Experiment Preregistration
title: EXP-648 structured natural-trajectory consistency
description: Matched K=8 endpoint-consistency screen on natural width-256 failure and success windows from the frozen cellular source.
status: approved-recovery-base-pending-commit-and-push
tags: [autoresearch, tier7, trajectory-consistency, endpoint-loss, matched-ab, exact-rollout]
timestamp: "2026-07-20T10:20:00+05:30"
last_updated: "2026-07-20T21:21:35+05:30"
---

# EXP-648 structured natural-trajectory consistency

## Status and falsifiable question

The Judge selected EXP-648 as the next hypothesis with the best expected information value per CPU and wall time. This document freezes a preregistration only. The runner, corpus, calibration, screen, and confirmation do not yet exist and no experiment root has been opened.

The falsifiable question is whether a fixed `0.1` endpoint-consistency term over eight natural autonomous transitions improves width-256 exact rollout by at least three percentage points over an otherwise identical `0.0` control, moves the mean first error no earlier, and retains width-64 and width-128 exact rollout within one point.

Candidate 005 remains unchanged at 59.5% checked-public and 59.3% independent overall accuracy. It has no learned path above 128 bits, so this is not a Tier-7 or candidate claim. EXP-649 remains planned/formal-blocked after its full input-only calibration produced `runtime-gate-miss`; its unopened formal root `20261149` is permanently locked for that frozen formulation. Its breadth was never tested.

## Evidence basis and transfer boundary

The dated [source note](../research/2026-07-20-structured-trajectory-consistency.md) records the Playwright-verified primary pages, local PDF hashes, author claims, and project limitations. DRM motivates multi-step trajectory supervision, while LoopFormer motivates shortcut consistency across recursive computation. EXP-648 borrows only natural-trajectory K-step endpoint consistency. It does not use synthetic corruption, reverse noise, time or step conditioning, variable compute, adaptive halting, or either source architecture.

The local diagnosis is more important than analogy. EXP-643 observes 99.70–99.75% width-256 transition accuracy but only 22%/32% rollout, while EXP-610's arbitrary one-/two-bit corruption regresses width-128 rollout from 87.5% to 75.0%. Accordingly, EXP-648 injects no state corruption. It begins from exact prefixes already encountered in natural source trajectories and reports both recovery and newly introduced drift.

## Frozen source and seed isolation

Both arms independently load the same raw hidden-92, two-cycle, 99,824-parameter checkpoint at `.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors`, SHA-256 `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`. The two initial parameter tensors must have the same hash before the first update.

Every seed below is currently unused and unopened:

| Role | Frozen root | Opening rule |
|---|---:|---|
| Resource calibration only | `20261151` | After the implementation and static-test gate; never scientific evidence |
| Corpus construction | `20261152` | Generated training-split identities only, under the resolved mining ceiling |
| Screen training/order | `20261153` | Only after corpus, code, config, tests, hashes, and measured budgets are frozen and Judge-approved |
| Screen paired evaluation | `20261154` | Same screen launch gate as `20261153` |
| Confirmation 1 training/order | `20261155` | Only after a complete screen pass and separate Judge authorization |
| Confirmation 1 paired evaluation | `20261156` | Same confirmation authorization |
| Confirmation 2 training/order | `20261157` | Only after a complete screen pass and separate Judge authorization |
| Confirmation 2 paired evaluation | `20261158` | Same confirmation authorization |
| Scaffold tests only | `648900001` | Never experiment or evaluation evidence |

Checked/public inputs and answers, every prior independent set, the official private benchmark, and EXP-649 formal root `20261149` are excluded. Tests may use only `648900001`; they may not invoke any calibration, corpus, screen, confirmation, checked, independent, private, or EXP-649 formal root.

## Frozen width-256 corpus

Run the frozen source autonomously on newly generated width-256 training-split identities from corpus root `20261152`. No benchmark identity is eligible. Keep exactly 512 failure windows and 512 successful windows, paired one to one. No state bit is injected, flipped, masked, or otherwise synthetically perturbed.

A failure is eligible only when its zero-based first wrong transition index is at most 248, so a complete `K=8` window can begin at the exact prefix immediately before that first wrong transition. The failure window then follows the source's natural hard-decoded autonomous states for eight transitions.

For every failure, select the first unused successful window with all of these exact matches:

- the same zero-based window-start index;
- the same exact modulus bit length; and
- the same normalized-modulus band, where normalized modulus is `p / 2**p.bit_length()` in `[0.5,1.0)` and the four bands are `[0.500,0.625)`, `[0.625,0.750)`, `[0.750,0.875)`, and `[0.875,1.000)`.

An eligible successful window starts from the exact prefix at that matched index and remains exact for all eight autonomous transitions. Corpus-root generator order determines failure order. For a matching stratum, the first unused eligible success in the same deterministic generator/trajectory order wins. No replacement, rematching, relaxed band, relaxed bit length, shortened window, or reduced count is permitted.

Before training, freeze and hash the identity manifest, complete source trajectory manifest, source checkpoint, exact offline teacher source, and final paired corpus. Exact arithmetic is allowed for the offline teacher and corpus labels only. If 512 matched pairs cannot be mined under the post-calibration resolved corpus-mining ceiling, record `corpus-construction-failure`; do not weaken the count or matching contract.

## Matched arms and sole changed variable

Each phase creates separate arm instances by independently cloning the raw source checkpoint. The arms must have identical parameter counts, initial tensor hashes, frozen windows, batch order, optimizer, schedule, batches, updates, clean teacher-forced branch, autonomous forward/backward work, endpoint-loss computation, checkpoint work, evaluation inputs, and serialization/hashing work.

Each batch contains sixteen windows: eight failures and their eight matched successes. Each window executes:

1. a clean teacher-forced branch over the same eight exact transitions; and
2. a `K=8` autonomous branch whose residue state is hard-thresholded in the forward pass and uses sigmoid straight-through gradients in the backward pass.

For each clean step, the existing transition loss is mean residue-bit BCE plus `0.45` times the worst-position mean BCE plus `0.10` times quotient cross-entropy. `base_clean_loss` is the mean of that loss over the eight exact teacher-forced steps.

Both arms compute the same endpoint residue-bit loss on the eighth autonomous logits against the offline exact eight-step endpoint: mean residue-bit BCE plus `0.45` times worst-position mean BCE, with no quotient term. The total loss is:

`base_clean_loss + lambda_endpoint * endpoint_loss`.

The sole changed variable is `lambda_endpoint`: control `0.0`, challenger `0.1`. There is no coefficient, horizon, corpus, or schedule sweep.

Use AdamW, learning rate `5e-6` to `5e-7` under one cosine schedule, weight decay `2e-5`, gradient clipping at `2.0`, four PyTorch threads, one interop thread, and deterministic algorithms.

## Screen

The screen uses roots `20261153` and `20261154`. Each arm is independently reinitialized from the source and receives exactly 200 matched updates. Evaluate both checkpoints on identical paired inputs:

- 2,000 fresh transitions at width 128 and 2,000 at width 256 per arm;
- 200 complete rollouts at each of widths 64, 128, and 256 per arm; and
- an exact repeat of the same twenty width-256 rollout cases.

The zero-based first-error index is the first wrong transition. A successful 256-transition rollout receives the frozen clipped value `257`. Report each arm's mean clipped first-error index, plus paired recovered cases (`control wrong, challenger exact`) and introduced drift (`control exact, challenger wrong`).

The challenger advances only if every gate passes:

- width-256 exact rollout gain over control is at least `+0.03`;
- challenger mean clipped first-error index is at least the control mean;
- width-64 and width-128 exact rollout deltas are each at least `-0.01`;
- challenger width-256 exact transition accuracy is at least `0.995`;
- challenger invalid-rollout rate is at most `0.01`;
- the twenty-case repeat is bit-identical and deterministic; and
- source, initial-state, parameter, data/order, seed, checkpoint, evaluator, and matched-work integrity checks all pass.

Any miss rejects this exact frozen screen. A pass is intermediate evidence only and requires a separate Judge decision before confirmation. It does not authorize a candidate, Tier-7 promotion, or benchmark evaluation.

## Conditional two-pair confirmation

Confirmation is prohibited unless the complete screen passes and a Judge explicitly authorizes it. It consists of exactly two fresh matched A/B reinitializations from the raw checkpoint, not continuations from the screen or from each other. Pair 1 uses training/evaluation roots `20261155`/`20261156`; pair 2 uses `20261157`/`20261158`. Each arm receives 400 updates.

For each pair, evaluate 2,000 transitions at widths 128 and 256, 200 full rollouts at widths 64 and 128, 500 width-256 full rollouts, and an exact repeat of twenty width-256 cases. Each pair must independently achieve:

- at least `+0.03` challenger-minus-control width-256 rollout accuracy;
- nonnegative movement in mean clipped first-error index;
- width-64 and width-128 retention deltas each at least `-0.01`;
- challenger width-256 transition accuracy at least `0.995`;
- challenger invalid-rollout rate at most `0.01`; and
- exact determinism and all integrity checks.

Pool the 1,000 paired width-256 cases across both confirmations. Let `b` be recovered cases and `c` introduced-drift cases. With one-sided `alpha=0.05`, compute the exact Clopper-Pearson lower bound for `b/(b+c)`: zero when `b=0`, otherwise `BetaQuantile(0.05, b, c+1)`. If `b+c=0`, define the net exact-gain lower bound as zero. Otherwise compute:

`(2 * p_lower - 1) * (b + c) / 1000`.

That lower bound must be strictly positive. Any per-pair or pooled miss rejects the frozen confirmation. A complete pass supports only a separately preregistered candidate/Tier-7 gate; it never promotes directly.

## Binding dynamic resource protocol

Resource calibration is a separate, non-scientific workload on root `20261151`. After one complete warmup, run three complete timed repeats. Every warmup and repeat must measure the same representative work: mine 64 matched pairs through the exact width-256 corpus path; include startup/model loading; execute 20 real K=8 updates per arm at batch 16; write each arm checkpoint; evaluate 200 transitions per arm at widths 128 and 256; evaluate twenty complete rollouts per arm at widths 64, 128, and 256; and perform the declared hashing and serialization.

Record external monotonic wall seconds, process user-plus-system aggregate CPU seconds, and peak RSS separately. Use four PyTorch threads and one interop thread. For every projected component, use the maximum complete-repeat rate and the slower arm. Wall and CPU are different resources and may not substitute for each other.

Project the screen and each confirmation pair separately from measured setup plus every declared corpus or corpus-integrity, training, checkpoint, transition, rollout, repeat, hash, and serialization unit. The screen includes construction of all 1,024 frozen windows. Confirmations reuse only the hash-valid frozen corpus and pay its full integrity-validation work; they do not silently remine or change it.

For wall and aggregate CPU independently:

- explicit measured setup headroom is twice the maximum complete-repeat startup/checkpoint/hash/serialization component;
- soft budget is `1.25 * projected component work + explicit measured setup headroom`; and
- hard runaway ceiling is `1.5 * soft budget`.

The initial calibration planning ceilings are 14,400 wall seconds and 57,600 aggregate CPU seconds. They are generous planning-only runaway protection relative to EXP-647's full 5,141.6-wall/20,465.9-aggregate-CPU-second run. They are not targets, scientific gates, submission-runtime limits, or permission to reduce fidelity.

Recalibrate after any runner, config, checkpoint, environment, thread, or declared-workload hash change, or if first-quarter throughput is more than 20% slower than projection. Cache and preserve complete unit evidence and checkpoints under keys containing checkpoint, evaluator, dataset, config, seed, environment, runner, scientific-protocol, and resource-protocol hashes. Never reduce windows, updates, cases, repeats, diagnostics, or evaluation fidelity to fit a budget. Evaluation-only overrun preserves the checkpoint and records missing evidence; it is not scientific rejection.

The screen roots remain unopened until corpus, runner, tests, config, scientific/resource protocol, environment, and measured budgets are frozen and a post-implementation Judge approves launch. Confirmation roots remain unopened until a screen pass and another Judge authorization.

## Compliance and decision boundary

EXP-648 changes training loss only. Inference architecture, semantics, and the 99,824-parameter count remain unchanged. Exact arithmetic is confined to offline corpus construction, teachers, and evaluation. No benchmark answers select data, hyperparameters, checkpoints, or gates, and no exact modular solver or output repair enters inference.

EXP-650 remains blocked until a separate proper-library readiness, CPU, and submission gate is approved; this preregistration does not install a dependency. EXP-651 is deferred because the prior EXP-617 gain and control geometry are too weak. EXP-648 is the only selected runnable follow-up, but remains blocked from execution pending implementation, corpus/resource preparation, frozen hashes, and Judge review.

## Post-corpus readiness amendment — 2026-07-20

The accepted corpus is an immutable historical artifact from execution HEAD `9f222687...` and runner `83181524...`, preserved by corpus commit `8d960fa...`; the current verifier checks those exact old bytes and their recorded limitations without claiming that the live runner executed them. The scientific counts, losses, optimizer, gates, fidelity, roots, and source checkpoint above are unchanged.

Before any scientific screen action, one canonical Judge authorization must bind the exact launch commit; live runner, tests, config, source and dependency bytes; environment and threads; teacher/evaluator/protocol/workload hashes; all frozen corpus and evidence hashes plus the provenance amendment; and a fresh post-hardening calibration receipt. A missing or mismatched authorization fails before a screen root, model, RNG, output directory, checkpoint, evaluator, or MLflow run is opened.

The screen implementation derives a complete deterministic pair permutation from training root `20261153` and the frozen corpus hash, uses that order identically for both arms, and performs a full source-checkpoint replay of all 736 frozen trajectories before training. It interleaves the two arms, atomically serializes exact numeric telemetry at one-quarter of the 400 total arm-updates, and enforces wall time, process user-plus-system aggregate CPU, and peak RSS as separate resources. Both checkpoints are atomically saved, hashed, reloaded, and state-hash verified before the first evaluation input is generated. Each paired arm/width/mode evaluation unit is recoverable only under a composite key binding checkpoints, evaluator, input bytes, config, roots, environment, runner, dependencies, scientific/resource protocols, workload, and corpus evidence; reused work is charged to both wall and CPU budgets. Evaluation-only overrun preserves completed checkpoints and cache units, writes an incomplete receipt with no gates, and never becomes a scientific rejection.

The historical calibration root `20261151` predates this changed runner and workload and is not valid launch evidence. At this amendment stage no new root or authorization manifest was assigned; that phase was later superseded by the explicit approval recorded below. A complete warmup plus three complete full-fidelity timed repeats must still resolve screen wall/CPU/RSS guards with safety headroom and bind every readiness hash. Only a later launch Judge may authorize roots `20261153`/`20261154`. Confirmation and all checked, independent, official/private, candidate-promotion, and Goal-completion surfaces remain prohibited.

The initial hardening audit rejected an earlier scaffold that described but could not execute this recalibration contract. That repair revision added the separately authorized CLI calibration executor, substantive receipt and environment verification, authenticated committed manifests, semantic recovery validation, per-update attempt state, exact MLflow comparison, and attempt-scoped completion sealing. At that stage it was explicitly **not authorization-ready** and assigned no calibration root; the later closure and sequential-gate outcome below supersedes that status.

Training checkpoint resume is not implemented or authorized. Every retry deterministically retrains both arms from the raw source checkpoint under the same committed authorization and root-derived order. Only after the newly trained checkpoint hashes exactly match may complete evaluation-cache units with matching composite provenance be reused and charged. Existing incomplete and sealed attempt directories are immutable evidence; a retry receives a monotonic suffix and never overwrites them.

Both authorization phases use a non-self-referential two-commit lifecycle. First commit the complete readiness or calibration-evidence base. The Judge then adds only the canonical authorization manifest in its single child commit. Launch requires that clean child commit at upstream HEAD and verifies the base-to-child diff contains exactly the manifest. Later descendant evidence verifies the unchanged historical authorization blob against its recorded base without pretending the old authorization commit is current HEAD. Any code, config, evidence, or extra-file change in the authorization child fails closed.

The renewed evidence-audit repairs keep current verifier observations outside the immutable corpus package in `readiness-scaffold/provenance.json`. Authorization reconstruction reads only approved-base Git blobs and base-recorded environment evidence, so later descendant edits cannot contaminate historical verification. Calibration accounting preserves the full superseded-evidence list and decomposes historical accepted work, prior superseded work, current process work measured after every added readiness unit, and valid reused charges. Complete warmup/repeat totals add source replay, MLflow probe, and readiness infrastructure exactly once; launcher/setup is charged once outside unit totals, and RSS is the maximum over all complete peaks. Core and receipt copies of every release-critical field must be exactly equal. These were scaffold controls awaiting the later approval below; full-fidelity calibration and a later launch Judge remain mandatory.

The inner calibration receipt now labels its resource snapshot as prepublication and provisional. The subsequent Judge gate returned `REQUIRED_REPAIR` because the first outer receipt overstated its resource boundary as authoritative/end-to-end even though function return, CLI output, stack unwind, interpreter teardown, and process exit necessarily occur after its final sample. The repaired v2 outer receipt performs every feasible non-self-referential inner hash and lifecycle check before sampling, then reports only observed launch-to-pre-outer-seal monotonic wall seconds, process user-plus-system aggregate CPU seconds, and peak RSS. Wall and CPU retain separate raw samples, recomputation, arithmetic, and negative swap/combination tests. The receipt explicitly enumerates every remaining payload, atomic-write, return, stdout, unwind, teardown, and exit operation and excludes them from the observed totals; no sealing-tail proxy is folded into those observations.

Screen launch orchestration likewise places root reauthorization and construction inside one preservation boundary, transitions the separate launch ledger on constructor success, and terminates it with reciprocal scientific-attempt linkage on every failure or sealed completion. No terminal launch path may remain constructor-pending. At that point the boundary repair was scaffold evidence only and remained blocked pending the renewed audit whose approval is recorded below.

## Readiness implementation approval and calibration authorization — 2026-07-20

Both closure audits under `T-EXP648-SCREEN-READINESS` returned `APPROVE`, and sequential Judge gate `T648` approved the readiness implementation. This is control and phase-transition evidence, not an observed calibration result. The implementation status is now `approved-base-pending-commit-and-push`; no base commit or push is performed by this bookkeeping pass.

The exact new calibration-only root is `20261159`, with status `authorized-pending-authentication-child`. It is not consumed. The binding order is: commit and push the approved base; add only the canonical calibration-authorization manifest in its sole-file child commit, push it, and verify a clean worktree at upstream HEAD; then execute exactly one complete warmup and three complete full-fidelity timed repeats. Initial runaway guards are `14,400` monotonic wall seconds and `57,600` process user-plus-system aggregate CPU seconds. They are independent safety ceilings, not targets or grounds to reduce fidelity.

Cache reuse is valid only for complete units whose composite provenance exactly matches checkpoint, evaluator, dataset/input bytes, config, root/seed, environment, runner, dependency, scientific-protocol, resource-protocol, workload, and frozen-corpus evidence hashes. Partial, mismatched, or unverifiable units are not reusable; every valid reused unit remains charged to both wall and aggregate-CPU accounting. Screen, confirmation, checked, independent, official/private benchmark, candidate/Tier-7 promotion, Goal-completion, and EXP-649 formal surfaces remain prohibited. Candidate 005 remains the observed `59.5%` checked-public / `59.3%` independent incumbent, and `completion_eligible` remains false.

## Root 20261159 chronology rejection and repaired-base contract — 2026-07-20

The preceding authorization was exercised. Root `20261159` is now consumed and preserved as `consumed-complete-components-but-chronology-nonconforming`. Its warmup and three repeats contain complete non-scientific base, source-replay, MLflow-probe, and readiness-infrastructure components, but the additions were measured after the frozen calibration runner had completed its base warmup and base repeats. The record therefore does not prove the required synchronous complete-unit chronology and is superseded. It is forbidden for screen launch, projection-rate selection, or cache reuse. It produced no model update, scientific accuracy, selection, candidate promotion, or completion evidence.

The preserved authorization, receipt, core, and outer-resource hashes are respectively `d3aa35008f646edf5171771056cea523e60d0415083d58970cf5efadc7492a96`, `cd27fabbd47683677c9b6465db00444445569ce32a0b8e8a595a42d11da82981`, `dae882cefc44cdff89ca4b3a4c18a6ebd55e1415a62fb9bd323e1f1b65f0dd6f`, and `9eeb228450ba8e684adbe34bb95ad057036ffdedc643fe3942a0aaf846cf1ab0`. Canonical observed launch-to-pre-outer-seal resource evidence is `3107.4547687239974` monotonic-wall seconds, `12442.915271225` process user-plus-system aggregate-CPU seconds, and `2675752960` peak-RSS bytes. The separately observed `3166.1`-second shell-wrapper duration has only `0.1`-second precision, was not persisted by the readiness process, spans command launch through process exit, and is not substituted for the narrower canonical boundary. Both facts retain the explicit post-sample-tail limitation.

The repaired executable contract never invokes the frozen runner's batch `run()` method. For each unit it must synchronously execute and guard `base -> actual source replay -> MLflow probe -> readiness infrastructure -> complete validation -> atomic augmented record -> chronology seal`. The warmup seal must precede repeat 1's start; each later repeat start authenticates the predecessor seal. A complete augmented repeat cache may be written only after validation and sealing and must bind the root, authorization, scientific and calibration runners, config, checkpoint, environment, evaluator, dependency manifest, scientific/resource protocols, workload, frozen corpus/provenance, and chronology contract. Recovery must first perform a fresh complete augmented warmup. Base-only caches from the frozen runner are never loaded or saved, and incomplete or tampered augmented caches fail closed.

At that historical repaired-base boundary, the completed historical and superseded charge was `9007.853947323996` wall / `36721.035879589996` aggregate-CPU seconds, and the then-operative independent ceilings were `14400` wall / `57600` aggregate CPU. The later root-`20261160` failure recovery below supersedes those current-accounting and ceiling statements. Root `20261159` timing may inform conservative planning context only. Projection rates must come exclusively from three complete chronology-conforming augmented repeats at a new root, and the resulting screen budgets must retain measured throughput, explicit setup headroom, separate wall/CPU/RSS treatment, and full fidelity.

Root `20261160` was then reserved as the distinct next calibration-root candidate and was unopened, unconsumed, and unauthorized at that boundary. Its later failure and permanent denial are the superseding current status recorded below. Screen, confirmation, checked, independent, official/private benchmark, candidate/Tier-7 promotion, Goal-completion, and EXP-649 formal surfaces remain prohibited. Candidate 005 remains unchanged at observed `59.5%` checked-public / `59.3%` independent accuracy with `completion_eligible: false`.

## Root 20261160 failure recovery (supersedes its prior candidate status)

Root `20261160` opened under authorization SHA-256 `9e1f63a9fe3c6613f8c02dea523147901da2aea862a3dec58d53c0cb7a054984` and failed during repeat 3 after a sealed warmup and sealed repeats 1–2. It is now `consumed-incomplete-non-scientific-permanently-denied`: it cannot launch, supply final or projection rates, support a scientific conclusion, or be reopened. No model update, screen/confirmation/benchmark surface, main scientific MLflow run, accuracy metric, or selection metric occurred. Repeat 3 is not complete.

The full attempt inventory is 160 files and 27,787,830 bytes. Its 21,780-byte canonical GNU-sha256sum-style manifest hashes to `44e1d930823c319f2fe4ecd98af46014af5594b54343d513fc666ecdc3336f72`; the equivalent compact sorted JSON mapping plus LF hashes to `e0495b330014557d4764a560c58591d2964d21ac9d40b66a7a33b8df093f4284`. Three and only three authorization-bound MLflow probes are FINISHED: warmup `a9f49728d8184a438beeb8751769e9cd`, repeat 1 `2cebd8d67f6d408da76cb330fdb10108`, and repeat 2 `e14df2c4633c41449d623b0df6449a56`. There is no main or scientific run.

Repeat-cache hashes `98a69670a229c15de7193c5878571fcc28a6d013e9b3ae23a7ac52be92624d80` and `8e8115d2b41974c20adfad734a847ec2ac64e07f5636325301b8793ffb1adc68` are retained only as historical planning evidence with their exact old root, authorization, checkpoint, evaluator, dataset, config, seed, environment, runner, dependency, scientific/resource protocols, workload, frozen-corpus, provenance-amendment, chronology, source-seal, and source-record bindings. They are never executable, reusable, weakened, or migrated.

Accounting carries the prior `9007.853947323996` wall / `36721.035879589996` process user-plus-system CPU seconds plus conservative failed-attempt allowances of `5294.4` wall / `21000` CPU seconds exactly once, for cumulative `14302.253947323995` wall / `57721.035879589996` CPU seconds. The allowances are not exact internal observations. Independently rounded cumulative runaway ceilings are `25000` wall and `100000` CPU seconds, derived from prior + allowance + `1.5 * (1.25 * R + E)` with `R=5305.469600509008` wall / `20967.89` CPU and `E=73.8932728219952` wall / `135.975487648` CPU. They are never per-attempt targets, scientific gates, or fidelity caps.

The repaired runner atomically emits an attempt-scoped catchable-`Exception` boundary receipt before re-raise, with separate monotonic wall, process user-plus-system CPU, peak RSS, exact warmup/repeat plus base/replay/probe/infrastructure/validation/record/seal/cache stage, the guard's exact last recorded check, root/auth/attempt context, exception identity, last completed chronology seal, and complete artifact-boundary manifest. It then atomically publishes terminal `phase: failed`, `complete: false` attempt state bound to the receipt hash; any receipt/state mismatch fails closed, and the same authorization cannot relaunch before config/root/data/model/MLflow access. Receipt serialization/write, terminal-state write, re-raise, stdout/stderr, unwind, teardown, and exit are explicitly outside the resource sample; no post-exit claim is made, and either publication failure fails closed while preserving original and publication exceptions. The historical root `20261160` failure did not persist this receipt, so its exact exception-boundary observations cannot be reconstructed.

Root `20261161` was preregistered as a negative-use-audited candidate and remains unopened and unconsumed. The two recovery closure audits returned `[APPROVE, APPROVE]`, and final sequential Judge T648 returned `APPROVE`. Its status is now `authorized-pending-authentication-child`: first commit and push this approved recovery base, then replace only the unchanged live root-`20261160` authorization with canonical root-`20261161` authorization bytes bound to that base, commit and push that sole-file child, prove clean upstream equality, and stop before execution. No scientific surface is authorized. Any separate later calibration execution requires a fresh augmented warmup and all three fresh full-fidelity repeats; no old cache execution or migration is permitted.
