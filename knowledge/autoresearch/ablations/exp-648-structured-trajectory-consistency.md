---
type: Experiment Preregistration
title: EXP-648 structured natural-trajectory consistency
description: Matched K=8 endpoint-consistency screen on natural width-256 failure and success windows from the frozen cellular source.
status: root161-archived-root162-v2-scaffold-reconstructed-awaiting-sole-file-authorization
tags: [autoresearch, tier7, trajectory-consistency, endpoint-loss, matched-ab, exact-rollout]
timestamp: "2026-07-20T10:20:00+05:30"
last_updated: "2026-07-22T00:30:00+05:30"
---

# EXP-648 structured natural-trajectory consistency

## Status and falsifiable question

Original historical preregistration status: the Judge selected EXP-648 as the next hypothesis with the best expected information value per CPU and wall time, before the runner, corpus, or readiness evidence existed. That sentence is preserved as history, not current lifecycle state. The corpus is now consumed-complete and immutable; roots `20261159`/`20261160` are historical denied attempts; root `20261161` is consumed historical readiness evidence whose ten tracked files are byte-archived; root `20261162` has a reconstructed v2 authorization source/scaffold but remains reserved, unopened, and unauthorized; and screen/confirmation remain unopened and prohibited.

The falsifiable question is whether a fixed `0.1` endpoint-consistency term over eight natural autonomous transitions improves width-256 exact rollout by at least three percentage points over an otherwise identical `0.0` control, moves the mean first error no earlier, and retains width-64 and width-128 exact rollout within one point.

Candidate 005 remains unchanged at 59.5% checked-public and 59.3% independent overall accuracy. It has no learned path above 128 bits, so this is not a Tier-7 or candidate claim. EXP-649 remains planned/formal-blocked after its full input-only calibration produced `runtime-gate-miss`; its unopened formal root `20261149` is permanently locked for that frozen formulation. Its breadth was never tested.

## Evidence basis and transfer boundary

The dated [source note](../research/2026-07-20-structured-trajectory-consistency.md) records the Playwright-verified primary pages, local PDF hashes, author claims, and project limitations. DRM motivates multi-step trajectory supervision, while LoopFormer motivates shortcut consistency across recursive computation. EXP-648 borrows only natural-trajectory K-step endpoint consistency. It does not use synthetic corruption, reverse noise, time or step conditioning, variable compute, adaptive halting, or either source architecture.

The local diagnosis is more important than analogy. EXP-643 observes 99.70–99.75% width-256 transition accuracy but only 22%/32% rollout, while EXP-610's arbitrary one-/two-bit corruption regresses width-128 rollout from 87.5% to 75.0%. Accordingly, EXP-648 injects no state corruption. It begins from exact prefixes already encountered in natural source trajectories and reports both recovery and newly introduced drift.

## Frozen source and seed isolation

Both arms independently load the same raw hidden-92, two-cycle, 99,824-parameter checkpoint at `.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors`, SHA-256 `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`. The two initial parameter tensors must have the same hash before the first update.

The table preserves each root's original role and gives its current lifecycle truthfully. Roots `20261153`–`20261158` remain unopened; the earlier resource and corpus roots do not.

| Role | Frozen root | Opening rule |
|---|---:|---|
| Historical resource calibration only | `20261151` | Consumed historical non-scientific calibration; not current launch evidence |
| Corpus construction | `20261152` | Consumed-complete immutable generated training-split corpus |
| Screen training/order | `20261153` | Only after corpus, code, config, tests, hashes, and measured budgets are frozen and Judge-approved |
| Screen paired evaluation | `20261154` | Same screen launch gate as `20261153` |
| Confirmation 1 training/order | `20261155` | Only after a complete screen pass and separate Judge authorization |
| Confirmation 1 paired evaluation | `20261156` | Same confirmation authorization |
| Confirmation 2 training/order | `20261157` | Only after a complete screen pass and separate Judge authorization |
| Confirmation 2 paired evaluation | `20261158` | Same confirmation authorization |
| Rejected readiness | `20261159` | Consumed, superseded, historical, and forbidden for launch/rates/cache reuse |
| Failed readiness | `20261160` | Consumed-incomplete, permanently denied, historical |
| Completed readiness | `20261161` | Consumed-complete, sealed, non-retryable, historical; cannot bind repaired bytes |
| Post-repair readiness | `20261162` | V2 source/scaffold reconstructed; reserved, unopened, unauthorized; requires a separate sole-file authorization child |
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
- the complete ordered predicted-state trajectory of every twenty-case repeat has the same independently recomputable SHA-256 digest; endpoint equality, accuracy, and first-error summaries are insufficient; and
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

The current root-`20261162` cumulative planning ceilings are independently rounded to 30,000 wall seconds and 120,000 aggregate CPU seconds. They are planning-only runaway protection derived from measured work and historical charge. They are not targets, scientific gates, submission-runtime limits, or permission to reduce fidelity.

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

## Root 20261161 full-fidelity readiness calibration

Root `20261161` is now `consumed-complete-non-scientific-sealed-non-retryable`. The immutable receipt SHA-256 is `1da93f456a54bfb61a36b4085cfb70b517953c4153d7378190237c872ff4167a`. Execution used one fresh full-fidelity warmup and three fresh full-fidelity repeats, with no root-`20261160` cache reuse or migration and no fidelity reduction. The warmup, repeat 1, repeat 2, repeat 3, and main resource-only MLflow runs are all FINISHED: `b49db1ca755e46e6b7b18e9e5c1fb859`, `8f887966943d44a3a4bebea6ad5dddb9`, `cea11ae498ef4243ac222aa732e8ed06`, `f89f15cf84334df9acdc41ca6625b25d`, and `c27161cb0f2a46228b94b47362769fb1`.

The authoritative canonical outer boundary is observed launch-to-pre-outer-seal evidence: `4770.442001834999` monotonic-wall seconds, `18657.139380582998` process user-plus-system aggregate-CPU seconds, and `2727993344` peak-RSS bytes. Carried charge is applied exactly once at `14302.253947323996` wall / `57721.035879589996` aggregate CPU, producing cumulative `19072.695949158995` wall / `76378.175260173` aggregate CPU, below the independent `25000` / `100000` ceilings. The separate `4727.1`-second tool trace is coarse, noncanonical, and clock-incomparable; it is never substituted for or averaged with the canonical boundary and cannot support a tail inference.

Resolved screen wall projection / soft budget / hard ceiling are `9385.25759757471` / `11748.364904834792` / `17622.547357252188` seconds. Aggregate-CPU projection / soft budget / hard ceiling are `36891.35438058306` / `46131.553842158` / `69197.330763237` seconds, with peak-RSS guard `3409991680` bytes. These are readiness resource evidence only, not scientific results or screen authorization.

The project verifier succeeded. A later `KeyError` came from an incorrect external assertion path and is not a failed verifier or a passing assertion; the correct field is `receipt['boundary']['scientific_accuracy_or_selection_reported']`, whose value is `false`. Both completion audits return `[APPROVE, APPROVE]`, and the Judge decision is `approved-complete-non-scientific-readiness-calibration`. Candidate 005 remains unchanged. No `judge-authorization.json` exists, so screen, scientific evaluation, accuracy, selection, benchmark, promotion, and Goal completion remain blocked.

Post-publication regression provenance is separate from execution provenance. Approved base `87384f92ab1c053c7791e8afd69a07936d5a7402`, immutable scaffold provenance SHA-256 `593d18523ff2888c1d57d8626d81f4ebb07a234427573efa3afc226e98e2c5a6`, and execution-bound focused-test SHA-256 `f66fb2c7e510171bb92a5cab20b842b9b08d9dd023fc44f37ee1e8b74ef08ef5` remain unchanged. Renewed Judge approval expands registration to exactly 17 paths by adding only the lifecycle-regression test, whose stabilized repaired SHA-256 is `967fdc7d4ce5e9b8deaf9d5ea0d4b0cd74d8b4729b66c75b5eadf461626cfb50`; `screen_readiness_binding` must resolve that separate hash from the committed registration base. The pre-repair full suite stopped before commit at 257 passed / two stale lifecycle failures, and the exact three repaired regressions passed. The first focused-file run then stopped before commit at 87 passed / four approved-base fixture failures; the second bounded repair changed only that test fixture. The expanded targeted regression/tamper set then passed 8/8, the focused file passed exactly 91/91, the full suite passed exactly 259/259, and the project validator passed.

## Scientific scaffold repair — 2026-07-21

A renewed launch audit found three scientific-integrity gaps in the unexecuted screen path. This repair changes no corpus bytes, source checkpoint, seeds already assigned to scientific phases, loss, coefficient, update count, evaluation count, gate, architecture, or evaluation fidelity.

Exact-repeat determinism now hashes the complete ordered prediction trajectory, including every intermediate state, and independently recomputes that digest during evaluation-cache validation. Endpoint hashes remain reported but cannot satisfy the repeat gate. A regression holds endpoint predictions, endpoint hash, endpoint accuracy, and first-error summaries equal while changing an intermediate state and proves that the repeat fails.

Matched-work integrity is now independent of paired data order. Each arm accumulates its own ordered event ledger, category counts, total declared work, and SHA-256 digest for the clean and autonomous training branches, endpoint computation, backward calls, gradient clipping, optimizer and scheduler steps, checkpoint write/reload/hash/receipt serialization, transition and rollout evaluation cases and steps, and evaluation-result hash/serialization work. `matched_work_valid` is true only when all declared categories were observed and the two independently accumulated count-and-digest receipts are exactly equal; `paired_data_order_valid` remains a separate check.

R3 additionally cross-binds every declared count to the actual operation ledger, so identical undercounting in both arms cannot pass equality. Fresh evaluator calls and validated cache reuse are alternative truthful paths: cache hits record zero evaluator calls and bind per-arm cached result/work receipts, while fresh calls bind their own result hashes and serializations. Scheduler and checkpoint events are recorded only after the corresponding operation and durable receipt persistence.

Every scientific attempt now writes `control.safetensors` and `challenger.safetensors` under its exclusive immutable attempt directory. A retry cannot reuse or overwrite another attempt's paths. Completion seals include both checkpoint files and verify their bytes against the attempt-bound checkpoint receipts. Failure evidence records whether each preserved checkpoint exists and whether its observed byte hash matches its receipt.

Failure inventory now hashes every regular checkpoint-directory artifact independently. Each artifact is explicitly `receipt-bound`, `unbound-no-receipt`, `receipt-hash-mismatch`, or ambiguous; no empty receipt set can pass vacuously. Completion verification also requires pointer/seal/attempt-state identity, terminal `sealed`/`complete`/`gates_computed` state, and consistent authorization across the seal and all applicable evidence.

Because the live runner, config, evaluator, scientific protocol, resource protocol, and declared workload bindings changed, root `20261161` remains immutable historical evidence but cannot authorize the repaired screen and none of its caches may be reused. The old authorization source and readiness scaffold are historical and cannot bind repaired bytes. Root `20261162` is reserved as the sole post-repair readiness-calibration root; its v2 source/scaffold is reconstructed, but the root remains unopened and unauthorized. The binding next transition is: publish this preservation/reconstruction base -> create and publish a sole-file root-`20261162` authorization child -> run the fresh full-fidelity calibration -> obtain completion audits/Judge -> obtain a separate screen Judge. Screen roots `20261153`/`20261154`, every model update and evaluation RNG, output/checkpoint directory, evaluator, MLflow run, benchmark, promotion, and Goal-completion surface remain blocked.

## Root 20261161 preservation and root 20261162 reconstruction — 2026-07-22

The preservation/reconstruction package performs no calibration, model, RNG, evaluator, MLflow, screen, or benchmark action. It moves the ten tracked root-`20261161` readiness files byte-identically from the generic live namespace to `screen/superseded/root-20261161/`. Canonical manifests cover every archive file except `SHA256SUMS` itself, all 218 in-place scratch files (`24,573,993` bytes), and all 27 in-place artifacts (`252,562` bytes) from the five recorded FINISHED MLflow runs. Scratch and MLflow bytes are not copied, queried, or made reusable. A dedicated historical verifier authenticates the archived v1 authorization against original path `screen/readiness-calibration-authorization.json`, authorization commit `16d903301f50e54e3f6d9ab8cb6435e61f0d648f`, and sole parent/base `87384f92ab1c053c7791e8afd69a07936d5a7402`; it is non-launching and cannot authorize root `20261162`.

The immutable old scaffold remains at SHA-256 `593d18523ff2888c1d57d8626d81f4ebb07a234427573efa3afc226e98e2c5a6` as pre-R3 v1 history. New canonical v2 source and provenance live under `readiness-scaffold/root-20261162/`. They bind exact approved-base Git blobs for the repaired runner, focused test, config, frozen calibration runner/executed config/environment, preregistration, corpus amendment, root-`20261161` classification, both executable dependencies, and all eight frozen corpus/evidence files. They separately name the historical calibration-segment evaluator and the current R3 screen evaluator. A later authorization child may add only four dynamic fields: root, approved-base commit, authorization-source blob SHA-256, and scaffold-provenance blob SHA-256. No live authorization exists in this package.

One `exp648-screen-execution-schedule-v1` drives the v3 workload payload, calibration infrastructure loops, rate denominators, projection, first-quarter estimate, and screen matched-work completion check. It requires 216 scientific-attempt ledger writes plus three outer launch-ledger writes; seven cache write/validate units; two checkpoint save/reload verifications; 29 physical atomic evidence writes; and 54 scientific-ledger writes before the first-quarter guard. Per arm it requires exactly 110,298 declared work units in 1,630 ordered events: 1,600 clean forwards, 1,600 autonomous forwards, 200 endpoint losses, 1,600 clean backwards, 200 endpoint backwards, 200 gradient clips, optimizer and scheduler steps, one each checkpoint write/reload/hash/receipt, 4,000 transition cases, 640 rollout cases, 99,840 rollout steps, and seven result hashes/serializations. A fully fresh path has 423 actual-operation events; with `c` validated screen cache units it has `423+c`. The readiness calibration itself must execute one fresh warmup and three fresh complete representative segments, with both cross-root and same-root segment-cache reuse disabled.

Those displayed totals are outputs, not independent control literals. The runner derives quarter updates and arm-update totals from `updates_per_arm`, ledger totals from named write-reason dictionaries, declared totals/events from training/checkpoint/evaluation dictionaries, actual-operation totals from fresh/cache operation dictionaries, and evidence totals from named physical-write reasons. A mutation regression changes the screen to 240 updates per arm and adds a 100-case width-512 transition unit; without editing any total, the first-quarter scientific ledger/arm-update values become 64/120, total arm updates become 480, the fully fresh actual-event count becomes 506, the workload hash changes, and the cache/profile/projection denominators become eight units.

Readiness cache instrumentation serializes and recovers the complete shape-matched seven-unit target: transition widths 128 and 256 at 2,000 cases each; rollout widths 64, 128, and 256 at 200 cases each; and two width-256 exact-repeat passes at 20 cases each. Every rollout payload contains both arms' full prediction and separately allocated teacher traces. The profile binds 4,640 cases, 200,960 prediction-trace elements, 200,960 teacher-trace elements, the observed canonical bytes for each unit, actual-cache projection and envelope hashes, semantic-fixture and per-field value-width-contract hashes, core/case/work-receipt hashes, and per-arm prediction/teacher trace hashes. Recovery performs canonical parse/round-trip, actual-schema projection, digest, width-contract, trace, receipt, result, and semantic validation. Projection charges the maximum measured complete-set duration directly once; it neither divides this work into a synthetic per-unit rate nor extrapolates from a reduced payload.

A subsequent closure audit found that the first complete-shape profiler still populated width-bearing fields with serialization-unrepresentative `0`/`1`/`3` integers. That profile and every root-`20261161` rate/cache remain historical only. Version 2 deterministically sets `modulus=(1<<width)-159` and `base=1<<(width-1)`, uses base-plus-bounded-offset transition states/operands with exact teacher targets/quotients, and derives complete rollout teacher trajectories from full-width operands; prediction trajectories are independent exact copies. Every nonzero width-bearing value must be at least `width-1` bits, required initial trace zeros are counted exactly, naturally small bit/quotient/index fields are separately bounded, and every field binds value extrema plus bit-length and decimal-digit histograms. The production cache's `unit`, `key_fields`, `timing`, results, receipts, and core hash are serialized as an explicit projection. A clearly labeled strict envelope—not an exact-schema claim—adds raw cases, specification, the width audit, and 16,384 bytes of fixed per-unit metadata padding. That padding explicitly covers the fact that the all-exact fixture's booleans and compact `0.0`/`1.0` metrics can serialize slightly shorter than error-bearing production values. Even so, larger canonical JSON is an empirical conservative workload envelope, not a mathematical CPU-time or RSS upper bound; maximum-repeat measurements and independent safety headroom remain binding for allocator, dictionary-shape, hashing, filesystem, and other runtime variance.

An independent wide-fixture reference established a seven-unit minimum of `20,147,941` canonical bytes (`710,748`, `1,168,748`, `1,320,122`, `3,595,231`, `11,100,815`, `1,126,137`, and `1,126,140` by scheduled unit), compared with the rejected narrow fixture's `5,926,597`. The repaired deterministic local diagnostic measures `31,567,530` envelope bytes: `30,080,731` bytes for the production-schema projection plus `1,486,799` envelope bytes. These are non-scientific local fidelity diagnostics, not fresh calibration rates or authorization. Historical sealed root-`20261161` corpus evidence (`4,135,139` bytes, SHA-256 `d49a042062ae1a06b7d7e5b5914c73411875aed12cd2a704862c3821147127f0`) supports only the observation that real width-256 operands and nonzero traces occupy broad high-bit-length ranges; it is never executable cache, launch, projection-rate, or scientific evidence. Fresh timing separately covers fixture construction/semantic validation, serialization/write, recovery parse/digest/receipt/semantic validation, and the complete set, with wall time, process aggregate CPU, and peak RSS kept distinct.

Root `20261161` contributes its canonical `4770.442001834999` wall / `18657.139380582998` aggregate-CPU seconds exactly once to the prior `14302.253947323996` / `57721.035879589996` carried charge. Root `20261162` therefore starts at `19072.695949158995` wall / `76378.175260173` aggregate CPU. With measured planning inputs `R=(5305.469600509008, 20967.89)` and `E=(73.8932728219952, 135.975487648)`, the binding formula `new_carried + 1.5 * (1.25 * R + E)` yields `29131.291359346378` wall and `115896.932241645` aggregate CPU, independently rounded upward to cumulative ceilings of `30000` and `120000`. RSS remains separate. These ceilings never reduce counts, fidelity, diagnostics, or scientific gates.

Current state is preservation, reconstruction, and resource-profile fidelity repair only: root `20261162` remains reserved, unopened, and unauthorized; no scientific claim changed. The regenerated bindings require a fresh package audit before publication, then a separate sole-file v2 authorization child, fresh full-fidelity calibration, completion audits/Judge, and only then a separate screen Judge.
