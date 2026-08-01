# T001 — Agentic Trace Audit

## Scope and evidence boundary

Observed evidence comes from the complete local archive:

`C:\Users\Ishan\.codex\sessions\2026\07\11\rollout-2026-07-11T22-47-57-019f522f-936f-77f1-a4b6-9ed26785b72b.jsonl`

The archive contains 45,416 JSONL records (74,655,075 bytes) spanning 2026-07-11 17:18 UTC through 2026-07-21 14:18 UTC. It ends during an interrupted turn, so the final EXP-648 state is not a completed-result record. Tool and agent counts cover the whole archived run, including subagent activity represented in the parent trace; they are not all root-PM actions.

## Chronological run map

1. **Scaffold and first POC (July 11).** The run created the private repository and a hybrid OKF/RO-Crate/Croissant structure, implemented a learned Horner/TCN POC, and initially reported custom rollout evidence. When asked directly, it correctly disclosed that the POC had not yet run the official 1,100-case evaluation.
2. **Submission candidates 001–004 (July 11–12).** The run built compliant three-file candidates and advanced exact overall independent accuracy from about 21.3% (Candidate 001) to 31.0% (Candidate 002), about 40.4% (Candidate 003, reported later), and 49.5% (Candidate 004). It used independent seeds, checked/public separation, randomized-weight controls, deterministic/container checks, and durable experiment records.
3. **Tier-6 search (July 12–13).** A long sequence of preregistered interventions (curricula, state recovery, straight-through state, corruption, widening, local correction, extra recurrence, CVaR, LoRA, beam/routing, and independent lineages) produced many honest rejections. EXP-635 and later EXP-642 found useful independent-lineage/commuted-orientation complementarity.
4. **Candidate 005 (by July 17).** The frozen tier-routed learned Horner ensemble reached 59.3% exact overall on one independent 1,100-case seed and 59.5% on the checked set, with Tiers 1–6 at 100/99/97/97/94/98% on the reported independent evaluation. Tiers 7–10 remained effectively unsolved because widths above 128 bits used a zero fallback.
5. **Research-direction expansion (July 17–19).** New Super Transformer/neural-computer documents expanded the backlog, paper library, and mechanism ledger. The run correctly distinguished compliant learned mechanisms from prohibited compiled or hand-constructed solvers, but the active portfolio grew faster than scientific throughput.
6. **EXP-647 persistence/recovery (July 19).** Significant effort moved from model science into background-process persistence, CI/LFS repair, task-scheduler probes, canonicalization, and recovery of a negative or interrupted run.
7. **EXP-649 runtime calibration (July 19–20).** A stochastic-trajectory/PTRM-inspired path underwent primary-source verification, multiple audits, scaffold hardening, and a full-fidelity calibration. It was blocked by the runtime gate before any accuracy experiment; Candidate 005 remained unchanged.
8. **EXP-648 lifecycle hardening (July 20–21).** EXP-648 was selected around 2026-07-20 04:38 UTC. By the archive end about 33 hours 40 minutes later, the trace had performed extensive provenance, authorization, fixture-size, trajectory-hash, archive, calibration, and publication repairs. A fresh non-scientific calibration completed, but no scientific screen result existed when the final turn was interrupted.

## Quantified trace facts

- 59 `task_started` events, 43 `task_complete` events, and 8 `turn_aborted` events; eight starts have no terminal event in this archive.
- 34 context compactions and 33 goal-state updates.
- Final goal telemetry before pause: 24,387,556 goal-accounted tokens and 336,855 seconds (about 93.6 hours).
- Model token telemetry at the archive end: 1,516,881,690 cumulative tokens, including 1,490,035,200 cached input tokens, 2,858,497 output tokens, and 703,117 reasoning-output tokens. This telemetry is not the same accounting unit as the goal counter, but it exposes the cost of repeatedly carrying a very large context.
- 2,362 assistant commentary messages and 36 final-answer messages.
- 4,015 orchestration `exec` calls, including 3,315 nested shell commands, 623 patch calls, and 110 plan updates.
- 4,425 waits on yielded execution cells across 437 distinct cells; 3,539 outputs still said the script was running. The most-polled individual cells were waited on 197, 166, and 152 times.
- 785 `wait_agent` calls; 545 returned `Wait timed out`.
- 156 spawn calls, 135 observed unique started agent paths, 114 follow-up-task calls, 125 inter-agent messages, 109 agent-list calls, and 8 interrupts.
- 56 started agents had audit/review/gate/closure/verdict terms in their task names; 17 had repair/recovery/hardening/reconcile terms.
- 70 of 135 observed started agent paths were named for EXP-648; 14 were named for EXP-649.
- On July 21 alone the trace records 52 spawns, 324 `wait_agent` calls, 43 follow-up tasks, and 40 agent-list calls.
- At least 361 orchestration calls referenced pytest, 320 referenced `validate_project.py`, 500 referenced `git status`, and 442 referenced `git diff`. These include legitimate gates, but their repetition shows verification was being applied at micro-transition frequency rather than mainly at coherent package boundaries.
- 233 orchestration outputs were explicitly marked `Script failed`. Some are expected negative gates or diagnostic failures, so this is not a count of bugs.

## Strengths worth preserving

1. **Scientific honesty improved after challenge.** The run explicitly corrected the initial benchmark ambiguity, kept public/checked/independent/private claims separate, and did not claim the unavailable private benchmark.
2. **Real incumbent progress.** The run advanced exact overall independent accuracy from roughly 21.3% to 59.3% while preserving a compliant submission artifact and refusing to promote weaker checkpoints.
3. **Strong leakage and compliance discipline.** Independent seeds, frozen gates, randomized-weight controls, deterministic/container tests, and the trained-parameters boundary repeatedly affected decisions.
4. **Durable negative evidence.** Failed experiments were preregistered, preserved, and usually recorded with wall versus aggregate CPU time, hashes, configs, metrics, and limitations.
5. **Audits caught genuine defects.** Examples include public-benchmark ambiguity, CRLF checksum drift, stale Candidate-005 release constants, incomplete trajectory hashing, and resource fixtures that materially understated byte volume.
6. **Later background-run persistence improved.** The run moved long experiments to systemd/Task Scheduler-style persistence and eventually acknowledged the user's request to avoid minute-scale polling.

## Failure modes and their likely causes

### 1. Coordination became the dominant workload

Observed: 135 started agent paths, 545 timed-out agent waits, 56 audit/gate-named agents, and 70 EXP-648-named agents. EXP-648 consumed about 34 hours without producing a scientific accuracy result.

Inference: the run optimized for procedural certainty at every micro-step rather than for scientific information per unit of time. Independent review was useful at genuine risk boundaries, but repeated Scout/Worker/Judge/repair/closure chains recreated the same context and multiplied state transitions.

### 2. Polling polluted context after the problem was known

Observed: 4,425 yielded-exec waits, 3,539 still-running outputs, 785 agent waits, 545 agent timeouts, and 2,362 commentary messages. The user explicitly asked on July 20 to estimate completion and wake later instead of checking every few minutes. July 21 still contained 324 agent waits.

Inference: long processes and agents lacked completion-triggered ownership. The PM repeatedly sampled progress rather than delegating one bounded package and receiving one receipt or scheduling one coarse heartbeat.

### 3. Context carrying was extreme

Observed: 34 compactions and about 1.49 billion cached input tokens in model telemetry. Late calls often carried around 150k cached input tokens for tiny status checks.

Inference: the monolithic session and repetitive narrative summaries made every poll expensive. Durable state existed in files, but execution continued to rely heavily on conversational history instead of compact task prompts and receipts.

### 4. Verification granularity was too fine

Observed: hundreds of pytest, validator, status, and diff invocations; EXP-648 accumulated 282 tests, a 20.15 MB wide fixture, multiple independent audits, and several reconstruction roots before the scientific screen.

Inference: repository correctness gates were repeatedly run after small control-plane edits. This caught real defects but imposed quadratic overhead. The system lacked a tiered gate matrix separating cheap local checks, package checks, phase-boundary checks, and the final oracle.

### 5. The portfolio drifted from the bottleneck

Observed: Candidate 005 solved Tiers 1–6 well and returned zero above 128 bits, yet the run expanded to a 42/44/81-record Super Transformer mechanism inventory and spent long tranches on paper-faithful process before establishing cheap Tier-7 feasibility.

Inference: backlog completeness and paper coverage displaced expected-value prioritization. The primary mathematical bottleneck—learned exact state transition over widths 256–1024 under CPU/runtime limits—needed a smaller portfolio built from error decomposition, algebraic symmetry, representation, and complexity arguments.

### 6. The completion oracle mixed a local objective with an unavailable external event

Observed: the revised goal required at least 85% on the official private evaluation when legitimate access became available, while also saying not to declare completion if that was the only missing evidence.

Inference: a permanently active local run could never complete even after exhausting safe local work. The private benchmark should be an external release blocker, while the local research tranche should have its own measurable completion/continuation rule.

### 7. More process was added in response to process defects

Observed: failed audits often spawned new workers, repaired roots, closure judges, and repeated final audits. Seventeen agent names explicitly referenced repair/recovery/hardening/reconciliation.

Inference: the recovery mechanism amplified the control plane. A better pattern is one bounded remediation package that fixes all same-cause defects, followed by one package-level verification and one risk-boundary review.

## High-leverage optimizations for the recovered run

1. **Use a compact file board as the only state.** One active task, one role, one receipt; render task-local prompts rather than carrying the full session.
2. **Cap coordination by policy.** Default to one Scout, one coherent Worker package, and a Judge only at phase/risk/final boundaries. Reuse an agent only for a tightly coupled correction. Avoid nested review chains.
3. **Replace polling with completion-triggered execution.** Long training gets a persistent runner, explicit exit/health artifacts, an ETA from measured throughput, and at most one coarse heartbeat near the expected boundary.
4. **Adopt a four-level verification ladder.** L0 static/unit checks for changed code; L1 focused scientific smoke; L2 coherent-package regression; L3 full project/oracle only at promotion or phase/final audit.
5. **Rank hypotheses by expected information gain per wall/CPU cost.** Use analytical feasibility and microbenchmarks before scaffolding; run the cheapest falsifier; reserve full-fidelity formal evaluation for survivors. Final comparisons retain full fidelity.
6. **Separate the local oracle from private-benchmark availability.** Complete the local tranche when a frozen candidate clears all locally available held-out/checked/compliance gates; record private evaluation as an external blocked release task.
7. **Make the Tier-7+ bottleneck explicit.** Require every hypothesis to state which failure mode it attacks, expected asymptotic/runtime effect, minimal falsifier, promotion threshold, and stop rule.
8. **Favor mathematically structured hypotheses over paper enumeration.** Explore representation and decomposition ideas that can be learned end-to-end without prohibited inference arithmetic: symmetry/equivariance, redundant learned residue/state representations, carry/borrow latent automata, divide-and-compose state transitions, confidence-calibrated commuted views, error-correcting latent codes, and curriculum based on bit-complexity—not generic architecture novelty alone.
9. **Stop exhaustive mechanism obligations.** “Test every paper mechanism” becomes a screened portfolio: incompatible or dominated mechanisms receive a documented analytical rejection; only high-value survivors receive implementation budget.
10. **Track throughput metrics beside accuracy.** Scientific result rate, wall/CPU hours per decisive experiment, agent/tool calls per completed experiment, verification minutes per changed line/package, and context/compaction counts become run-health metrics.

## Unknowns for later tasks

- Which TradeArch result documents encode a successful version of this recovery and which patterns are actually transferable.
- Current repository/registry truth after the interrupted final turn: dirty state, active experiment, preserved EXP-648 roots, and whether any later process completed outside the trace.
- Current exact incumbent and independent-seed coverage beyond the trace's Candidate-005 claims.
- Competition runtime/parameter constraints and current code hot spots that determine feasible Tier-7+ decompositions.
- Which mathematical/DSA hypotheses have already been tried under different experiment names and which remain genuinely novel.

## Recommended next evidence sequence

1. Read-only TradeArch precedent audit.
2. Read-only SAIR current-state and experiment-registry audit.
3. Constraint-aware mathematical/DSA hypothesis mapping tied to observed Tier-7+ failures.
4. One Judge decision selecting the largest safe optimization package.
5. One Worker package, tiered verification, then a phase-boundary review only if warranted.
