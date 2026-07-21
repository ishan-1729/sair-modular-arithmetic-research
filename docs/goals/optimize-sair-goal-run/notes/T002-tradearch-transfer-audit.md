# T002 — TradeArch Transfer Audit

## Documents inspected

1. `/mnt/d/Projects/TradeArch/docs/knowledge/research/repository-optimization-study.md` — 948-line canonical synthesis of implemented algorithms, data movement, process/runtime, logging, verification, measurements, discovery method, and remaining opportunities.
2. `/mnt/d/Projects/TradeArch/docs/work/active-reviews/REPOSITORY_AUDIT_2026-07-17.md` — 1,260-line canonical execution Kanban, finding ledger, remediation/re-audit history, verification evidence, and limitations.
3. `/mnt/d/Projects/TradeArch/AGENTS.md` — repository execution/delegation rules confirming one canonical Kanban, bounded task specifications, principal integration ownership, and worker completion not equaling project completion.
4. Current Git status and history for the two result documents.

The two result documents are currently untracked in the local TradeArch worktree. The broader candidate contains roughly 120 tracked modifications plus 14 intended additions, matching the documents' own reported handoff state. The active review still has `status: active` even though its Kanban cards are all marked done, because its archive/history guard requires a later committed transition. These are direct observations and important durability caveats.

## What TradeArch demonstrably achieved

The result set describes a repository-wide correctness/performance remediation and later re-audit. Its strongest measured optimization outcomes are deliberately scoped:

- 43.4% historical full-gate wall reduction from DSATUR conflict phases plus LPT shard balancing.
- About 979x for a local permutation-reconstruction kernel with exact equality.
- 836.5x and 7,032.5x local contiguous portfolio-row extraction kernels at different shapes.
- 11.1x for a native normal CDF kernel.
- More than 667x opportunity in a repeated source-discovery kernel and 4.68x for the implemented shared-inventory route.
- 12.75x for a runtime-identity inventory phase.
- 3,014x for a local indexed chain-slicing kernel.
- 1.991x for single-pass bounded JSON encoding.
- Final integration evidence of 1,509 fast tests across 15 shards in 798.29 seconds, 24 slow tests across five shards in 272.25 seconds, and an 8-second Windows package smoke.

The documents never turn a kernel ratio into an application-wide claim. They label evidence as current measurement, historical measurement, current verification, source-derived reasoning, or candidate. That discipline transfers directly to SAIR's distinction among observed results, source claims, and hypotheses.

## Transferable process patterns

### 1. Compile durable state once, then operate on compact maps

TradeArch's `PermutationPlan` performs semantic normalization once and represents each later draw as compact index maps. Its shared source inventory similarly replaces adapter-specific tree traversals.

**SAIR trace failure addressed:** 34 compactions, massive repeated context carrying, repeated repo/status/validation scans, and agents reconstructing the same state.

**Transfer:** compile the run into `goal.md`, `state.yaml`, compact notes, a single experiment-family matrix, and a gate inventory. Every task receives only its task-local inputs and returns one receipt. Do not repeatedly reread the monolithic prior trace, full backlog, or whole repository.

### 2. Optimize the route invariant, not one favorable branch

TradeArch found fast helpers coexisting with routes that still rebuilt plans, spawned pools, or filtered growing prefixes. It followed public entry points through default/fallback/one/many-worker branches before declaring improvement.

**SAIR trace failure addressed:** training and inference experiments were often compute-conscious, but orchestration, provenance repair, repeated audits, status sampling, and full verification dominated total wall time.

**Transfer:** measure end-to-end experiment latency from hypothesis selection through durable decision, not only model training. A scientific optimization is accepted only if the actual SAIR experiment route—data generation, training, evaluation, registration, and verification—improves information throughput.

### 3. Use mathematical decision bounds to stop only when the decision is fixed

TradeArch's decision-only permutation test computes exact lower/upper final p-value bounds and stops once remaining permutations cannot change the decision, while persisted distribution reports still run the full declared universe.

**SAIR trace failure addressed:** long evaluations and calibrations continued after some gates were already impossible, while arbitrary wall caps were rightly rejected.

**Transfer:** preregister decision-safe bounds. For an `N`-case exact-accuracy gate after `m` cases and `k` correct, the attainable final interval is `[k/N, (k + N - m)/N]`. Stop a selection-only evaluation once the promotion threshold lies outside that interval. Use sequential confidence/futility rules for noisy training comparisons. Frozen release evidence still runs the full declared evaluation.

### 4. Pre-admit combinatorial and resource work

TradeArch counts split products, grids, archive members, rows, bytes, descriptors, workers, and in-flight copies before materialization.

**SAIR trace failure addressed:** EXP-649 and EXP-648 built extensive scaffolds before runtime or scientific feasibility was settled.

**Transfer:** every hypothesis card declares parameter count, sequence-length complexity, estimated activation/state bytes, projected wall/aggregate CPU, expected inference time, artifact size, and smallest falsifier before implementation. Reject dominated or infeasible mechanisms analytically.

### 5. One persistent topology; no nested orchestration

TradeArch compiles one plan, creates one bounded worker pool, initializes immutable state once, derives RNG from logical indices, prevents nested pools, and owns deterministic teardown.

**SAIR trace failure addressed:** 135 agent paths, 156 spawns, 114 follow-ups, 785 agent waits, 545 wait timeouts, and repeated background-run recovery.

**Transfer:** one active GoalBuddy task, one assigned role, no nested review chain, one persistent long-run process, logical seed identity independent of completion order, and one receipt at completion. Long jobs use an exit artifact and coarse ETA-based wake rather than agent polling.

### 6. Schedule validation as conflict phases plus makespan balancing

TradeArch models tests as a conflict graph, colors it with deterministic DSATUR, then uses LPT to balance duration within conflict-free phases. Static checks run before expensive fast/slow/Windows gates; expensive independent jobs fan out only after the cheap gate passes.

**SAIR trace failure addressed:** hundreds of repeated test/validator/status/diff invocations and package-level gates after micro-edits.

**Transfer:** define L0 static/diff checks, L1 focused unit/scientific smoke, L2 package regression, and L3 full project/oracle. Run L0/L1 during implementation; run L2 once per coherent package; run L3 at promotion/phase/final boundaries. Where safe, parallelize disjoint test groups internally, not via multiple write-capable agents.

### 7. Monitor mutations/events, not snapshots in a polling loop

TradeArch replaced per-shard full Git polls with inotify plus exactly three complete identity checks and gave each shard attributable logs/telemetry.

**SAIR trace failure addressed:** 4,425 yielded-exec waits, 3,539 still-running outputs, and repeated Git/status checks.

**Transfer:** use filesystem/process completion markers, systemd/Task Scheduler, bounded logs, and start/end identity checks. Status views read a detached immutable result rather than re-traversing live state.

### 8. Preserve evidence semantics and scope every multiplier

TradeArch separates kernel, phase, and integration measurements; equality/golden oracles accompany microbenchmarks; current and historical evidence are not mixed.

**SAIR trace failure addressed:** risk of confusing transition accuracy, public benchmark results, cached evidence, and exact held-out full-set accuracy.

**Transfer:** label every metric by dataset identity, seed status, model hash, selection versus release role, exact-answer denominator, and scope. A transition/kernel gain never substitutes for end-to-end exact rollout or full candidate accuracy.

### 9. Consolidate duplicated authority only after behavior is pinned

TradeArch centralizes source inventory, research views, strict loading, JSON bounds, publication, and batch queries, but defers splitting large modules until characterization exists.

**SAIR trace failure addressed:** recovery added more roots, validators, and closure tasks when existing authorities drifted.

**Transfer:** create one experiment status authority, one gate definition, one registry update path, and one long-run launcher. Remove or quarantine superseded paths only after focused equivalence tests.

## TradeArch patterns that should not transfer blindly

1. **Mega-diff execution.** The current local candidate spans roughly 120 modified tracked files and 14 additions. SAIR should use coherent, reversible packages and commit/record each durable scientific result rather than accumulate a repository-wide unpublished tranche.
2. **Untracked “published” documentation.** Both cited result documents are currently untracked. SAIR's result/board artifacts must be committed or otherwise durably registered before being treated as published evidence.
3. **Completion by reviewer consensus.** TradeArch's earlier final reviewer reported no remaining P0/P1/P2 defect, but the later fresh re-audit retained 28 findings (7 P1, 18 P2, 3 P3). SAIR must use an explicit oracle and recognize that an audit samples risk; it does not prove universal absence of defects.
4. **Audit expansion as the default response.** TradeArch legitimately used multiple domain reviewers for a whole-repository remediation. SAIR's narrower research loop should not reproduce that reviewer count per experiment. Use one adversarial Judge at risk/phase/final boundaries.
5. **Full-suite verification for read-only discovery.** TradeArch explicitly stopped silent multi-minute broad invocations and made no pass/fail claim. SAIR Scouts should not run full gates; Workers own scoped verification, and the PM/Judge owns the oracle.
6. **Performance sophistication without bottleneck proof.** TradeArch's Rust/Go section correctly rejects rewrites absent a dominant profiled kernel. SAIR likewise should not choose exotic model architectures because they are intellectually deep; they must attack the measured Tier-7+ failure surface within runtime limits.

## Concrete deltas proposed for the SAIR recovery board

1. Add a run-health budget to the Judge package: maximum one Scout and one Judge per phase boundary; zero repeated `wait_agent` after a timeout; no nested reviewers; one full verification per coherent package.
2. Require T003 to inventory the experiment route once: status authority, runners, gate commands, data movement, background-process topology, and current dirty state.
3. Require T004 to use explicit evidence labels (`observed-current`, `observed-historical`, `source-claim`, `hypothesis`, `candidate`) and to include analytical pre-admission plus a decision-safe futility rule for every experiment.
4. Require T005 to select by expected information gain per end-to-end wall/CPU cost, not novelty, paper count, or local kernel elegance.
5. Require T006's verification list to be tiered and deduplicated; its package must include the experiment's start/end identity, immutable logs, and one durable registry update path.
6. Add run-health telemetry to the final oracle: decisive experiments per wall day, wall/CPU per decision, agent/tool calls per decision, polling count, verification minutes, compactions, and context size.
7. Treat unavailable private evaluation as a separately blocked release task, not a reason the locally executable research board can never close.

## Transfer conclusion

TradeArch's deepest transferable result is an algorithmic one: remove repeated work and duplicate authority around the critical route before adding more sophisticated inner algorithms. For SAIR, the critical route is the experiment-decision loop. The optimized loop should compile state once, pre-admit cost, run the cheapest mathematically decisive falsifier, persist one result, and verify at package boundaries. Deep DSA belongs inside that route only when tied to a measured failure mode and exact oracle.
