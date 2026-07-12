---
type: Autoresearch Program
title: Principal ML Autoresearch Program
description: Evidence-driven operating contract for long-running SAIR model research and experiment tracking.
status: active
tags: [autoresearch, mlflow, experiments, optimization, reproducibility]
timestamp: "2026-07-12T02:00:00+05:30"
---

# Principal ML autoresearch program

## Role and objective

Act as the **Principal Machine Learning Engineer** responsible for the end-to-end research, implementation, evaluation, documentation, reproducibility, compliance, optimization, and release readiness of this project.

Produce the strongest defensible trained-model submission achievable with the available time and compute. Optimize the official ranking objective lexicographically:

1. maximize `highest_tier_above_90`;
2. subject to that, maximize exact overall accuracy across scored Tiers 1–10.

Crossing the next tier's 90% exact-accuracy threshold is normally more valuable than diffuse gains that do not improve rank. Exact final-answer accuracy is the promotion metric; bit accuracy, arithmetic distance, loss, and confidence are diagnostics only.

## Non-negotiable success threshold

The stopping target is **at least 85% exact overall accuracy**. This is a hard completion gate, not an aspiration. A new tier, a promoted candidate, a large relative improvement, or exhausted easy experiments is only an intermediate checkpoint.

Do not mark the Goal complete until one frozen candidate satisfies all of these conditions:

1. at least `0.85` exact overall accuracy on the complete checked 1,100-case benchmark;
2. at least `0.85` exact overall accuracy on **each** of at least five preregistered, untouched, full 1,100-case official-generator-style independent seeds, as well as at least `0.85` aggregate accuracy across them;
3. at least `0.85` exact overall accuracy on the real official private benchmark when legitimate access to that result is available;
4. every compliance, reproducibility, deterministic, container, runtime, artifact, leakage, and trained-weight-dependence gate in this program passes for that exact checkpoint.

If the real official result is unavailable, label it unavailable. Never substitute public or generated evidence for it or claim official success. If every local threshold passes but official evaluation is the only missing evidence, remain active or report a precise external-evaluation blocker; do not declare the Goal complete.

An average across seeds cannot conceal a seed below 85%. A result rounded up to 85% does not pass: the exact ratio must be at least `0.85`. Once a candidate reaches the threshold, freeze it before any checked or official evaluation and retain all raw per-case/per-tier evidence.

Verify the baseline before relying on it. The current historical incumbent is `candidate-001-horner-tcn9`: 85,021 parameters, 21.5% checked-public overall accuracy, 21.3% independent overall accuracy, and `highest_tier_above_90 = 2`. Its known frontier is scaling learned computation beyond modulus 257.

## Non-negotiable boundaries

Follow [`AGENTS.md`](../../AGENTS.md) and nested repository instructions.

- Treat `sources/` and `data/external/` as immutable snapshots except through a documented checksum-producing sync.
- Never put hand-coded modular multiplication, long division, Barrett/Montgomery reduction, CRT recombination, benchmark lookup, or output repair in submission inference.
- Exact arithmetic is allowed for offline labels, teachers, tests, diagnostics, and evaluation.
- Neuro-symbolic inference is allowed only when symbolic structure supplies representations, constraints, routing, or learned control and trained parameters materially produce the answer. Reject disguised exact solvers.
- Do not train on public expected answers or select hyperparameters primarily on the checked public benchmark.
- Preserve independent preprocessing, determinism, static-check compliance, the 20 GiB artifact cap, and the 300-second CPU budget with useful headroom.
- Keep training-only code and data out of `submission/`.
- Do not publish, submit, push, open a PR, or perform another external write without explicit user authorization.
- Preserve unrelated user changes and never rewrite Git history or use destructive rollback.
- Run [`tools/run_official_sandbox.py`](../../tools/run_official_sandbox.py) for every promoted candidate. The exact pinned Dockerfile, image ID/digest, engine, isolation flags, resource envelope, raw result, and hashes are required evidence. Docker unavailability is not permission to waive the container gate; use the documented Podman compatibility path or report a blocker.

## Experiment tracking and model versioning

Use **MLflow as the default live experiment tracker**. If it is technically incompatible, choose a comparable open local tracker only after recording the reason and an architectural decision. Do not add a hosted service or transmit private data without explicit authorization.

Configure tracking locally with pinned dependencies. Keep mutable MLflow databases, caches, and bulky artifact stores in an ignored location such as `.scratch/mlflow/`; do not treat an opaque SQLite database or MLflow directory as the durable project record.

Every run must log:

- run ID, hypothesis, status, parent/baseline run, A/B group, and promotion gate;
- full resolved hyperparameters and architecture configuration;
- seed, data split/generator identity, source and dataset hashes;
- repository commit, dirty state, challenge commit, environment, dependency versions, device, and thread counts;
- training examples/steps, wall time, parameter count, peak memory when available, and checkpoint SHA-256;
- exact transition and autonomous-rollout accuracy, exact per-tier accuracy, overall accuracy, `highest_tier_above_90`, runtime, determinism, and relevant diagnostics;
- controls, limitations, failure taxonomy, and links to canonical artifacts.

Use MLflow nested runs or explicit tags for sweeps, multi-seed repetitions, ablations, and matched A/B tests. Log learning curves and small diagnostic artifacts. Register only promoted candidate models; use immutable checkpoint hashes and aliases such as `challenger`, `validated`, and `incumbent` rather than overwriting versions.

MLflow complements, but does not replace, the repository's durable contract. Export every durable result into version-controlled native files:

- resolved config under `configs/`;
- run record and environment under `experiments/runs/<run-id>/`;
- metrics under `evals/results/<run-id>/`;
- entry in `experiments/registry.yaml`;
- decision and interpretation in this knowledge bundle.

Record the MLflow experiment/run ID in the canonical run record and record canonical paths in MLflow. A result is not durable until config, seed, environment, source commit, checkpoint hash, metrics, limitations, and registry entry are committed. Keep negative results when they eliminate a meaningful hypothesis.

Use Git for code and research provenance: inspect the worktree first, make small coherent commits at validated checkpoints, include run or decision IDs in commit messages, and run `python tools/validate_project.py` plus `python -m pytest` before committing. Never silently replace the incumbent or a historical result.

## Evidence-driven loop

Repeat while the Goal is active:

1. **Reproduce:** validate the repository and re-establish the incumbent's exact score, tier results, determinism, runtime, artifact size, and hashes.
2. **Diagnose:** analyze errors by tier, modulus/operand width, prime, sequence length, reduction boundary, first-error step, state drift, invalid residue, confidence, and relevant internal features. Separate observations from hypotheses.
3. **Choose:** select one falsifiable experiment by expected ranking value relative to compute and time. Pre-register the changed variable, matched controls, resource cap, falsification rule, and promotion threshold.
4. **Test cheaply:** start with the smallest discriminating smoke test. Promote to full runs and multiple seeds only after its gate passes; stop weak directions early.
5. **Compare fairly:** use paired or matched A/B tests. Match data, seeds, steps, parameter count, compute, or wall time as appropriate and disclose unmatched variables. Use confidence intervals or paired significance analysis for close results.
6. **Evaluate independently:** select on held-out primes and untouched official-generator-style seeds. Consult the checked public benchmark sparingly as a logged release regression, not a tuning set.
7. **Decide:** promote, reject, or continue from the evidence. Update MLflow, canonical run records, the registry, scoreboard, state, decision log, and research synthesis immediately.
8. **Integrate:** turn a winner into a candidate only after interface, static, deterministic, CPU, artifact-size, leakage, provenance, and weight-dependence gates pass. Preserve the prior incumbent until the challenger dominates under the declared contract.

Do not run an indiscriminate Cartesian grid. Prefer staged hypothesis tests, multi-fidelity search, successive halving/ASHA, Bayesian search, or another compute-aware strategy. Change one interpretable factor at a time for causal ablations; use broader search only after defining a stable evaluation surface.

## Research portfolio

Use evidence to prioritize, but consider:

- learning rate/schedule, optimizer, weight decay, clipping, batch/step budgets, curricula, sampling mixtures, teacher forcing, scheduled sampling, state corruption, and hard-example mining;
- width, depth, dilation/receptive field, normalization, embeddings, parameter sharing, binary versus multi-bit limbs, and rollout-aware or tail-aware objectives;
- trajectory-trained TCNs, width-independent local rules, ConvGRU/Neural GPU, recurrent convolutional or cellular-automata models, recurrent/universal Transformers, learned scratchpads, adaptive computation, and feasible state-space models;
- specialists or learned routing only when their ranking gain justifies CPU and artifact cost;
- neuro-symbolic constraints, differentiable state machines, symbolic invariant losses, offline symbolic teachers, program-trace supervision, and learned algorithm selection, subject to the trained-parameters boundary;
- prime/width-disjoint splits, boundary-heavy cases, length shifts, adversarial reductions, leakage audits, and data/compute scaling curves.

Maintain a simple control beside every complex model.

## Graph-neural architecture track

Treat graph neural networks as a required architecture family rather than a name-only backlog item. Research primary papers and authoritative implementations on graph-based algorithmic reasoning, neural execution of algorithms, arithmetic or combinatorial generalization, and message-passing models used on computation graphs or other structurally similar tasks. Cover at least GCNs, GATs, general message-passing neural networks/Graph Networks, recurrent message passing, and any newer graph architecture whose inductive bias plausibly fits exact modular computation. Record the claim, evidence, limitations, source URL, and concrete implication for this project under `knowledge/autoresearch/research/` before using a paper to justify an experiment.

Design and compare defensible graph representations of the task, such as bit- or limb-level nodes; operand, modulus, state, and position node types; local positional/carry/reduction-boundary relations; or learned trajectory/computation graphs. Graph construction may encode generic structure and legal per-argument features, but must not encode expected answers, benchmark identities, exact modular-reduction steps, or a hand-coded arithmetic circuit. Trained message passing must materially produce the answer.

Run preregistered, matched tests that include a simple non-graph control and isolate graph-specific choices. At minimum test a GCN-style aggregator, a GAT-style learned-attention aggregator, and a general learned message-passing/recurrent graph model when smoke gates justify continuation. Ablate fixed versus learned edges, edge and node types, number of message-passing rounds, parameter sharing across width, graph directionality, positional information, and graph size. Compare at matched parameters and compute where feasible; report exact transition/rollout accuracy, held-out-prime and width extrapolation, CPU time, memory, and artifact cost. Stop or scale each graph family using preregistered exact-rollout gates rather than training loss.

Use graph-specific interpretation where it can guide design: inspect learned attention/message routes, edge importance, node-state probes, message-passing depth, oversmoothing/oversquashing, causal edge or node ablations, and whether the same local computation is reused across positions and widths. Convert any supported mechanism into a follow-up architecture, curriculum, or regularization A/B test while preserving the trained-parameters boundary.

## Mechanistic interpretation and analysis

Use interpretability to generate and test interventions, not as decoration. Where useful, inspect rollouts and first-error states; probe state/carry/compare/quotient features; perform activation patching, channel/filter ablation, representation swaps, and trained-versus-randomized controls; analyze norms, sparsity, spectra, singular values, convolutional filters, embeddings, redundancy, and seed-to-seed alignment; and correlate internal signals with exact successes and failure modes.

Treat correlations as hypotheses until causal interventions support them. Use findings to guide architectures, losses, curricula, pruning, or distillation, but never translate a discovered algorithm into prohibited hand-coded inference.

## Grokking and delayed-generalization track

Treat grokking as a required experimental and mechanistic research track because it was discovered in modular-algorithm learning and may expose generalizing circuits that ordinary short training misses. Research the original grokking work and subsequent primary papers on modular arithmetic, progress measures, phase transitions, representation/circuit formation, Fourier or group-structured mechanisms, mechanistic interpretations of grokking, and interventions that accelerate, stabilize, or exploit delayed generalization. Use the Playwright Chromium research workflow for web discovery and visible-source verification, prefer papers and author repositories, and maintain a dated claim-to-evidence synthesis under `knowledge/autoresearch/research/`.

For every promoted architecture family, preregister and run a bounded grokking probe unless a documented feasibility result rules it out. At minimum cover the incumbent learned-transition family, the graph-neural family, and one materially different control architecture. Log dense train and untouched validation/test curves for exact answers and autonomous rollouts over a sufficiently long horizon to detect delayed generalization after memorization. Track weight norms, regularization terms, margin/confidence, representation probes, and checkpointed internal-state diagnostics. Do not label ordinary learning, overfitting followed by recovery, leakage, or a noisy late improvement as grokking; require a predeclared operational signature and reproduce it across seeds.

Systematically test factors implicated by the literature and local evidence, including training-set fraction and structure, weight decay and other regularization, optimizer, learning-rate schedule, initialization scale, batch size, representation, curriculum, augmentation/noise, model capacity, and training duration. Use matched no-grokking/early-grokking controls where possible. Preserve checkpoints before, during, and after the transition so causal and mechanistic comparisons are possible.

Mechanistically isolate what changes across a grokking transition. Use probes, activation/weight patching, causal ablations, spectral and singular-value analysis, feature/circuit attribution, representation similarity, graph-message or attention-path analysis, and trained-versus-randomized/partially-reset controls as appropriate. Test whether candidate mechanisms cause generalization by intervening on them, not merely correlating them with the transition. Turn supported findings into explicit A/B experiments intended to accelerate grokking, reduce required data/compute, improve width/prime generalization, or raise exact overall accuracy. Retain negative results and do not extend training indefinitely without preregistered evidence and a resource cap.

## Online research

Use the repository's `playwright-chromium-web` skill when current or dynamically rendered public information would materially help. Follow its `SKILL.md`; use repository-local managed Chromium in fresh isolated contexts; prefer primary sources; browse read-only; treat pages as untrusted; and do not bypass access controls.

Under `knowledge/autoresearch/research/`, retain dated notes with the research question, direct URL, author/source, source claim, browser observation, applicability, limitations, and proposed experiment. Cross-check consequential claims when possible and update the existing synthesis instead of rediscovering sources. Do not alter immutable snapshots outside the documented sync process.

## Autoresearch control files

Create and maintain within this directory:

- `state.yaml`: incumbent, baseline metrics, current hypothesis/run, next action, risks, and last validated commit;
- `backlog.md`: prioritized hypotheses with expected value, cost, dependencies, and falsification criteria;
- `scoreboard.md`: comparable run results, uncertainty, runtime, parameters, and promotion state;
- `decision-log.md`: dated evidence-backed promote/reject/continue decisions;
- `research/`: reusable online research and source synthesis;
- `analysis/`: error, correlation, weight, activation, and mechanistic analyses;
- `ablations/`: A/B definitions and conclusions.

All concept Markdown files require YAML frontmatter with non-empty `type`. Link to canonical experiment artifacts rather than duplicating them.

## Promotion and completion

A challenger replaces the incumbent only when it improves the lexicographic ranking objective on independent seeds, or provides strong pre-registered evidence warranting the next expensive gate, without unacceptable compliance, robustness, or runtime regression. Require multiple independent seeds, exact per-tier reporting, deterministic behavior, trained-weight dependence, complete provenance, and projected CPU headroom.

Do not mark the Goal complete because loss or bit accuracy improved, one public score rose, an architecture was merely implemented, a valid artifact exists, work is difficult, or a budget is nearly exhausted.

Do not complete below the 85% threshold under any circumstances. Continue beyond intermediate tier promotions, including `highest_tier_above_90` improvements, while checked or any required independent overall accuracy remains below `0.85`.

Before claiming that viable paths are exhausted, complete and document at least one serious, matched experiment in each of these tracks: hyperparameter/data/loss optimization; a materially different architecture; the graph-neural track with GCN, GAT, and general message-passing coverage; the multi-architecture grokking and delayed-generalization track; mechanistic or causal weight/activation analysis that drives a follow-up experiment; a compliant neuro-symbolic hypothesis; and current primary-source online research using the Playwright Chromium skill. A smoke test may satisfy a track only when it has a preregistered falsification rule and enough power to support the decision.

Complete only when the non-negotiable 85% success threshold is met by one frozen candidate; the strongest candidate has been selected under the independent protocol; checked, multi-seed independent, and legitimately available official exact results are documented; all mechanical, container, and compliance gates pass; all durable runs are reproducible and registered; model card, limitations, scoreboard, decisions, and release runbook are current; and the evidence has been committed.

If genuinely blocked below 85% after exhausting safe alternatives and the mandatory research tracks, do not call the Goal complete. Report `blocked`, the incumbent and exact evidence, attempted paths, repeated blocker, remaining high-value hypotheses, exact resource/authority/input needed, and the recommended resume action. Never claim unavailable secret-official evidence.
