---
type: Experiment
title: EXP699 complete h92 W256 backbone factorial
description: Fresh diagnostic-complete rerun of the unchanged EXP698 four-arm backbone factorial.
status: execution-interrupted
tags: [autoresearch, backbone, factorial, diagnostics, straight-through, scheduler]
timestamp: "2026-08-02T00:00:00+00:00"
---

# EXP699 complete h92 W256 backbone factorial

EXP699 is the fresh successor to parked EXP698. The only executable additions are evidence-only diagnostics and persistence: frozen source, a byte-identical reset to source, one preregistered randomized-weight model per trained arm, and actual 500-case Tier 1-5 parity. Candidate005/EXP609 checkpoint `cbc4a1bc...d4bd`, h92/two-cycle topology, initialization, arm factors, W128/W256 data order, AdamW optimizer, native PyTorch schedules, work, hard learned-residue deployment route, and primary denominators remain unchanged. Exact source-component and function hashes gate reuse of EXP698 root20261386 throughput.

The four arms cross teacher-forced versus hard straight-through state exposure with single cosine `1e-5 -> 5e-7` versus native LinearLR warmup to `5e-6` followed by CosineAnnealingWarmRestarts to `5e-7`. Each arm executes exactly 4,800 updates at batch 16/horizon 8: updates 1-1,200 are W128-only and updates 1,201-4,800 use the same alternating balanced W128/W256 order. Checkpoints are written at 400, 800, 1,200, 2,400, 3,600, and 4,800; curves are written every 100 updates.

Formal primary evidence per arm is 3,000 W128 transitions, 3,000 W256 transitions, 300 W128 rollouts, and 600 W256 rollouts. Frozen source and the one byte-proven reset model use those same denominators. Each of the four randomized controls uses 300/300 transitions and 100/200 W128/W256 rollouts. Diagnostic caches are reusable only when checkpoint/state, evaluator, generated-case, and complete config hashes match exactly. All surfaces persist paired vectors and hashes, exact accuracy, confidence, first error, invalids, reload/work/resource evidence, and trained-minus-randomized interpretation. Tier 1-5 parity executes all 500 official-generator seed-1 cases and compares two independently loaded unchanged lower-tier routes.

Fresh roots are calibration `20261395`, formal `20261396`, confirmations `20261397-98`, and independent `20261399-20261403`. All are audited across repository records, scratch paths, and MLflow before the first reservation. Both confirmations open only if the formal winner preserves healthy W128 and reaches at least 85% W256 rollout; all five independent roots open only after both confirmations pass. A negative or unhealthy formal result still completes every formal denominator. Runtime over 300 seconds, weak intermediate accuracy/loss, failed health, and lack of early W256 gain never truncate science.

Calibration reuses EXP698 root20261386 throughput only after exact route equivalence and profiles added paired-evidence/diagnostic orchestration plus a fresh complete candidate route. It records revised wall, process-CPU, cgroup-CPU, RSS, and artifact expectations with 1.25x soft and 1.75x hard wall ceilings. The hard ceiling is a runaway-health control, not a scientific futility rule. Native PyTorch schedulers, safetensors, MLflow, and standard-library deterministic hashing are maintained choices; no new dependency is needed.

## Terminal partial result

EXP699 is terminalized as execution-interrupted, not scientifically rejected. Calibration root
`20261395` completed and admitted formal execution. Formal root `20261396` completed both
teacher arms, including every primary denominator. `teacher_cosine` reached 2,994/3,000 W128
transitions, 260/300 W128 rollouts (2 invalid), 2,991/3,000 W256 transitions, and 373/600 W256
rollouts (6 invalid). `teacher_restarts` reached 2,994/3,000, 257/300 (2 invalid),
2,989/3,000, and 342/600 (6 invalid), respectively. Both teacher arms pass the frozen W128
health rule.

The `straight_through_cosine` arm produced authenticated checkpoints only through update 1,200;
`straight_through_restarts` never opened. Frozen/reset/randomized/lower-tier diagnostics never
opened, so there is no factorial, diagnostic, winner, confirmation, or rejection verdict. Roots
`20261397` through `20261403` remain unopened. The partial progress artifact is preserved with
SHA-256 `64545c3ce37696770e70b5bb7962808ffe9ede77163fa5dfb2cb44ee499b5b6e`;
the update-1,200 straight-through checkpoint is
`abe06cb263fcc84831d97b6ebe5bfc8e16de1c8dad34795c73c8258eddb4fe5a`.
