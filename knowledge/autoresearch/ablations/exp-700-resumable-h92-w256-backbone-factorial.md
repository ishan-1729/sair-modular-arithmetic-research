---
type: Experiment
title: EXP700 resumable h92 W256 backbone factorial
description: Crash-resumable persistent execution of the unchanged EXP699 four-arm backbone factorial.
status: preregistered
tags: [autoresearch, backbone, factorial, recovery, systemd, straight-through, scheduler]
timestamp: "2026-08-04T00:00:00+05:30"
---

# EXP700 resumable h92 W256 backbone factorial

EXP700 repeats the complete EXP699 h92/two-cycle four-arm factorial under fresh roots. The
source checkpoint, model topology, initialization, data curriculum and order, AdamW optimizer,
schedules, losses, 4,800 updates per arm, batch 16/horizon 8, checkpoints, curves, hard deployed
route, primary denominators, diagnostic denominators, confidence contract, gates, and conditional
root policy are unchanged. Only crash recovery and persistent process lifecycle change.

Atomic recovery state is written at update zero and at least every 100 updates. It includes the
model, AdamW, scheduler, Python/NumPy/Torch RNG, sampler and offset RNG states, curves, width/data
order ledgers, work counters, root/source identity, and integrity hashes. Every completed train,
evaluation, and diagnostic stage has an immutable hash-gated phase manifest. Resume requires the
same root, frozen source/config/runner/test hashes, recovery snapshot hash, logical work ledger,
and a prior abnormal-exit marker. Completed stages are reused; semantic, compliance, identity,
and work failures terminalize instead of retrying.

Fresh roots are calibration `20261404`, formal `20261405`, confirmations `20261406-07`, and
independents `20261408-12`. Both confirmations open only when the formal winner is healthy at W128
and reaches at least 85% W256 rollout. All five independent roots open only if both confirmations
pass. Runtime, weak intermediate accuracy, failed health, or lack of early W256 gain never truncates
formal work. Actual candidate runtime at or below 300 seconds is release evidence only.

Formal execution is owned by a named transient WSL user-systemd service with fixed workdir,
environment, logs, completion marker, single-instance locking, bounded recoverable restarts, and
`RuntimeMaxSec` equal to the calibrated 1.75x hard wall ceiling. Recovery artifacts and systemd are
training infrastructure only; no dependency, state, answer, or exact arithmetic enters
`submission/`.
