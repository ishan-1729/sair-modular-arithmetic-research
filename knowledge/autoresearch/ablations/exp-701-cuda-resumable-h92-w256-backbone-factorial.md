---
type: Experiment
title: EXP701 fresh CUDA resumable h92 W256 backbone factorial
description: Full-fidelity fresh-root RTX 4090 replay of the unchanged EXP700 four-arm scientific contract.
status: preregistered
tags: [autoresearch, backbone, cuda, factorial, recovery, runpod, straight-through, scheduler]
timestamp: "2026-08-05T00:00:00+05:30"
---

# EXP701 fresh CUDA resumable h92 W256 backbone factorial

EXP701 is a complete fresh-root replay, not a migration or salvage of EXP700's partial CPU
run. EXP700 roots `20261404/05` remain untouched historical evidence. Every EXP701 arm,
update, primary denominator, diagnostic, conditional confirmation, and independent run is
executed on the same deterministic RTX 4090 CUDA route. GPU recovery checkpoints may resume
only an interrupted EXP701 GPU root and never substitute CPU work.

The scientific contract is byte-for-byte equal to EXP700 for model, arms, data, training,
evaluation, gates, and resource policy: h92/two-cycle/99,824 parameters; the four
teacher/straight-through by cosine/restart arms; 4,800 updates per arm; batch 16 and horizon
8; identical optimizer, schedules, curriculum, losses, checkpoints, curves, deployed hard
route, exact-answer denominators, diagnostics, confidence rules, and promotion gates. The
only changed variable is deterministic device execution under the accepted pinned cu124
environment.

Fresh roots are calibration `20261413`, formal `20261414`, confirmations `20261415-16`, and
independents `20261417-21`. Both confirmations open only when the complete formal winner is
healthy at W128 and reaches at least 85% W256 rollout. All five independent roots open only
if both confirmations pass. No wall-clock target, runtime result, weak accuracy, or early
trajectory may reduce or skip a denominator.

Atomic recovery state is written at update zero and at least every 100 updates. It contains
model, AdamW, scheduler, Python/NumPy/CPU-Torch/CUDA RNG, sampler and offset RNG states,
curves, width/data order ledgers, work counters, root/source identity, and integrity hashes.
Completed train/evaluation/diagnostic phases use immutable manifests. Resume requires the
same EXP701 root and frozen identities; it never opens a second root marker.

The accepted hardware gate is `cuda-hardware-acceptance-runpod-20260805g` from source-fix
commit `02678914...` and receipt commit `9f155b2...`: RunPod Secure RTX 4090, PyTorch
2.5.1+cu124/CUDA 12.4, with all four representative CUDA cases passing. That gate proves
compatibility, not long-run throughput.

Read-only CPU evidence projects the complete GPU route at 6-12 hours, with important
uncertainty because bigint sampling/labels, rollout bookkeeping, paired-vector hashing,
MLflow/checkpoint I/O, and the isolated candidate runtime remain partly CPU-bound. The first
authorized slice is therefore crash-resumable: at most 10 provider hours and $9 total. At
the observed $0.74 compute plus a conservative $0.010795/hour combined container and
50 GB network-volume storage rate, its lifecycle upper bound is $7.50795, below the
controller's $7.65 post-reserve budget. Reaching the paid slice boundary is not a scientific
stop: authenticated EXP701 state is exported, the Pod is deletion-confirmed, and the same
GPU root can resume only after later authorization.

The guest writes each atomic recovery triplet beneath `/workspace/exp701-durable` on a
controller-created network volume. Science cannot start until an external host has pulled
the first triplet over strict-host-key-checked SSH, verified its hash and full restoration,
and returned a session-bound acknowledgement. The host repeats this at five-minute
intervals; a stale acknowledgement checkpoints the guest. Focused testing deleted the
entire simulated disposable Pod/mount tree, restored solely from the external copy, and
proved exact same-root resumed-versus-uninterrupted signatures. The controller deletes the
Pod first and deletes the network volume only after verifying the external restore receipt.
This external copy remains mandatory because RunPod documents that a zero balance can
terminate unrecoverable Pods without network volumes and can eventually remove billed
network volumes as well.

The exact preflight, cost model, artifact paths, and authorization value are frozen in
`experiments/cloud_benchmarks/exp701_runpod/decision-card.yaml`. Provider credentials remain
external; no answers or training/recovery/provider dependencies enter `submission/`.
