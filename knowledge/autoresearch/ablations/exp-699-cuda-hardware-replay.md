---
type: Experiment
title: EXP699 CUDA hardware replay
description: Non-scientific device replay of the completed EXP699 teacher-cosine arm.
status: implemented-awaiting-cloud-measurement
tags: [autoresearch, exp699, cuda, hardware-replay, performance, cost]
timestamp: "2026-08-02T00:00:00+00:00"
---

# EXP699 CUDA hardware replay

This package measures whether a rented NVIDIA GPU is a defensible execution
platform for a future fresh EXP699 root. It does not migrate, resume, replace,
or reinterpret the consumed CPU formal root. The comparator is the already
completed teacher_cosine arm at root 20261396: 4,800 updates, 614,400 examples,
38,400 model calls, complete W128/W256 primary evaluation, complete paired
evidence, and 11,452.156460124002 scoped route wall seconds.

The source EXP699 runner remains frozen. A separate device adapter moves
features, targets, quotient labels, recurrent rollout state, and the model to
an explicit CPU or CUDA device while retaining the source sampler and Python
exact-label semantics. GPU timing synchronizes before and after every scoped
timer. PyTorch deterministic algorithms remain enabled; cuDNN benchmarking and
TF32 are disabled. The expected environment is Python 3.12, uv 0.11.28, and
PyTorch 2.5.1+cu124 from the explicit official CUDA 12.4 index. All other
dependencies remain on the pinned PyPI manifest.

The cheapest falsifier is CPU parity, not another long laptop run. A one-update
W128 replay must produce a byte-identical model checkpoint, data order, width
order, curves, and state hash relative to the frozen source function. The cloud
gate then executes one complete CUDA arm. Work identity, all generated-case
hashes, and all exact correct counts must match the completed CPU reference.
Cross-device checkpoint and prediction hashes are recorded but are not gates
because CPU and CUDA floating-point reductions need not be byte-identical.

CUDA is eligible for a future fresh scientific root only if those semantic
gates pass and the complete replay is at least 3x faster. Failure remains useful
timing evidence but cannot authorize scientific substitution. The replay has a
two-hour process deadline and a $5 compute ceiling. These controls stop the
process; they cannot delete a provider resource. The operator must therefore
use prepaid credit with automatic recharge disabled, keep provider credentials
outside the VM, arrange an external provider-side deletion watchdog, copy the
receipt out, and terminate rather than stop the Pod or instance.

## Cloud launch

From a fresh checkout, install the exact CUDA environment:

~~~bash
cd /workspace/SAIR_mod_challenge
bash experiments/cloud_benchmarks/exp699_cuda_replay/bootstrap.sh
~~~

Copy the EXP609 source checkpoint to the host without committing it, then run a
hash/environment-only preflight:

~~~bash
.venv-exp699-cuda/bin/python -m \
  experiments.cloud_benchmarks.exp699_cuda_replay.run \
  --profile full \
  --device cuda \
  --source-checkpoint /workspace/input/exp609-weights.safetensors \
  --hourly-rate-usd 0.29 \
  --dry-run
~~~

Only after preflight passes, remove --dry-run and execute through the outer
process watchdog:

~~~bash
EXP699_MAX_WALL_SECONDS=7200 \
  bash experiments/cloud_benchmarks/exp699_cuda_replay/run_guarded.sh \
  .venv-exp699-cuda/bin/python -m \
  experiments.cloud_benchmarks.exp699_cuda_replay.run \
  --profile full \
  --device cuda \
  --source-checkpoint /workspace/input/exp609-weights.safetensors \
  --hourly-rate-usd 0.29
~~~

Use the actual offer rate rather than the example $0.29/hour. Copy
preflight.json, receipt.json, and checkpoints away from ephemeral storage
before deleting the resource. A RunPod controller should execute "runpodctl
pod delete POD_ID"; a Vast controller should execute "vastai destroy instance
INSTANCE_ID". The controller must remain outside the rented host so cloud
credentials are never available to experiment code.
