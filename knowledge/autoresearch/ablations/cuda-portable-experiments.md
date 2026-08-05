---
type: ablation
title: CUDA-portable experiment adapters
description: Explicit device runtime, lifecycle-closed inventory, and representative graph/recurrent replay adapters.
status: active
readiness: cuda-provider-startup-blocked
timestamp: "2026-08-05T06:57:08+00:00"
---

# CUDA-portable experiment adapters

This branch preserves historical experiment runners and introduces explicit
device adapters for current hardware replays. The inventory is mechanically
closed through exact registry lifecycle evidence, and the environment,
acceptance, and provider-control packages are locally prepared. This is not a
claim that CUDA has executed successfully.

## T003 boundary

- `experiments/cuda_portability/runtime.py` resolves only CPU/CUDA devices,
  configures deterministic execution, and never changes PyTorch's global
  default device.
- `experiments/cuda_portability/exp300.py` preserves EXP300's Python-bigint
  sampler and exact-label semantics while preparing tensors on an explicit
  device.
- Exact trajectory labels are prepared before the measured recurrent model
  loop. The EXP699 replay converts rollout results to host values only after
  the loop, so semantic evidence remains available without a synchronization
  at every position.
- `experiments/cuda_portability/inventory.json` is generated from every
  `experiments/**/*.py` file. Unported torch routes remain
  `adapter-required`; they are not silently treated as ready or archived.

## Evidence and limitations

CPU parity covers EXP300 feature/loss behavior, prepared training labels, one
frozen-route training update, and a deterministic rollout comparison. Static
inspection enforces the no-transfer recurrent-loop boundary. A real GPU has
not run this code, so kernel support, CUDA memory use, numerical agreement, and
speedup are hypotheses reserved for the later authorization-gated acceptance
matrix.

Regenerate and check the manifest with:

```bash
python -m experiments.cuda_portability.inventory --write
python -m experiments.cuda_portability.inventory --check
```

No long local experiment is required by this package.

## T004 representative families

The EXP510-selected typed-max PyG model is wrapped by a device edge cache. CPU
topology is prepared once per batch/width/dilation/device, then the message-
passing recurrence consumes device-resident `edge_index` and `edge_attr`
without `.to(...)` inside the loop. The adapter covers only the typed-max route;
the frozen module's GCN and GATv2 variants remain `adapter-required`.

The EXP654 neural-RAM adapter replaces the frozen CPU-only `bits` and
`static_features` preparation boundary and precomputes recurrent position
features outside the recurrence. CPU outputs and all diagnostic tensors match
exactly in the bounded tests.

These are structural portability and CPU-parity results. They do not show that
PyG CUDA kernels are installed, that either family fits a rented GPU, or that a
GPU improves end-to-end time.

## T005 adversarial correction and T006 repair

The T004 statement that EXP654's recurrence was fully device-native was
incomplete. Its frozen `forward` calls `position_features` per step, and that
helper constructs `torch.tensor` from Python scalars on the selected device.
On CUDA this is a small but repeated host-to-device boundary that a
method-call-only `.to/.cpu` audit misses.

The portable EXP654 adapter now precomputes the complete position-feature
matrix once with the other inputs, then executes the same frozen layers through
`neural_ram_forward_prepared`. CPU logits and every diagnostic tensor match for
all four memory ablations. The inventory audit now rejects configured
Python-backed factories (`tensor`, `as_tensor`, and `from_numpy`) inside a
declared loop and follows same-file helper calls; a regression test proves it
detects the frozen `forward -> position_features -> torch.tensor` path. This
repair remains hardware-unvalidated.

## T012 lifecycle closure

The generated inventory now classifies all 235 discovered experiment Python
files without an `adapter-required` remainder:

- 9 current adapter/runtime files or fully covered active routes are
  `device-native-unvalidated`;
- 5 orchestration or frozen stochastic-contract files are
  `intentionally-cpu-boundary`; and
- 221 files are `archived-frozen-route` because every exact registry route
  that owns or consumes them has a terminal allowlisted status and at least one
  hash-bound durable artifact.

This is a route-specific disposition, not a default for old-looking paths. The
generator resolves exact `exp_NNN`, `candidate_NNN`, and durable run-directory
identities, then folds in import reachability. It records each run id, exact
registry status, current artifact SHA-256, and the registry SHA-256 in the
manifest. Superseded EXP648 source snapshots are therefore evidence, while the
fully covered EXP699 source remains assigned to its CUDA replay adapter.

EXP649 is the one registry-active (`planned`) experiment route. Its runner
dynamically loads the shared EXP220 model, so that relationship is declared and
hash-checked rather than guessed from static imports. Both files remain on CPU:
the frozen protocol requires `iid_cpu_float32_gaussian`, an exact logical
coordinate traversal, and generator resets before warmup, after warmup, before
each replicate, and before the exact rerun. The authenticated calibration record
says the runtime gate failed, formal work was not authorized, and the formal
root was never opened. Moving this route to CUDA would be a new experiment, not
a portability refactor.

The closure authenticates EXP649's config, runner, seed-retirement manifest,
and run record by SHA-256. Physical CUDA execution, CUDA dependency resolution,
VRAM limits, kernel availability, and speedup remain unobserved and are deferred
to the authorization-gated GPU acceptance package.

## T007 locked environment and acceptance harness

The centralized `environments/cuda` package defines Linux x86_64/Python 3.12
profiles for PyTorch 2.5.1 with CUDA 12.1 and CUDA 12.4. Each profile has a
124-package transitive requirements lock generated by uv 0.11.28 with hashes
for every artifact. Torch is pinned to an HTTPS direct wheel under its matching
PyTorch index; all other packages use the explicit `https://pypi.org/simple`
default. The lock SHA-256 values are:

- cu121: `817f01d626ed840d4a8912968229d87450ba60da7d2cb42552fccccfca9c4f70`
- cu124: `2ef0ecd02324eb9d840d06000dfa3f11d7b6f75223796db95188edbf71278355`

The fresh-VM bootstrap does not assume uv or Python. It accepts only Linux
x86_64, hash-checks the selected lock, verifies `nvidia-smi`, driver and compute
capability before installing, and either reuses exact uv 0.11.28 or downloads a
versioned installer whose SHA-256 is pinned. It then installs a uv-managed
Python 3.12, syncs with `--require-hashes`, runs `uv pip check`, verifies the
exact torch/CUDA build and live GPU, and writes an environment receipt. An
existing unmarked environment directory is rejected instead of overwritten.

The hardware command is also fail-closed. It requires explicit `--require-cuda`,
reruns live preflight, and executes four exact GPU tests: runtime plus an EXP699
train-update/rollout path, EXP510 typed-max PyG forward/backward with cached GPU
edges, EXP654 neural-RAM forward/backward with prepared inputs, and direct
safetensors GPU checkpoint loading. Its JUnit parser rejects missing, failed,
errored, skipped, or extra cases, and the receipt records commit/config/test/
lock identities, runtime properties, peak CUDA allocation/reservation, and
wall time.

Observed locally: both generated locks pass the offline integrity verifier;
bootstrap and launcher shell syntax pass; bootstrap dry-run creates no
environment; acceptance refuses CPU fallback; and the real-hardware tests are
skipped unless launched through the explicit CUDA command. No CUDA environment
was installed, no GPU ran, and no cloud resource was created. The current
inventory therefore has 236 files (10 device-native-unvalidated, 5 intentional
CPU boundaries, and 221 archived routes) and remains
`cuda-inventory-closed-gpu-unvalidated`.

## T008 external billing controls

The provider controller is deliberately outside the guest VM. RunPod and Vast
adapters use their documented Bearer-authenticated REST create/status/delete
lifecycles, but repository verification injects a fake transport and never
contacts either service. Plan mode does not read credentials and records only a
SHA-256 of the proposed guest command.

Admission requires explicit compute and storage hourly rates, a total-dollar
cap, and a provider-lifetime cap. It reserves 15% headroom and computes the
upper bound from create request through confirmed deletion, so provisioning and
cleanup are charged as well as guest runtime. If the provider later reports a
higher hourly rate, the controller immediately enters deletion. Provider
lifecycle time, guest running time, and the as-yet-unavailable provider invoice
amount are separate receipt fields.

Credentials stay in the external Authorization header. Guest payloads reject
provider credential variable names, authorization markers, and the actual token.
After creation, the resource id is persisted immediately. Guest exit, stop,
timeout, rate mismatch, SIGINT/SIGTERM, and ordinary failures all flow through a
retrying DELETE in `finally`; a stop or exited state is never accepted as cost
closure. Unit evidence covers both endpoint shapes, over-budget rejection,
credential rejection, price increase, cleanup retry, deletion confirmation, and
secret-free receipts.

This reduces ordinary runaway-billing risk but cannot guarantee cleanup after
client power loss, prolonged network loss, or an uncatchable process kill.
Operator controls remain mandatory before T010: prepaid balance, auto-recharge
disabled, provider alerts/limits, an exact offer rate, a hard launch cap, and
separate confirmation that the provider invoice/resource list is clean. No live
provider request was made in T008.

## T009 clean-worktree gate

The repository deliberately excludes historical `.scratch` artifacts from
Git, while the complete regression suite authenticates selected archived
evidence. A clean worktree must therefore restore that external dependency
before running the full suite. `tools/restore_external_test_fixtures.py` uses a
closed root allowlist, refuses a source inside the target worktree, never
overwrites divergent evidence, and writes an ignored restoration receipt.

Most large, read-only historical roots are explicit links to the operator's
existing evidence. Five roots whose tests enforce repository-local path or
physical-file integrity are independent copies: EXP658, EXP663, EXP675,
EXP685, and MLflow. The primary source is the development worktree's
`.scratch`; a physical historical overlay is also required because later
archives contain evidence absent from the current top-level MLflow store. For
the current workspace the restoration command is:

```bash
python tools/restore_external_test_fixtures.py \
  --repository . \
  --source /mnt/d/Projects/SAIR_mod_challenge/.scratch \
  --source /mnt/d/Projects/SAIR_mod_challenge/.scratch/exp-651-visit-dilation-adapters/.scratch
```

This mechanism does not regenerate data, run training, or make ignored
artifacts durable; it only supplies the already-existing fixture dependency to
the clean validation worktree. The branch readiness is now
`cuda-prepared-hardware-unvalidated`: local source, lock, mocked-provider, and
CPU parity gates are prepared, but a real supported NVIDIA GPU is still needed
for the four-case acceptance matrix and any speed claim.

The first complete pass also exposed test-process contamination: EXP699's dry
run configured PyTorch's one-shot global inter-op thread setting, causing a
later frozen EXP671 guard test to fail before reaching its calibration check.
Dry run now performs admission and evidence capture without mutating the torch
runtime, and the one-update CPU parity test executes its faithful frozen
thread setup in a child process. The paired EXP699-then-EXP671 regression passes
without modifying either frozen scientific runner.

## T010 authorized RunPod hardware attempt

The user authorized one Secure Cloud RTX 4090 with a $1.00 total cap and a
30-minute provider-lifetime cap. The external controller observed the live
offer at $0.74/hour, rejected an initially admitted $0.69/hour rate, and
confirmed deletion before readmitting the actual price. Four billable Pods
were created sequentially, never concurrently. Every Pod was deleted and the
final API inventory was zero Pods, templates, endpoints, and network volumes.
The create-to-delete upper bounds total 1,014.301608578 seconds and $0.210378;
the console balance changed from $10.00 to $9.80, while the billing API had not
yet posted invoice records.

No acceptance compute ran. On two Secure 4090 hosts the pinned old image
resolved to the expected digest
`sha256:2594f25beeef751aeebcc207ba4c45a382ba174655f01e15cbbfad0e2bdb5ef8`,
but RunPod system logs stopped at `start container ...: begin` with no guest
logs or usable shell. A temporary account SSH key was added only to diagnose
the official entrypoint and was removed at cleanup. The current official image
was resolved to OCI index
`sha256:60baa36d3fb6b98fd4f4ece6b96776c83c01a8b7c540e54460ab4d496816141f`
and linux/amd64 manifest
`sha256:3e874356857adfa3e8faa3fd913b65bd127f77a0fe2e489513e7775e1c1e16b1`,
but RunPod rejected both create requests before resource allocation with its
generic HTTP 500.

The durable run record is
`experiments/runs/cuda-hardware-acceptance-runpod-20260805/run.json`. This is a
provider-startup failure receipt, not CUDA evidence: uv and the cu124 lock were
never installed, all four hardware tests remain unexecuted, and semantic
parity, VRAM, utilization, and speedup remain unknown. Readiness is therefore
`cuda-provider-startup-blocked`, a more specific hardware-unvalidated state;
neither `cuda-hardware-validated` nor `plug-and-play` is permitted.

## T011 successful RunPod hardware acceptance

The corrected archive runner at commit
`02678914d6112a6ff12030b3c05be89ea12c7b5d` completed the pinned cu124
bootstrap on a RunPod Secure RTX 4090 and passed all four required hardware
cases. The environment receipt observed PyTorch `2.5.1+cu124`, CUDA runtime
12.4, driver `580.159.04`, compute capability 8.9, and 24,564 MiB of device
memory. The four-case pytest matrix passed in 7.92 seconds and the acceptance
receipt completed in 11.163757524918765 seconds; the exact-archive guest
lifecycle, including first-time environment setup, was 122.194056918 seconds.

The successful Pod was deleted after a provider-lifetime upper bound of
228.5337558629999 seconds. Its cost upper bound was $0.04735727274272165;
including every authorized diagnostic and setup attempt, the conservative
series upper bound was $0.7932961325195431, below the $1.00 authorization.
Final read-only inventory observed zero Pods. The durable receipt is
`experiments/runs/cuda-hardware-acceptance-runpod-20260805g/run.json`.

Readiness is now `cuda-hardware-validated-acceptance-only`: the representative
CUDA families and checkpoint round trip are operational, but no long training
experiment has run, so no speedup, throughput, utilization, or representative
peak-VRAM claim is permitted yet.
