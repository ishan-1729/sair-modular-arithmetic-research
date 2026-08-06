---
type: Operations Runbook
title: EXP701 RunPod Resumable GPU Execution
description: Fail-closed operating procedure for preparing, launching, monitoring, checkpointing, recovering, billing, and closing the full-fidelity EXP701 CUDA experiment on RunPod without duplicate resources or lost scientific state.
status: active
tags: [runbook, exp701, runpod, cuda, gpu, checkpoint, recovery, billing, orchestration]
timestamp: "2026-08-07T04:30:00+05:30"
---

# EXP701 RunPod resumable GPU execution

## Purpose

Use this runbook for every paid RunPod execution or resume of
`exp-701-cuda-resumable-h92-w256-backbone-factorial`. It turns the working
implementation and the failures observed in attempts J through L into one
operator contract.

The priorities are ordered:

1. never leak credentials or benchmark answers;
2. never create a duplicate paid resource;
3. never lose the newest scientific checkpoint;
4. never mistake guest exit, Pod stop, or a thread heartbeat for confirmed
   provider cost closure;
5. preserve the exact preregistered scientific identity;
6. spend only inside an explicit dollar and provider-time authorization;
7. maximize useful scientific updates per billed provider second;
8. leave compact receipts so the next agent does not reconstruct state from
   chat history or raw logs.

This is an operations runbook, not authorization. Reading or following it does
not permit a paid launch. The exact approval and admitted ceilings must be
present before each newly funded tranche.

## Canonical implementation and evidence

Read these files before acting:

- [`decision-card.yaml`](../../experiments/cloud_benchmarks/exp701_runpod/decision-card.yaml)
  freezes scientific identity, roots, denominators, payload policy, and the
  original resource model.
- [`prepare_source.sh`](../../experiments/cloud_benchmarks/exp701_runpod/prepare_source.sh)
  builds the explicit allowlisted training payload.
- [`orchestrate.py`](../../experiments/cloud_benchmarks/exp701_runpod/orchestrate.py)
  is the host coordinator and duplicate-resource gate.
- [`guest_supervisor.sh`](../../experiments/cloud_benchmarks/exp701_runpod/guest_supervisor.sh)
  owns bootstrap, CUDA acceptance, smoke verification, conditional science,
  slice termination, export, and final acknowledgement.
- [`offpod_sync.py`](../../experiments/cloud_benchmarks/exp701_runpod/offpod_sync.py)
  copies an atomic archive/checksum/receipt generation, restore-verifies it,
  and publishes the matching acknowledgement.
- [`recovery_transport.py`](../../experiments/cloud_benchmarks/exp701_runpod/recovery_transport.py)
  exports, verifies, and restores checkpoint archives.
- [`recover_retained_volume.py`](../../experiments/cloud_benchmarks/exp701_runpod/recover_retained_volume.py)
  is a narrow retained-volume recovery tool. It is not the normal cleanup
  command and it must not run while any Pod exists.
- [`T019-attempt-l-live-handoff.md`](../../docs/goals/cuda-portable-experiments/notes/T019-attempt-l-live-handoff.md)
  records the latest exact resume boundary.
- [`attempt-l-slice-recovery-and-cleanup.json`](../../experiments/runs/exp-701-cuda-resumable-h92-w256-backbone-factorial/raw/attempt-l-slice-recovery-and-cleanup.json)
  records the terminal attempt-L checkpoint and cleanup.
- [`runpod-billing-20260805-20260807.json`](../../experiments/runs/exp-701-cuda-resumable-h92-w256-backbone-factorial/raw/runpod-billing-20260805-20260807.json)
  separates provider-recorded charges from controller upper bounds.

`state.yaml` is GoalBuddy board truth. The files above are execution evidence;
chat summaries and heartbeats are not authority.

## Frozen scientific contract

Do not change these values to make a resume cheaper or faster:

- topology: H92/two-cycle, 99,824 parameters;
- arms: teacher/straight-through crossed with cosine/restart;
- updates: 4,800 per arm, 19,200 formal training updates total;
- batch: 16; horizon: 8;
- transition, rollout, diagnostic, confidence, and promotion denominators;
- source checkpoint SHA-256:
  `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`;
- calibration root: `20261413`;
- formal root: `20261414`;
- confirmation roots: `20261415`, `20261416`;
- independent roots: `20261417` through `20261421`;
- device delta: CUDA plus the pinned CUDA runtime only;
- CPU partials: preserve as historical evidence, never restore them into a GPU
  root.

The conditional route is important. Completion of formal root `20261414` can
open two confirmation roots. If both confirmation gates trigger, five
independent roots open. Never estimate the whole job from the formal training
updates alone without stating this branch.

## Host environment

Run routine commands through Ubuntu 24.04 WSL from the existing worktree. Do
not copy the repository to another filesystem.

```bash
cd /mnt/d/Projects/SAIR_mod_challenge/.scratch/exp-651-visit-dilation-adapters
source /home/ishan/.venvs/sair-mod-challenge/bin/activate
```

Credential files are host-only:

- RunPod REST: `/home/ishan/.config/sair/runpod.env`
- RunPod S3: `/home/ishan/.config/sair/runpod-s3.env`

Never print their values, persist them in a receipt, put them in a guest
command, include them in the source archive, or copy them into `submission/`.
To inspect only names:

```bash
sed -n 's/=.*//p' /home/ishan/.config/sair/runpod.env
sed -n 's/=.*//p' /home/ishan/.config/sair/runpod-s3.env
```

### CRLF credential pitfall

The S3 file was written from Windows and carried carriage returns. Sourcing it
unchanged produced an invalid AWS authorization header containing `\r`. Export
normalized values without modifying or echoing the file:

```bash
set -a
source <(sed 's/\r$//' /home/ishan/.config/sair/runpod.env)
source <(sed 's/\r$//' /home/ishan/.config/sair/runpod-s3.env)
set +a
```

`source file` without `set -a` leaves non-`export` S3 assignments invisible to
Python. This caused a second avoidable failure. Normalize and export both.

## The three cost ledgers

Never collapse these into one number.

### 1. Authorization ledger

This is the hard operator ceiling. Admission must include all previously
completed attempts plus the proposed lifecycle:

```text
cumulative provider seconds
  = completed conservative provider seconds
  + proposed create-to-delete ceiling

cumulative cost upper
  = completed conservative cost
  + proposed seconds * (Pod rate + all storage rates) / 3600
```

Reserve time for create, bootstrap, external synchronization, terminal export,
acknowledgement, delete retries, and independent absence confirmation. Guest
science time is strictly smaller than the controller ceiling.

### 2. Controller ledger

`provider-lifecycle-*.json` is controller-observed create-to-delete evidence.
It is an upper-bound operations receipt, not an invoice. A non-terminal status
such as `controller-started` means the controller did not prove cleanup. Never
edit it to look terminal after a controller crash; create a separate recovery
and cleanup receipt.

### 3. Provider billing ledger

RunPod's billing endpoints are the provider-recorded usage source:

- `GET /v1/billing/pods`
- `GET /v1/billing/networkvolumes`

The REST OpenAPI exposes usage history but not the current `clientBalance`.
Use the Billing console or `runpodctl user` for the exact balance. A remaining
balance derived from initial credits minus billing history must be labeled an
estimate and list its assumptions.

### Attempt-L billing reconciliation

For the 2026-08-05 through 2026-08-07 window, RunPod reported:

| Item | Provider-recorded USD |
|---|---:|
| All Pods | 7.3655502368 |
| All network volumes | 0.0583333354 |
| Total | 7.4238835723 |
| Attempt-L Pod `vl7erkqk6z29ev` | 5.3146533056 |
| Other Pods | 2.0508969313 |

If the account began at exactly `$10.00` and had no unrelated usage or
adjustments, the estimated balance is `$2.5761164277`. This is not an exact
`clientBalance` reading.

Attempt L was billed for `25,612.953` Pod seconds. Comparing this with the Pod
creation-to-terminal-export interval attributes approximately `6,670.943`
seconds and `$1.3842` of Pod cost to the period after the final science export.
RunPod did not itemize that split; it is a calculation from provider billing.

## Performance reality

The original `6–12 hour` GPU projection was a planning hypothesis and proved
too optimistic. Measured attempt-L results are:

| Scope | GPU wall | Local/historical wall | Measured speedup |
|---|---:|---:|---:|
| Calibration route | 240.634 s | 549.480 s | 2.28x |
| `teacher_cosine` training, 4,800 updates | 8,381.822 s | 17,279.228 s | 2.06x |
| Primary 128+256 evaluation | 2,393.649 s | 3,857.472 s | 1.61x |

The local full-route baseline is approximately `158,945` seconds, or `44.15`
hours. A roughly 2x route does not turn that into a sub-10-hour job; it suggests
about 22 hours before uncertainty in conditional roots and orchestration. The
model is only about 100k parameters, while Python big-integer sampling, exact
labels, rollout bookkeeping, decoding, serialization, hashing, MLflow, and
candidate runtime remain partly CPU-bound. The RTX 4090 is not the bottleneck
for every phase.

At the final attempt-L checkpoint, formal training was `7,800 / 19,200`
updates (`40.625%`): `teacher_cosine` was complete and evaluated;
`teacher_restarts` was at `3,000 / 4,800`; the two straight-through arms had
not started. Do not claim experiment completion from the amount spent or from
one completed arm.

Before any future launch, project time from measured per-phase evidence and
state explicitly whether the estimate covers formal only or the maximum
conditional path.

## Prelaunch gate

No paid mutation occurs until every checkbox passes.

### Identity and source

- [ ] Current branch and committed source SHA are recorded.
- [ ] Worktree is clean except explicitly preserved unrelated files.
- [ ] Challenge commit is
      `82510bba00a1126649bd76dd1a451f14d0b3eb60`.
- [ ] Source checkpoint hash matches the frozen value.
- [ ] `prepare_source.sh` succeeds from committed bytes.
- [ ] Archive checksum, payload inventory, repository commit, executable bits,
      and forbidden-prefix audit pass.
- [ ] Resume archive triplet passes SHA-256 and full restore verification.
- [ ] Resume tree contains the intended root/update and no rejected terminal
      failure.

Build the allowlisted source archive only from a clean commit:

```bash
bash experiments/cloud_benchmarks/exp701_runpod/prepare_source.sh \
  .scratch/exp701-launch/source-<attempt>.tar.gz
sha256sum -c .scratch/exp701-launch/source-<attempt>.tar.gz.sha256
```

Never replace this with a repository-wide tarball. The development tree
contains evidence and benchmark snapshots that must not enter the training
guest.

### Local gates

```bash
python -m pytest -q \
  tests/test_exp701_runpod_orchestrator.py \
  tests/test_cuda_provider_control.py \
  tests/test_exp_701_cuda_resumable_h92_w256_backbone_factorial.py
python tools/validate_project.py
git diff --check
```

Run full pytest when executable/config/test/dependency bytes changed or at the
publication boundary. Reuse an exact-identity full-suite receipt after
documentation-only changes.

### Provider and authorization gates

- [ ] Exact explicit approval names provider, GPU, additional dollar ceiling,
      and additional provider lifetime.
- [ ] Auto-recharge is disabled unless the operator explicitly wants it.
- [ ] Current credit covers at least the admitted lifecycle plus emergency
      deletion headroom.
- [ ] Authenticated inventory contains zero Pods and zero network volumes.
- [ ] A fresh Secure RTX 4090 offer exists in the selected data center.
- [ ] Actual rate is no higher than the admitted rate.
- [ ] Proposed cumulative upper bounds fit both hard ceilings.
- [ ] Session, destination, provider receipt, and orchestrator receipt paths are
      unique and do not exist.
- [ ] Exactly one source archive and one resume archive are selected by hash.

An allocation failure can create a volume without creating a Pod. Always
delete-confirm that volume before retrying under a new session suffix.

## Dry run and launch

First render the complete nonpaid orchestration plan:

```bash
python -m experiments.cloud_benchmarks.exp701_runpod.orchestrate \
  --data-center-id EU-RO-1 \
  --source .scratch/exp701-launch/source-<attempt>.tar.gz \
  --preflight experiments/runs/exp-701-cuda-resumable-h92-w256-backbone-factorial/preflight-<attempt>.json \
  --resume-receipt .scratch/exp701-offpod-<previous>/offpod-sync-receipt.json \
  --session .scratch/exp701-runpod-session-<suffix> \
  --offpod-destination .scratch/exp701-offpod-<suffix> \
  --provider-receipt experiments/runs/exp-701-cuda-resumable-h92-w256-backbone-factorial/raw/provider-lifecycle-<suffix>.json \
  --orchestrator-receipt experiments/runs/exp-701-cuda-resumable-h92-w256-backbone-factorial/raw/orchestrator-lifecycle-<suffix>.json \
  --dry-run
```

Review the emitted hashes, rate, source, resume tree, slice seconds, controller
seconds, storage, and cumulative admission. Remove only `--dry-run` after the
approval and all gates pass.

Do not paste an old launch command blindly. Bind a new unique suffix and fresh
preflight to the exact resumed archive.

## Lifecycle ownership

The provider controller is the sole paid-resource owner. The process chain is:

```text
Windows owner / Codex host
  -> Ubuntu WSL orchestrator
     -> provider controller
     -> off-Pod S3 synchronizer
        -> one RunPod Pod + one managed network volume
```

Record the Windows owner and all WSL PIDs immediately after launch. A Codex
heartbeat is notification only; it does not keep WSL alive and it is not a
cleanup controller.

### Laptop and WSL survival rule

Attempt L proved that a correct guest and checkpoint system can still waste
money when the host owner disappears. Before the next paid launch, satisfy one
of these controls:

1. preferred: install a tested Windows-native deadline watchdog that can query
   and delete the exact admitted Pod/volume even when WSL exits;
2. otherwise: keep the Windows session awake, disable sleep/hibernate for the
   admitted window, keep Codex and WSL running, and set a human deadline alarm
   before the provider ceiling.

The independent watchdog must be ID-bound, deadline-bound, and delete-only. It
must never launch or select resources. Do not call an untested watchdog safety.

Prepaid balance and disabled auto-recharge remain a backstop, not lifecycle
ownership.

## Guest gates and phase order

Current-attempt evidence must pass in this order:

1. source archive and source checkpoint hashes;
2. initial export and matching restore-verified off-Pod acknowledgement;
3. pinned `cu124` bootstrap;
4. environment receipt: uv, Python, PyTorch, CUDA runtime, determinism, TF32;
5. RTX 4090 hardware acceptance, 4/4;
6. focused EXP701 tests, 14/14;
7. exact restored-smoke verifier;
8. pre-science export and matching acknowledgement;
9. calibration root `20261413`;
10. formal root `20261414`;
11. conditional confirmations and independent roots only if preregistered
    triggers open them.

Restored evidence and current-attempt execution are different. A restored J
archive can prove prior smoke state; it does not prove the new attempt ran its
environment, hardware, or focused gates. Record both explicitly.

## Off-Pod publication protocol

The recoverable object is a generation triplet:

```text
sair-exp701-checkpoint-export.tar.gz
sair-exp701-checkpoint-export.tar.gz.sha256
sair-exp701-checkpoint-export.tar.gz.receipt.json
```

`offpod_sync.py` downloads checksum and receipt, fetches the matching archive,
verifies the triplet, fully restores it into a temporary directory, validates
the tree, atomically installs the local generation, writes
`offpod-sync-receipt.json`, and only then publishes the nonce-bound ack.

### Publication-race rule

Attempt J saw a newly rotated archive while the external receipt still
described the previous generation. The old monitor treated the transient
mismatch as corruption and killed a healthy pre-science guest.

Never compare independently sampled archive and receipt paths as if they are
one generation. A mismatch during rotation is retryable while the guest and
publisher are healthy. It becomes terminal only after a bounded settle window
or an explicit terminal failure. Use `offpod_sync.py`; do not invent a second
triplet reader in monitoring code.

### Restored-smoke idempotency rule

Attempt K reran an interruption smoke whose immutable phases were already
complete in the restored archive. The expected synthetic interruption could no
longer occur, so the attempt failed before science.

On resume, reuse completed smoke only after `pre_science.py` verifies the exact
identity and artifacts. Do not rerun immutable completed smoke phases. Reject
an archive containing a pre-root terminal failure even when its tree itself is
restore-valid.

## Event-driven monitoring

Monitor one lifecycle, not the provider generally. A healthy observation
contains:

- exact Pod ID, volume ID, source SHA, source commit, resume SHA, and suffix;
- live Windows owner and WSL orchestrator/controller/sync chain;
- provider state and accepted hourly rate;
- newest off-Pod receipt timestamp, archive SHA, bytes, and restore status;
- current root, arm, update, completed phase count, and terminal marker;
- elapsed provider seconds and remaining admitted headroom;
- GPU evidence from environment/hardware receipts and phase CUDA memory metrics.

Report only a material phase, arm, immutable checkpoint, terminal, failure, or
cleanup transition. Do not repeatedly scan full logs or launch another process
because output is quiet.

Useful archive probes are:

```bash
tar -tzf <archive> | grep -E \
  'guest-terminal|root-summary|progress.json|latest.json$|\.manifest\.json$'

tar -xOzf <archive> \
  experiment/exp-701-cuda-resumable-h92-w256-backbone-factorial/guest-terminal.json
```

Prefer the restore receipt's exact snapshot table over guessing from file
mtime. A mutable `latest.json` may be newer than the last immutable
`update-*.safetensors`; label both correctly.

## Terminal protocol

Guest terminal states are:

- `completed`, `resume_required: false` — science route completed;
- `slice-checkpointed`, `resume_required: true` — valid same-root resume;
- `failed`, `resume_required: false` — terminal guest failure;
- `offpod-*-sync-failed` or `final-offpod-sync-failed` — preserve resources or
  recover externally until a verified copy exists.

Do not treat the lifecycle as closed until all are true:

1. terminal export contains the newest recovery snapshot and
   `guest-terminal.json`;
2. external archive SHA matches its sidecar and receipt;
3. a fresh full restore/tree verification passes;
4. off-Pod ack matches the final archive and session nonce;
5. Pod DELETE is accepted and exact GET returns 404;
6. volume DELETE occurs only after step 3 and exact GET returns 404;
7. authenticated inventories return zero Pods and zero volumes;
8. provider, orchestrator, recovery, billing, and limitations receipts are
   persisted.

Guest exit, stopped state, empty terminal output, controller process exit, and
network-volume detachment are not cost closure.

## Host-controller crash recovery

Use this path when the WSL process chain is absent but the exact Pod or volume
still exists.

### 1. Freeze decisions

- Do not launch a replacement.
- Record exact IDs and query their current status and rates.
- Identify the newest local restore-verified archive.
- Confirm the external S3 bucket is the admitted volume ID.

### 2. Rescue the newest S3 generation

Use a fresh destination and a new acknowledgement key. The command is read plus
ack write only; it does not launch a Pod.

```bash
set -a
source <(sed 's/\r$//' /home/ishan/.config/sair/runpod-s3.env)
set +a

python -m experiments.cloud_benchmarks.exp701_runpod.offpod_sync \
  --s3-bucket <volume-id> \
  --s3-endpoint https://s3api-<region>.runpod.io/ \
  --remote-archive exp701-durable/sair-exp701-checkpoint-export.tar.gz \
  --remote-ack exp701-durable/offpod-emergency-ack.json \
  --nonce <unique-at-least-16-character-nonce> \
  --destination .scratch/exp701-offpod-<suffix>-final \
  --poll-seconds 30 --max-seconds 300 --once
```

Require `status: verified` and restore verification `status: verified` before
deletion.

### 3. Delete in order

1. `DELETE /v1/pods/<exact-pod-id>`;
2. poll `GET /v1/pods/<exact-pod-id>` to HTTP 404;
3. `DELETE /v1/networkvolumes/<exact-volume-id>`;
4. poll its exact GET to HTTP 404;
5. list both inventories and require zero.

Persist HTTP statuses, probe counts, timestamps, IDs, archive hash, and final
inventory in a separate emergency-cleanup receipt. Never rewrite the stale
controller receipt.

### Retained-volume recovery tool boundary

`recover_retained_volume.py` requires zero Pods and exactly one retained
volume, then launches a short bounded recovery Pod to mount that volume. Use it
only when S3 recovery is unavailable and the original Pod is already gone. It
must never run while the original Pod exists and must never start science.

## Billing reconciliation

After cleanup, query provider billing before reporting balance or cost. Example
logic:

```python
from experiments.cloud_benchmarks.exp701_runpod.orchestrate import api_get

window = {
    "startTime": "<UTC start>",
    "endTime": "<UTC end>",
    "bucketSize": "hour",
}
pods = api_get(token, "/billing/pods", {**window, "grouping": "podId"})
volumes = api_get(token, "/billing/networkvolumes", window)
```

Persist:

- query window and observation time;
- total Pod and network-volume amount;
- target Pod amount and `timeBilledMs`;
- all other Pod amount;
- controller upper bound separately;
- exact balance from console/`runpodctl user`, or a clearly labeled estimate;
- any post-terminal cost attribution as a calculation, not provider itemization.

## Attempt failure ledger

| Attempt | Observed failure | Useful evidence preserved | Prevention |
|---|---|---|---|
| J | Archive rotated before external receipt; monitor treated transient generation mismatch as corruption and terminated pre-science. | CUDA environment, hardware 4/4, focused 14/14, smoke, restore-verified pre-science archive; no root opened. | Use one atomic triplet reader, retry rotation mismatch, terminate only after settle/terminal evidence. |
| K allocation | Selected machine returned HTTP 500 for capacity. | No Pod; volume cleanup receipt. | Treat capacity as allocation failure, delete volume, requery availability, use a new suffix; never reuse receipt paths. |
| K live | Restored immutable smoke was rerun and could not reproduce the synthetic interruption. | Hardware/focused gates and restore-valid failure export; no root opened. | Exact idempotent restored-smoke verifier; reuse completed immutable smoke; reject terminal-failure archive. |
| L allocation (`s3g`) | No instances available after managed volume creation. | Exact allocation receipt; no Pod/root. | Delete-confirm orphan volume before retry; new lifecycle suffix. |
| L science (`s3h`) | Science cleanly slice-checkpointed, then WSL/controller ownership vanished before cleanup; Pod remained desired RUNNING. | Calibration, complete/evaluated teacher-cosine, teacher-restarts update 3000, restore-verified terminal archive. | Independent Windows deadline watchdog, host no-sleep policy, immediate final-archive rescue and ID-bound deletion. |

Do not describe these attempts collectively as GPU failure. The RTX 4090,
CUDA environment, deterministic route, checkpoint restoration, and science
training worked. The failures were primarily control-plane, idempotency,
capacity, and lifecycle-ownership defects.

## Current resume boundary (2026-08-07)

This section is time-specific evidence, not a reusable default:

- branch: `codex/exp701-cuda-resumable-h92-w256-backbone`;
- source commit: `0027f66197a00ea9392e04cc7060c899500aa091`;
- source archive SHA:
  `574f8523f3ac78f7ecd537dfa83cfe642d711c1c885b6edad0899d9785409678`;
- final attempt-L archive:
  `.scratch/exp701-offpod-s3h-final-b/sair-exp701-checkpoint-export.tar.gz`;
- final archive SHA:
  `0a4ea76bfb3a9f6933008badf187206eff58c11c4ad111e5ddc2121861bc0370`;
- formal root: `20261414`;
- completed formal training: `7,800 / 19,200` (`40.625%`);
- active arm: `teacher_restarts`, update `3,000 / 4,800`;
- snapshot SHA:
  `e5e5337382024ff5ed1cf53765b3ac99aafbed8698832b807e7caccc4098ba15`;
- terminal: `slice-checkpointed`, `resume_required: true`;
- provider resources: zero Pods, zero network volumes;
- original cumulative authorization: exhausted for useful science even though a
  small formal headroom remains;
- new paid launch: forbidden until a fresh explicit authorization expands both
  dollar and provider-time ceilings.

Do not resume from attempt K, from the clean J pre-science archive, or from a
CPU partial. Resume only from the final attempt-L archive after verifying its
hash and tree again.

## Future-agent handoff template

Every live or terminal handoff must state all fields below. Omission means the
next agent performs read-only reconciliation before acting.

```text
Goal/task:
Branch and HEAD:
Source commit and source archive SHA:
Resume archive path and SHA:
Scientific identity SHA:
Current root / phase / arm / update:
Completed immutable phases:
Guest terminal state:

RunPod Pod ID / status / rate:
RunPod volume ID / status / rate:
Data center / GPU:
Windows owner and WSL process-chain PIDs:

Session suffix and paths:
Latest off-Pod receipt timestamp:
Latest archive SHA / bytes / restore status:
Final ack nonce match:

Completed provider seconds / USD:
Current admitted ceiling:
Remaining cleanup headroom:
Provider-billing observation time:

Exactly one next decision:
Forbidden actions:
```

## Final operator checklists

### Before create

- [ ] Exact source and resume archive restore-verified.
- [ ] Local focused gates pass.
- [ ] Empty authenticated Pod and volume inventory.
- [ ] Fresh capacity and rate checked.
- [ ] New unique session/receipt paths.
- [ ] Explicit approval and cumulative admission recorded.
- [ ] Prepaid balance, auto-recharge policy, and deadline alarms checked.
- [ ] Independent cleanup owner exists outside the WSL process chain.
- [ ] No CPU partial or rejected archive selected.

### While running

- [ ] One Pod and one volume only.
- [ ] Host owner, controller, and sync processes alive.
- [ ] Provider rate unchanged.
- [ ] Initial and pre-science ack gates passed.
- [ ] Newest off-Pod generation restore-verifies.
- [ ] Progress labels current-attempt evidence separately from restored evidence.
- [ ] Provider seconds and cleanup headroom remain inside admission.
- [ ] No duplicate launch on quiet output or capacity retry.

### At terminal

- [ ] Terminal archive rescued and restore-verified.
- [ ] Guest terminal state read from that archive.
- [ ] Exact newest snapshot/root/update recorded.
- [ ] Pod DELETE and 404 confirmed.
- [ ] Volume DELETE and 404 confirmed after restore gate.
- [ ] Full inventories empty.
- [ ] Provider billing queried and reconciled.
- [ ] Run/evaluation/registry artifacts written only if a terminal science result exists.
- [ ] Goal/automation monitor updated or deleted so it cannot act on stale IDs.

### If anything is ambiguous

Stop provider mutation. Preserve the current external archive, query exact IDs,
and resolve the ambiguity read-only. The safe default is no new resource, not a
replacement launch.
