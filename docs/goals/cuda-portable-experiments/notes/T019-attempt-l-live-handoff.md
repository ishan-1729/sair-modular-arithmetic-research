# T019: attempt L terminal slice handoff

Task: `T019`
Kind: `pm`
Status: `slice-checkpointed`, cleanup confirmed, fresh paid authority required

## Closed paid lifecycle

- RunPod Secure Pod: `vl7erkqk6z29ev`
- Managed network volume: `q0fir47rfw`
- Data center and accelerator: `EU-RO-1`, one RTX 4090
- Compute and storage rates: `$0.74/hour` and `$0.010795/hour`
- Host session: `.scratch/exp701-runpod-session-s3h`
- Restore-verified off-Pod destination: `.scratch/exp701-offpod-s3h`
- Provider and orchestrator receipts: `raw/provider-lifecycle-s3h.json` and
  `raw/orchestrator-lifecycle-s3h.json`
- Historical WSL owner/orchestrator, controller, and off-Pod sync PIDs:
  `463`, `466`, and `475`; all are now absent.

The WSL process chain stopped before the controller's `finally` cleanup. A
fresh authenticated inventory check found the exact Pod still in desired
`RUNNING` state. The newest network-volume export was copied and
restore-verified first; Pod `vl7erkqk6z29ev` and volume `q0fir47rfw` were then
deleted with HTTP `204`, each absence-confirmed with HTTP `404`, and a final
authenticated inventory returned zero Pods and zero volumes at
2026-08-06T22:48:02.922930Z.

## Immutable execution identity

- Branch: `codex/exp701-cuda-resumable-h92-w256-backbone`
- Source repository commit: `0027f66197a00ea9392e04cc7060c899500aa091`
- Source archive: `.scratch/exp701-launch/source-e.tar.gz`
- Source archive SHA-256:
  `574f8523f3ac78f7ecd537dfa83cfe642d711c1c885b6edad0899d9785409678`
- Clean attempt-J resume archive SHA-256:
  `0877289894ee77ab521078e2a072a31f3911e82b6ac759289c723814b570c4e1`
- Formal root: `20261414`
- Formal `teacher_cosine` identity SHA-256:
  `4652143d6ee35be631d8269d942900d486551fa235132f6cd502ad69e28721a9`

The guest is the exact pinned cu124 environment: uv `0.11.28`, Python
`3.12.13`, PyTorch `2.5.1+cu124`, CUDA runtime `12.4`, deterministic
algorithms enabled, and TF32 disabled.

## Completed gates and terminal science checkpoint

- CUDA bootstrap and exact 124-package environment: passed.
- RTX 4090 hardware acceptance: 4/4 passed.
- Focused tests: 14/14 passed.
- Restored recovery smoke: passed for all four modes with exact recovery.
- Calibration root `20261413`: completed and admitted formal execution.
- Calibration wall time: `240.6339715179` seconds versus the local
  `549.479805459`-second baseline (`2.28x` measured speedup).
- Formal root `20261414`: open.
- Latest externally restore-verified formal checkpoint at the
  2026-08-06T17:41Z observation: `teacher_cosine` update `4800`, snapshot
  SHA-256
  `a0badd6c22c1350e9023cb5bd05ae690a3d947ac6ccd3445b480f6a8ec06856f`.
- Corresponding off-Pod archive SHA-256:
  `a5308d46fcda1c3be8a1c084dc6fcac1037e77f4808d2df9c624f20ace31726f`.
- Restore verification: passed, 62 files, 8 completed phases, 4 snapshots,
  and both calibration/formal root markers.
- The first declared `w128_only_updates` segment is complete, and immutable
  model checkpoints exist at updates `400`, `800`, `1200`, `2400`, and
  `3600`, and `4800`. The exact update-4800 safetensors SHA-256 is
  `b2ef6691ce98195a857401692cbf94335ebeebd22d0e4077a2ee30d6669f715e`.

The terminal attempt-L archive completed both `teacher_cosine` evaluations and
then checkpointed `teacher_restarts` at update `3000 / 4800`. Formal training
progress is therefore `7800 / 19,200` updates (`40.625%`). The terminal
`teacher_restarts` snapshot SHA-256 is
`e5e5337382024ff5ed1cf53765b3ac99aafbed8698832b807e7caccc4098ba15`,
with the unchanged scientific identity
`4652143d6ee35be631d8269d942900d486551fa235132f6cd502ad69e28721a9`.

The final recovered archive is
`.scratch/exp701-offpod-s3h-final-b/sair-exp701-checkpoint-export.tar.gz`, SHA-256
`0a4ea76bfb3a9f6933008badf187206eff58c11c4ad111e5ddc2121861bc0370`.
It passed an outside-RunPod restore verification with 74 files, 10 completed
phases, 5 snapshots, both root markers, and a `guest-terminal.json` declaring
`slice-checkpointed` with `resume_required: true`. There is no terminal science
result yet.

The exact
training phase used 8,381.821873488 wall seconds, 8,367.69092 process CPU
seconds, 1,501,601,792 peak RSS bytes, 1,697,468,928 CUDA peak allocated bytes,
and 2,606,759,936 CUDA peak reserved bytes. Compared with EXP700's historical
17,279.227519224-second teacher-cosine training phase, this is a measured
`2.0615x` training speedup. The observed route still projects approximately
16--19 hours for the formal root including its full evaluation workload. The
maximum conditional science path contains the formal root, two confirmation
roots, and five independent roots; downstream roots open only if their
preregistered gates pass.

## Closed accounting and resume boundary

- Completed-attempt cumulative upper bound before s3h:
  `6267.559869931` provider seconds and `$1.3002306497822448`.
- Attempt-L conservative upper bounds: `28,524.688272238` provider seconds and
  `$5.947142882706` compute plus storage.
- Completed cumulative upper bounds: `34,792.248142169` provider seconds and
  `$7.247373532488`, within the strict 36,000-second / $9 authorization.
- Remaining authority: `1,207.751857831` provider seconds and
  `$1.752626467512`. Provider time is the binding constraint and is not enough
  for a scientifically useful full-fidelity resume.

At slice or science terminal, require all of the following before treating the
lifecycle as closed:

1. The terminal export contains the newest recovery snapshot and
   `guest-terminal.json`.
2. The host copy has a matching SHA-256 and passes a fresh restore verification.
3. The off-Pod acknowledgement matches the terminal archive and sync nonce.
4. The controller confirms deletion of Pod `vl7erkqk6z29ev`.
5. The controller confirms deletion of volume `q0fir47rfw` only after the
   external restore gate.
6. Authenticated RunPod inventory is empty and both terminal lifecycle receipts
   are persisted.

Do not substitute CPU work, restart a fresh root, reuse the rejected K archive,
or launch a duplicate. The next paid execution must restore the exact final
s3h archive above. It requires a fresh explicit expansion of the cumulative
cost and provider-time ceilings because the remaining `1,207.75` seconds are
cleanup headroom rather than a scientifically useful resume tranche.
