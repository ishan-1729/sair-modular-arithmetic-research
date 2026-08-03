# CUDA-Portable Experiments

## Objective

Create a separate branch that makes the mechanically defined, actively
reachable experiment surface CUDA-prepared while preserving CPU behavior,
frozen scientific identities, reproducibility, and the trained-parameters
boundary. The branch must remain labeled `cuda-prepared-hardware-unvalidated`
until a real GPU acceptance run proves it is plug-and-play.

## Original Request

"Please create another branch where you port the entire repo to CUDA (the
relevant experiments part) so that I can plug and play when I get access to
Cloud GPU."

## Intake Summary

- Input shape: `existing_plan`
- Audience: repository owner renting a cloud GPU
- Authority: `requested`
- Proof type: `test`
- Completion proof: two explicit milestones: (1) a committed isolated branch
  with a mechanically closed entrypoint/import inventory, reproducible locked
  environment, shared device/runtime support, bounded CPU parity, clean-clone
  regression evidence, and a fail-closed hardware acceptance command; then
  (2) one real CUDA acceptance run covering every ported family before the
  stronger `cuda-hardware-validated` or plug-and-play claim is allowed
- Goal oracle: the generated CUDA readiness inventory must have no unclassified
  reachable experiment entry points or tensor/device hotspots, and the hardware
  acceptance receipt must match that closed inventory
- Likely misfire: adding a global CUDA flag or porting only EXP699 while hidden
  CPU tensor construction, checkpoint loading, and evaluator paths remain
  undiscovered
- Blind spots considered: not every route benefits from GPU; exact-label and
  orchestration work may remain intentionally CPU-side; historical hashes must
  remain authentic; small recurrent models can become slower through launch
  overhead and CPU/GPU synchronization; direct dependency pins are not a
  transitive lock; a process timeout cannot stop provider billing; fragmented
  fixture-assisted tests are not a clean-clone regression pass; physical CUDA
  execution is unavailable until the user rents a GPU
- Existing plan facts: use a new isolated branch/worktree; port relevant
  experiments only; reuse EXP699 CUDA replay lessons; provide requirements.txt
  manifests with uv; do not run long local experiments

## Goal Oracle

The oracle for this goal is:

`On codex/cuda-portable-experiments, a generated machine-readable inventory accounts for every mechanically reachable experiment entry point, imported shared module, tensor factory, checkpoint load, model construction, and recurrent device-transfer hotspot as ported, already device-agnostic, or intentionally CPU-only with evidence; a clean-clone locked uv bootstrap and one final regression pass succeed; an external billing-safe launcher is present; and a real GPU acceptance receipt covers every ported family before hardware validation is claimed.`

The PM must keep comparing task receipts to this oracle. Discovery, one ported
experiment, or a passing CPU-only slice is not enough. The goal finishes only
when a final Judge audit maps receipts and verification back to the oracle and
records `full_outcome_complete: true`.

## Goal Kind

`existing_plan`

## Current Tranche

Continuously define, map, port, package, and verify the reachable experiment
surface. First create a mechanically reproducible inventory and reconcile the
branch base with uncommitted governance/source state. Next implement an
explicit shared device boundary and port representative high-value families,
removing hot-loop CPU/GPU transfers before authorizing broad migration. Expand
only after a Judge confirms correctness and credible acceleration structure.
Finish the local milestone with a locked fresh-VM environment, provider-side
termination controller, clean-clone regression pass, and committed branch
labeled `cuda-prepared-hardware-unvalidated`. Do not claim plug-and-play until
the separately authorized real-GPU acceptance task passes.

## Non-Negotiable Constraints

- Create and work only in a new isolated worktree on
  `codex/cuda-portable-experiments`; preserve the dirty root and all active
  experiment worktrees.
- Do not create the worktree until a read-only audit chooses a base that includes
  the committed EXP699 replay work without silently losing the root worktree's
  uncommitted governance and research-source changes.
- Treat `sources/` and `data/external/` as immutable.
- Do not alter consumed scientific results, roots, recorded metrics, or frozen
  source hashes merely to make a hardware replay pass.
- Preserve CPU execution and exact-answer semantics; use explicit device
  boundaries instead of a global default-device side effect.
- Do not rewrite historical frozen runners wholesale. Port active shared cores;
  use separate hardware adapters/replay wrappers for frozen or consumed routes.
- Keep exact Python/big-integer label generation CPU-side unless tensor-native
  behavior is already proven equivalent.
- Do not add forbidden hand-coded arithmetic to submission inference paths.
- Bootstrap the exact uv version rather than assuming it is installed. Use a
  complete transitive lock with hashes, explicit index-to-package mapping,
  versioned CUDA/PyTorch profiles, NVIDIA driver and compute-capability
  preflight, deterministic controls, executable launchers, and environment
  receipts. No silent dependency or CPU fallback is allowed.
- Eliminate per-step `.cpu()`, `.item()`, host reconstruction, or device-copy
  synchronization from measured hot loops unless the inventory records a
  correctness boundary and a benchmark justifies it.
- Do not run long training/evaluation locally. Tiny parity/smoke checks are
  allowed when they are bounded and recorded.
- After the final source edit, run `python tools/validate_project.py` and one
  single clean-clone `python -m pytest` pass using a documented fixture
  restoration mechanism. Targeted reruns may diagnose failures but cannot
  substitute for the final clean pass.
- A timeout inside the rented host is not a billing stop. Provider credentials
  stay outside the experiment VM; an external controller must enforce prepaid
  or hard budget settings, disable auto-recharge when available, destroy the
  resource from a `finally` path, and record provider-billed lifetime.
- Do not push, rent, purchase, or launch a cloud resource without explicit
  user authorization.

## Stop Rule

Stop only when a final audit proves the full original outcome is complete.

Do not stop after planning, inventory, shared-runtime scaffolding, or one
representative experiment if safe local Worker work remains. If physical GPU
access is unavailable, finish every local portability, packaging, static,
CPU-parity, billing-controller, and fail-closed hardware-acceptance task and
commit the branch as `cuda-prepared-hardware-unvalidated`. At that point the
board may pause for explicit cloud authorization, but it must not record
`full_outcome_complete: true`, `cuda-hardware-validated`, or plug-and-play.

## Slice Sizing

Safe means bounded, explicit, verified, and reversible. It does not mean tiny.
Group repeated same-shape device migrations by shared experiment family and
review the whole family package. Use a small task only when an isolated
failure, frozen identity, or ambiguous dependency boundary blocks a larger
slice.

Do not use one catch-all Worker task for every remaining experiment. Gate each
expansion tranche on a representative-family Judge review that checks CPU
parity, frozen identities, hot-loop transfer structure, memory feasibility,
and whether GPU execution is technically plausible for that family.

## Readiness States

- `cuda-inventory-open`: at least one reachable entry point or hotspot is
  unclassified.
- `cuda-prepared-hardware-unvalidated`: the branch, closed inventory, locked
  bootstrap, CPU parity, clean regression pass, billing controller, and
  fail-closed hardware acceptance command are complete, but no real GPU receipt
  exists.
- `cuda-hardware-validated`: the exact committed branch and locked environment
  passed the real GPU acceptance matrix with semantic, resource, and provider
  billing receipts.
- `plug-and-play`: allowed only as a user-facing synonym for
  `cuda-hardware-validated`, never for preparation alone.

## Mechanical Scope Closure

The inventory generator must begin from all of the following roots and follow
their repository-local imports:

- every `experiments/**/run.py` and `experiments/**/model.py` reachable from a
  committed config, registry entry, documented operator command, or test;
- every committed experiment CLI/module entry point even when not registered,
  with an explicit archived/dead-code classification if it is excluded;
- shared model, sampler, evaluator, checkpoint, and resource-measurement modules
  imported by those roots;
- every reachable PyTorch tensor factory, model/checkpoint construction,
  optimizer-state load, `.to(...)`, `.cpu()`, `.item()`, NumPy conversion, and
  recurrent transfer boundary.

Each manifest row must record path, symbol or entry point, reachability roots,
experiment family, classification, evidence, CPU/CUDA responsibility, frozen
identity constraints, smoke command, and exclusion reason where applicable.
The inventory is closed only when a regeneration check produces no missing or
stale rows. `intentionally-cpu-only` requires a concrete semantic or measured
reason and a tested device boundary; it cannot be used as an escape hatch.

## Adversarial Acceptance Gates

- No actual CUDA execution means no hardware-validation claim.
- CPU parity for one EXP699 update is evidence for that route only, not the
  other experiment families.
- The scoped 11,452-second EXP699 arm is not the full historical 44-hour
  workload and cannot alone establish repo-wide speedup.
- Porting is rejected when device transfers remain inside recurrent hot loops
  without an explicit benchmark-backed exception.
- Direct pins without a transitive lock and hashes fail reproducibility.
- A bootstrap that assumes uv, a driver, or a compatible GPU already exists
  fails fresh-VM readiness.
- A process timeout without external provider destruction fails cost safety.
- Split test passes after ad hoc ignored-fixture copying do not equal a final
  clean-clone regression pass.
- The branch and goal control files must be committed together or have an
  explicit cross-worktree control-plane receipt; untracked goal files do not
  count as durable work.

## Board Health

The PM owns board health. Validate it with:

```bash
node /mnt/c/Users/Ishan/.codex/plugins/cache/goalbuddy/goalbuddy/0.4.2/skills/goal-prep/scripts/check-goal-state.mjs docs/goals/cuda-portable-experiments
```

## Canonical Board

Machine truth lives at:

`docs/goals/cuda-portable-experiments/state.yaml`

If this charter and `state.yaml` disagree, `state.yaml` wins for task status,
active task, receipts, verification freshness, and completion truth.

## Run Command

```text
/goal Follow docs/goals/cuda-portable-experiments/goal.md.
```

## PM Loop

On every `/goal` continuation:

1. Read this charter and the GoalBuddy execution contract.
2. Read `state.yaml`; work only on its active task.
3. Keep one write owner and preserve all unrelated dirty state.
4. Dispatch the role named by the task, collect a compact receipt, and update
   board truth.
5. Recheck the generated closed inventory, hot-loop transfer audit, readiness
   state, and smoke-matrix oracle after each Worker package.
6. Advance to the next largest safe package while local work remains.
7. Review at architecture, frozen-identity, rejected-verification, and final
   boundaries.
8. If only real-GPU validation remains, record the branch-preparation milestone,
   preserve the exact required authorization, and pause without overstating
   completion.
9. Run the GoalBuddy can-stop checker before ending execution.
