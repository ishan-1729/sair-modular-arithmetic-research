# T001: CUDA reachability, hotspot, and branch-base audit

Task: `T001`
Kind: `scout`
Status: `current`

## Summary

The audit rejects a blind rewrite of every historical experiment. The
authoritative committed CUDA base is `c40dc7a`, a two-commit descendant of the
root HEAD `07c9e1a`; it includes EXP698 and the EXP699 CUDA replay, and its
EXP699 config/runner hashes still match the untracked active source exactly.
It does not include the current uncommitted GoalBuddy controls, AGENTS/program
changes, or implementation runbook. Those bytes must be preserved and handled
explicitly rather than silently copied or discarded.

A read-only AST baseline over the `c40dc7a` worktree found 229 experiment
Python files, 159 `run.py`/`model.py` roots, and 165 repository-local reachable
modules. Of those, 158 reference PyTorch, 122 contain tensor factories, only 49
show an explicit device boundary, 74 combine tensor factories with no explicit
device boundary, 78 contain host-transfer/conversion boundaries, 107 load
checkpoints, and 74 construct optimizers. This is a migration program, not a
safe global flag.

## Authoritative state and branch reconciliation

- Root worktree: `agent/submission-ready-horner-tcn9` at `07c9e1a`, dirty with
  pre-existing user governance/research changes plus this untracked goal.
- Existing replay branch: `codex/exp699-cuda-hardware-replay` at `c40dc7a`.
- Ancestry: `07c9e1a...c40dc7a` is `0 2`; `07c9e1a` is the merge base.
- `c40dc7a` adds EXP698 durable artifacts and EXP699 source/replay artifacts.
- Active EXP699 config hash is
  `8290e327b38cc035e106a2640618223dd84eaebb2ffdc818f5bd0098c2b7a0c6`;
  active and committed copies match.
- Active EXP699 runner hash is
  `eca1ee3787c3ac292375546d35e7518e140d5f9401285140dc9e37f67f8df344`;
  active and committed copies match.
- The progress receipt remains unchanged at
  `64545c3ce37696770e70b5bb7962808ffe9ede77163fa5dfb2cb44ee499b5b6e`.
- `c40dc7a` does not contain
  `knowledge/runbooks/implementation-process-optimization.md`; its committed
  AGENTS file also does not contain the new runbook mandate.
- Current governing/control hashes are:
  - `AGENTS.md`: `22745dcaa88776a2a3e63d0640fbaf800a9d2c01e95d416b0b04b78202c2a99b`
  - autoresearch program: `cb4133e672349091620db541910e83c6fee1d290da181b82d1bcf2108655f27b`
  - implementation runbook: `cabd23d4af573f9631aba36f146000af7ca77314ff3b5538ef21932585c10b67`
- Recommendation: base `codex/cuda-portable-experiments` on `c40dc7a`; carry
  this goal control plane deliberately in the first Worker package. Treat the
  other dirty governance files as user-owned authoritative instructions, not
  implicit staging authorization. The Judge must state whether they remain
  external governing inputs or are explicitly included in the new branch.

## Regenerable inventory baseline

The current root contains 111 `exp_*` directories, 108 experiment `run.py`
files, 50 experiment `model.py` files, and 134 experiment configs. The
`c40dc7a` worktree contains 113 experiment directories and 110 runners because
it adds EXP698 and EXP699. The registry has 141 durable runs; EXP699 is not a
registered terminal run. Recent durable statuses are mostly parked, failed, or
non-comparative, so historical executability must not be confused with active
promotion relevance.

The future generator must use AST-qualified symbols rather than regex counts.
It should emit one deterministic row per execution/device boundary with:

1. path and qualified symbol;
2. reachability roots and local-import parents;
3. experiment family and durable registry/config status;
4. tensor/checkpoint/optimizer/host-transfer operations;
5. classification and evidence;
6. CPU versus CUDA responsibility;
7. frozen hash/provenance constraints;
8. representative smoke command;
9. exclusion reason and owner; and
10. source-tree/config/dependency identity used to generate the row.

Inventory roots must include every committed experiment `run.py` and
`model.py`, configs, registry records, documented operator commands, tests,
and `experiments/cloud_benchmarks`. Repository-local imports are followed to
closure. `experiments/runs/**/superseded/**`, generated receipts, and immutable
external snapshots remain evidence, not runnable migration roots. A committed
CLI with no config/registry row is still included and may only be excluded as
explicit archived/dead code with evidence.

Allowed classifications:

- `device-native-unvalidated`: operations derive device from inputs/model, but
  no real CUDA receipt exists;
- `adapter-required`: the route constructs CPU tensors or loads CPU state into
  an otherwise accelerator-capable model;
- `intentionally-cpu-boundary`: exact-label generation, orchestration, or
  evidence serialization stays on CPU behind a tested boundary;
- `archived-frozen-route`: retained historical evidence that must not be
  rewritten; a separate adapter/replay is required for execution;
- `cuda-hardware-validated`: allowed only after an exact-commit real-GPU
  receipt.

`intentionally-cpu-boundary` is invalid without a concrete semantic or measured
reason and a smoke test proving the boundary does not pull the hot model route
back to the host.

## Hotspots and family structure

The two highest-leverage shared authorities are:

- `experiments/exp_220_width_independent_cellular/model.py`, imported by 24
  later experiment modules; and
- `experiments/exp_664_sprm_boundary_perturbation/run.py`, imported by 24 recent
  experiment runners including EXP698/EXP699.

The EXP300 bit-tensor sampler/training route is also imported throughout the
width64, tier6, and recent candidate-derived lineages. These shared modules are
heavily frozen by configs and tests, so changing them in place would invalidate
historical identities. Use explicit adapters/wrappers and preserve originals.

Performance-plausibility groups:

1. **Cellular/ConvGRU lineage (EXP220/EXP300 and dependents):** high reuse and
   substantial batched tensor work; best first adapter family. Python bigint
   sampling/labels stay CPU-side, while features, model, loss, optimizer, and
   rollout state need an explicit device boundary.
2. **Recent SPRM/candidate-derived lineage (EXP664-EXP699):** shared runner
   authority and long formal routes, but frozen/consumed identities make
   separate replay adapters mandatory. EXP699 is the first existing prototype,
   not family-wide proof.
3. **Graph/PyG lineage (EXP400/EXP401/EXP510):** naturally accelerator-suitable
   but requires its own torch-geometric dependency and batched graph-data smoke
   route.
4. **Independent recurrent families (EXP650-EXP663):** Mamba, TRM, neural RAM,
   layer-loop, UT, LoopFormer, DRM, and related models have distinct state and
   external-library semantics. Port only as bounded representative subfamilies
   after the shared boundary is approved.
5. **Analysis, causal probes, beam/reranking, registry/evidence, exact-label,
   and submission routes:** often Python/checkpoint/host dominated. Keep their
   orchestration on CPU; only model-forward components enter accelerator scope.
   Submission inference remains CPU by competition contract.

High raw hotspot files include EXP240 causal analysis, EXP627 reranking,
EXP637 macro-transition distillation, and recent EXP663-EXP697 runners, but
many counts reflect checkpoint/evidence processing rather than GPU-suitable
compute. Rank candidates by end-to-end route share, not call count.

## Existing EXP699 contradictions that bind later tasks

- Only the EXP699 replay exposes a `--device` CLI. No existing test executes a
  CUDA training/evaluation step; the suite proves CPU parity and fail-closed
  CUDA absence only.
- Its bootstrap requires an already-installed exact uv and contains direct
  version pins, not a complete hashed transitive lock or driver/compute
  capability matrix.
- Its recurrent rollout calls `.cpu()`/host reconstruction inside every width
  position, creating synchronous transfers that can erase GPU benefit.
- Its process deadline cannot destroy the provider resource.
- Reported `actual_compute_charge_usd` uses scoped experiment time, not full
  provider-billed lifetime.
- The 11,452-second comparator is one completed teacher-cosine arm, not the
  historical full 44-hour route.
- The final source edits did not receive one subsequent clean-clone full-suite
  pass; prior broad and targeted fixture-assisted passes are useful but
  fragmented evidence.

## Recommended packages and gates

T003 should be one coherent decision-enabling package, not a mass rewrite:

- create the isolated branch/worktree from the Judge-approved base;
- make the GoalBuddy control plane durable there;
- implement the deterministic inventory generator/checker and initial manifest;
- add a small shared explicit device/runtime module;
- adapt one cellular/EXP300 representative route without modifying frozen
  source files; and
- include CPU golden parity, device-boundary lifecycle counts, a fake/no-CUDA
  fail-closed test, and the exact real-CUDA smoke command.

Suggested focused verification for that package:

```text
python -m py_compile <new runtime and inventory modules>
python <inventory checker> --check
python -m pytest -q <new CUDA runtime tests> <representative family tests>
python tools/validate_project.py
```
The first Judge boundary should approve broader expansion only if the manifest
is mechanically closed for the selected source identity, frozen hashes remain
unchanged, CPU parity passes, recurrent host-transfer lifecycle counts are zero
inside the measured model loop, and the hardware smoke fails closed without a
GPU. Actual performance and hardware validation remain unproven until the
authorized real-GPU task.

## Commands and evidence used

- `git status --short --branch` and `git worktree list --porcelain`
- `git rev-list --left-right --count` and `git merge-base` for branch ancestry
- SHA-256 comparison of active and committed EXP699 source/progress bytes
- `find` counts for experiment directories, runners, models, and configs
- Python AST scan over `experiments/**/*.py` in the `c40dc7a` worktree
- `rg` import scans for EXP220, EXP300, EXP664, explicit device flags, CUDA
  references, and config hash/provenance fields
- YAML registry summary for 141 durable run records

## Board Receipt Snippet

```yaml
receipt:
  result: done
  task_id: T001
  note: notes/T001-cuda-scope-audit.md
  summary: "Closed baseline: 165 reachable modules, 74 factory-without-device files, 78 host-boundary files; base c40dc7a with explicit dirty-governance reconciliation; adapters required instead of historical rewrites."
```
