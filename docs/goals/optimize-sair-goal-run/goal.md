# Optimize the SAIR Goal Run

## Objective

Recover the prior SAIR goal run into an evidence-driven, continuously executable research program: diagnose the run from its Agentic Traces, transfer only validated process improvements from the TradeArch precedent, and turn deep algorithms and mathematical-optimization ideas into benchmarkable hypotheses that respect the competition's trained-parameters boundary.

## Original Request

Optimize the goal run of `SAIR_mod_challenge` by first inspecting the Agentic Traces of session `019f522f-936f-77f1-a4b6-9ed26785b72b`, using deep DSA and mathematical-optimization results, and taking cues from the similar TradeArch drive and its published result documents.

## Intake Summary

- Input shape: `recovery`
- Audience: the SAIR research owner and future goal-run agents
- Authority: `requested`
- Proof type: `artifact`
- Completion proof: a final Judge audit maps trace findings, TradeArch-transfer evidence, implemented run-control improvements, current verification, and benchmark/experiment evidence to the original request and records `full_outcome_complete: true`.
- Goal oracle: the optimized run maintains a prioritized hypothesis portfolio and completes successive reproducible experiments whose exact-answer accuracy is compared against named baselines, while every process defect found in the trace is either mitigated or explicitly deferred with evidence.
- Likely misfire: producing an impressive algorithms survey or planning document without changing the run's execution quality, or claiming progress from public/weak metrics without private-seed generalization evidence.
- Blind spots considered: leakage and trained-parameters compliance; hand-coded arithmetic in inference; overfitting public seeds; compute budget; dirty-worktree ownership; weak exact-answer evaluation; false transfer from TradeArch; premature completion after planning.
- Existing plan facts: inspect the named trace first; inspect TradeArch result documents second; use deep DSA, contest, Project Euler, and research ideas as hypothesis sources; preserve repository operating rules and durable experiment provenance.

## Goal Oracle

The oracle for this goal is:

`A final Judge can reproduce the run-control checks, trace each selected optimization to trace or TradeArch evidence, verify project gates, and point to registered experiments comparing exact-answer accuracy against baselines on non-public or held-out seeds.`

The PM must keep comparing task receipts to this oracle. Planning, discovery, a passing tiny slice, or a clean-looking board is not enough. The goal finishes only when a final Judge/PM audit maps receipts and verification back to this oracle and records `full_outcome_complete: true`.

## Goal Kind

`recovery`

## Current Tranche

Diagnose the prior run, identify transferable execution improvements, select the largest safe useful optimization package, implement and verify it, then continue through benchmarkable algorithmic/model hypotheses until the full owner outcome is proven or all remaining work has truthful blocked receipts.

## Non-Negotiable Constraints

- Treat `sources/` and `data/external/` as immutable snapshots except through a documented checksum-bearing sync.
- Never place hand-coded modular multiplication, long division, Barrett/Montgomery reduction, or CRT recombination in a submission inference path.
- Register every durable result with config, seed, environment, checkpoint hash, metrics, and limitations under `experiments/runs/` and `experiments/registry.yaml`.
- Report exact-answer accuracy; distinguish observed results, source claims, and hypotheses; do not equate public-benchmark success with private-seed generalization.
- Keep development material out of the minimal `submission/` artifact.
- Use Ubuntu 24.04 WSL and `/home/ishan/.venvs/sair-mod-challenge` for repository work.
- Preserve user-authored dirty-worktree changes and require explicit bounded write scopes.
- Research-derived ideas are candidates, not facts; adopt them only after repository-specific verification or experiment evidence.

## Stop Rule

Stop only when a final audit proves the full original outcome is complete.

Do not stop after planning, discovery, or Judge selection when a safe Worker task can be activated. Do not stop after a single verified package when the broader outcome still has safe local follow-up work.

## Slice Sizing

Safe means bounded, explicit, verified, and reversible. It does not mean tiny. Prefer coherent vertical packages that improve the actual research loop: hypothesis selection, experiment design, evaluation, provenance, or submission-safe implementation.

## Board Health

The PM owns board health and checks it with the bundled GoalBuddy checker after every transition. `state.yaml` is machine truth.

## Canonical Board

Machine truth lives at:

`docs/goals/optimize-sair-goal-run/state.yaml`

## Run Command

```text
/goal Follow docs/goals/optimize-sair-goal-run/goal.md.
```

## PM Loop

1. Read this charter, the GoalBuddy execution contract, and `state.yaml`.
2. Work only on the active task with the exact required GoalBuddy role.
3. Record a compact receipt and atomically advance the board.
4. Re-run the oracle at phase, risk, rejected-verification, and final boundaries.
5. Continue until a final audit records `full_outcome_complete: true`.
