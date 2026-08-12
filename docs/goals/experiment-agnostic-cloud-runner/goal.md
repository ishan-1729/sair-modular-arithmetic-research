# Experiment-Agnostic Cloud Runner

## Objective

Create a reusable, manifest-driven cloud experiment lifecycle that can package, canary, launch, monitor, checkpoint, resume, clean up, and finalize arbitrary repository experiments without embedding EXP701 or any other experiment's scientific semantics in the control plane.

## Original Request

"Cool, then create it. But why are you creating for 701? Make it experiment agnostic"

## Intake Summary

- Input shape: `specific`
- Audience: future Codex agents and SAIR experiment operators
- Authority: `approved`
- Proof type: `test`
- Completion proof: a synthetic experiment completes and resumes through the generic CLI, safety failures are covered, and the installed skill validates and invokes only the generic interface
- Goal oracle: focused lifecycle tests plus the repository validation suite on the final bytes
- Likely misfire: wrapping EXP701 orchestration in a differently named script while retaining experiment-specific assumptions
- Blind spots considered: provider API drift, secrets, duplicated paid resources, mutable checkpoints, budget overruns, cleanup after controller failure, and experiment-specific evaluation schemas
- Existing plan facts: keep provider logic in deterministic tested code; keep the Codex skill thin; use immutable manifests and receipts; perform no paid provider launch for this implementation

## Goal Oracle

The oracle for this goal is:

`A provider-neutral synthetic experiment passes package -> canary -> launch -> checkpoint -> interrupted resume -> finalize -> idempotent cleanup through the public CLI, while malformed manifests, budget violations, identity mismatches, and duplicate resource ownership fail closed; the personal run-cloud-experiment skill passes its validator.`

The PM must keep comparing task receipts to this oracle. Planning, documentation, or an EXP701-only compatibility path is not completion.

## Goal Kind

`specific`

## Current Tranche

Implement one coherent local vertical slice: generic manifest and receipt schemas, a deterministic lifecycle engine, a provider protocol with an offline provider, immutable checkpoint generations, budget and cleanup controls, public CLI commands, focused adversarial tests, a durable runbook, and a thin auto-discovered Codex skill. No live paid launch is part of this tranche.

## Non-Negotiable Constraints

- The core may not import, name, or special-case EXP701, EXP702, EXP703, their roots, arms, seeds, checkpoints, or evaluation denominators.
- Scientific semantics live in the experiment command and manifest; the control plane treats outputs as declared artifacts.
- Never read credentials or contact a paid provider while building or validating this tranche.
- Never mutate `sources/` or `data/external/`.
- Preserve unrelated work and the competition's trained-parameters boundary.
- Use immutable content hashes, explicit budgets, single-owner resource identity, resumable phase receipts, and idempotent cleanup.
- Follow `knowledge/runbooks/implementation-process-optimization.md`: one vertical package, staged verification, evidence reuse, and no scaffold/audit loop.

## Stop Rule

Stop only when a final PM audit proves the full original outcome complete. Stop immediately if implementation would require credentials, provider mutation, or changes outside the active task scope.

## Canonical Board

Machine truth lives at:

`docs/goals/experiment-agnostic-cloud-runner/state.yaml`

## Run Command

```text
Codex: /goal Follow docs/goals/experiment-agnostic-cloud-runner/goal.md.
Claude Code: /goalbuddy Follow docs/goals/experiment-agnostic-cloud-runner/goal.md.
```
