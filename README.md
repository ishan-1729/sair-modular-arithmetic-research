# SAIR Modular Arithmetic Research

Private research workspace for the [SAIR Modular Arithmetic Challenge](https://competition.sair.foundation/competitions/modular-arithmetic-challenge/overview). The competition asks a trained model to emit exact digits for `(a * b) mod p` while forbidding a hand-coded arithmetic solver in the inference path.

The repository now has a reproducible research scaffold and a deliberately small proof of concept for the most promising direction found in the supplied research: a trajectory-trained, learned Horner transition. On five held-out primes, the 21,441-parameter CPU POC reached 98.205% exact one-step accuracy and 92.10% / 85.275% / 76.95% autonomous final accuracy at 8 / 16 / 32 scan bits. It proves the workflow and exposes rollout degradation; it is not yet a competition submission.

## Start here

- [Knowledge bundle](knowledge/index.md) — the OKF v0.1-compatible project map.
- [Competition contract](knowledge/competition/task-contract.md) — task, interface, scoring, and hard constraints.
- [Approach portfolio](knowledge/research/approach-portfolio.md) — ranked technical options and evidence.
- [Execution roadmap](knowledge/planning/roadmap.md) — work plan through the submission freeze.
- [Current POC](knowledge/pocs/learned-horner-tcn.md) — model card, evidence, and limitations.
- [Reproduction runbook](knowledge/runbooks/reproduce-poc.md) — exact validation and POC commands.

## Quick validation

```powershell
python tools/validate_project.py
python -m pytest
```

The POC training and evaluation commands are recorded in the reproduction runbook and in its immutable experiment record.

## Repository layout

```text
knowledge/       OKF knowledge bundle: decisions, plans, cards, and runbooks
data/            immutable external data, generated/interim/processed zones, Croissant metadata
evals/           evaluation configs, result schemas, and durable result artifacts
experiments/     machine-readable registry and one directory per run
pocs/            executable proofs of concept and reference checkpoints
submission/      clean-room staging area for a future Hugging Face artifact
sources/         captured SAIR pages, research memo, and pinned official-repo submodule
tests/           fast contract and regression tests
tools/           project, metadata, and provenance validators
```

This is intentionally a hybrid rather than an attempt to encode every file as OKF. OKF governs the `knowledge/` bundle. Croissant describes the ML benchmark, RO-Crate describes the aggregate research object and provenance, and native formats remain authoritative for code, configuration, JSONL data, metrics, and weights. See the [format decision](knowledge/decisions/0001-format-stack.md).

## Two publication targets

This private GitHub repository is the development and research workspace. It is not the ranked submission target. Under the current competition rules, the eventual ranked artifact must be a **public Hugging Face repository** identified by repository ID and immutable 40-character commit hash. Release material must be copied deliberately into `submission/` and pass static, deterministic, sandbox, runtime, and provenance gates before publication.

## Source snapshot

The official challenge repository is pinned as a submodule at commit `82510bba00a1126649bd76dd1a451f14d0b3eb60` (2026-07-05). The public benchmark copied into `data/external/public_benchmark/` is derived from that exact revision and retains the upstream Apache-2.0 license. SAIR website captures and the supplied research memo are retained under `sources/`.

No license is granted for original work in this private repository. Third-party material remains under its original terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
