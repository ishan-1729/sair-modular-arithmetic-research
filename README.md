# SAIR Modular Arithmetic Research

Private research workspace for the [SAIR Modular Arithmetic Challenge](https://competition.sair.foundation/competitions/modular-arithmetic-challenge/overview). The competition asks a trained model to emit exact digits for `(a * b) mod p` while forbidding a hand-coded arithmetic solver in the inference path.

The repository contains a submission-ready 170,577-parameter learned Horner ensemble. On a full independent official 1,100-case set it scores **31.0% overall**, with **Tier 1 at 100%, Tier 2 at 97%, and Tier 3 at 99%**, advancing `highest_tier_above_90` to 3. The checked post-selection regression scores 31.3%, and randomized weights collapse Tier 3 to 2%. The exact Python 3.12/PyTorch 2.5.1 runtime reproduces the independent result with 2.1-second inference.

## Start here

- [Knowledge bundle](knowledge/index.md) — the OKF v0.1-compatible project map.
- [Competition contract](knowledge/competition/task-contract.md) — task, interface, scoring, and hard constraints.
- [Approach portfolio](knowledge/research/approach-portfolio.md) — ranked technical options and evidence.
- [Execution roadmap](knowledge/planning/roadmap.md) — work plan through the submission freeze.
- [Submission candidate](knowledge/submissions/candidate-002-tier3-hybrid.md) — current model card, compliance rationale, release state, and hashes.
- [Full-set evaluation](knowledge/evals/candidate-002-results.md) — independent, checked, pinned-runtime, and perturbation results.
- [Current POC](knowledge/pocs/learned-horner-tcn.md) — model card, evidence, and limitations.
- [Candidate evaluation runbook](knowledge/runbooks/evaluate-candidate.md) — exact official-contract and sandbox commands.

## Quick validation

```powershell
python tools/validate_project.py
python -m pytest
python -m modchallenge.cli check submission
```

The POC training and evaluation commands are recorded in the reproduction runbook and in its immutable experiment record.

## Repository layout

```text
knowledge/       OKF knowledge bundle: decisions, plans, cards, and runbooks
data/            immutable external data, generated/interim/processed zones, Croissant metadata
evals/           evaluation configs, result schemas, and durable result artifacts
experiments/     machine-readable registry and one directory per run
pocs/            executable proofs of concept and reference checkpoints
submission/      clean three-file Hugging Face release artifact
sources/         captured SAIR pages, research memo, and pinned official-repo submodule
tests/           fast contract and regression tests
tools/           project, metadata, and provenance validators
```

This is intentionally a hybrid rather than an attempt to encode every file as OKF. OKF governs the `knowledge/` bundle. Croissant describes the ML benchmark, RO-Crate describes the aggregate research object and provenance, and native formats remain authoritative for code, configuration, JSONL data, metrics, and weights. See the [format decision](knowledge/decisions/0001-format-stack.md).

## Two publication targets

This private GitHub repository is the development and research workspace. It is not the ranked submission target. The clean [`submission/`](submission/) artifact has passed local and sandbox gates and is ready to copy, but ranked entry still requires a **public Hugging Face repository** identified by repository ID and immutable 40-character commit hash. That external publication and final submission have not been performed.

## Source snapshot

The official challenge repository is pinned as a submodule at commit `82510bba00a1126649bd76dd1a451f14d0b3eb60` (2026-07-05). The public benchmark copied into `data/external/public_benchmark/` is derived from that exact revision and retains the upstream Apache-2.0 license. SAIR website captures and the supplied research memo are retained under `sources/`.

No license is granted for original work in this private repository. Third-party material remains under its original terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
