---
type: Incident Receipt
title: EXP-649 calibration-seed contamination and retirement
description: Durable receipt for retiring calibration root 20261148 after a scaffold parity test computed official-reference labels in memory, and for selecting audited replacement root 20261150.
status: resolved-seed-retired
tags: [autoresearch, exp-649, data-hygiene, leakage-control, incident]
timestamp: "2026-07-20T03:59:35+05:30"
---

# EXP-649 calibration-seed contamination and retirement

## Incident

During scaffold verification, `tests/test_exp649_stochastic_trajectory_breadth.py` called the pinned official `generate_private_test_set` with then-calibration root `20261148`. That reference generator necessarily computed `expected` for all 1,100 generated identities in process memory even though the test assertion compared only input dictionaries. This violated EXP-649's stricter rule that calibration-root labels must never be generated, loaded, inspected, or scored.

The exact command was:

```text
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/d/Projects/SAIR_mod_challenge && source /home/ishan/.venvs/sair-mod-challenge/bin/activate && pytest -q tests/test_exp649_stochastic_trajectory_breadth.py'
```

The process completed in 50.9 wall seconds with exit status 1: ten tests passed and a later, unrelated bookkeeping-path test failed. The parity test had already completed. The exact process start timestamp was not captured; the incident was reported and stopped from further propagation before the durable audit recorded at `2026-07-20T03:56:35+05:30`.

The generator was the immutable upstream snapshot at commit `82510bba00a1126649bd76dd1a451f14d0b3eb60`; `generator.py` had SHA-256 `77f4142fe097ef66f2bd4452547b465ef3f75646191ed8258c508f5e3caa85c0`. Repository HEAD was `ad696bc6ba9275e60ed33f8a03265fca77982b58`, with the uncommitted EXP-649 scaffold present.

## What did and did not occur

- The official reference constructed 1,100 `expected` strings in volatile process memory under root `20261148`.
- No assertion accessed or compared an `expected` value; the test used `case.to_input_dict()` only.
- No reference input, label, target, score, calibration, or formal artifact was written.
- Neither the calibration CLI nor the formal CLI ran.
- No checked, independent, or official-private benchmark answer was accessed.
- The failure does not produce model evidence, a metric, a runtime verdict, or authorization to continue with that root.

Because generation alone violated the frozen input-only boundary, root `20261148` is permanently retired and denylisted. Its evidence cannot be repaired by deleting process memory or by noting that labels were not inspected.

## Replacement-root prior-use audit

Root `20261150` was proposed only after the incident. The initial preselection search completed with no workspace or all-reference Git-history match before the literal was added to the scaffold. A reproducible audit receipt was then captured at `2026-07-20T03:58:18+05:30`, at HEAD `ad696bc6ba9275e60ed33f8a03265fca77982b58`, while the EXP-649 scaffold worktree was dirty:

```text
git grep -n 20261150 HEAD -- knowledge experiments configs tests submission sources
# no match; native exit 1; audit wrapper output: head_match=false

git log --all --format=%H -S20261150 -- knowledge experiments configs tests submission sources
# empty output; exit 0

rg -n "20261150" . --glob "!.git/**" --glob "!.scratch/**" \
  --glob "!configs/exp_649_stochastic_trajectory_breadth.yaml" \
  --glob "!experiments/exp_649_stochastic_trajectory_breadth/**" \
  --glob "!tests/test_exp649_stochastic_trajectory_breadth.py" \
  --glob "!knowledge/autoresearch/ablations/exp-649-stochastic-trajectory-breadth.md" \
  --glob "!knowledge/autoresearch/analysis/exp-649-calibration-seed-retirement.md"
# no unrelated workspace match; native exit 1; audit wrapper output: unrelated_workspace_match=false
```

The history checks establish that `20261150` had no committed prior role. The unrelated-workspace check excludes only the new EXP-649 files that intentionally freeze or guard the replacement root.

## Frozen seed roles

| Seed | Frozen role | Label policy | Status |
|---:|---|---|---|
| `20261148` | retired former calibration root | no further generation of any kind | permanently denylisted |
| `20261150` | EXP-649 calibration | input generation only; no labels, targets, oracle, or scoring | Judge-approved but unopened pending clean scaffold commit |
| `20261149` | EXP-649 formal | no generation or execution before collision audit and later Judge authorization | unopened |
| `649900001` | scaffold parity test only | official offline reference labels allowed only in the test process | distinct from every experiment root |

A preliminary process using distinct test-only seed `649999001` was terminated after about ten seconds during test refactoring and wrote no artifact. It has no experiment role; `649900001` is the sole frozen parity-test seed.

## Refreshed scaffold freeze

After the root replacement and denylist were applied, the calibration scaffold froze the following hashes:

| Object | SHA-256 |
|---|---|
| complete scientific config, excluding only `resource_protocol.resolved` | `ea70a40c01e1c407be05ce7d5e7740862e670a0dd36715635614c521cae3415a` |
| complete workload/resource protocol, excluding only `resource_protocol.resolved` | `8d13c0f5d6a859a36d9360ad4209b492bc10dae42a6e6be80c0d9bdc5b3fb7cd` |
| `configs/exp_649_stochastic_trajectory_breadth.yaml` | `99b22e87eb6a55c6d362ae350dc35c0de68d2f3c8ee99befcdff829d6056f134` |
| `experiments/exp_649_stochastic_trajectory_breadth/run.py` | `04321d39de730cc653535faa8928ba7a0e975868d274503a5e53714f41daecda` |
| `experiments/exp_649_stochastic_trajectory_breadth/seed_retirement_manifest.json` | `b00eaca8a12a397851103d570ca5b00cb94f925d4f05d96a827f77eb752b712f` |
| `tests/test_exp649_stochastic_trajectory_breadth.py` | `69138188cb017d8ba37690a68e872dea3f7056bc49e1f44c5c9a53c390ad735a` |

The runner hard-fails if either canonical config hash changes outside the explicitly mutable resolved subtree. Calibration artifacts additionally bind the committed config, runner, preregistration, source note, dataset, environment, official-source receipt, checkpoint, candidate artifact, and partial-cache material hashes.

## Corrective gate

Tests must exercise the input generator through the distinct parity-test seed and assert inequality with retired, calibration, and formal roots. The production calibration wrapper must reject every denylisted root, while the formal scaffold remains fail-closed. Calibration cannot start until the updated config, runner, tests, protocol, source hashes, and freeze hashes pass focused/full validation from a clean committed tree. This receipt does not authorize formal generation, formal scoring, a selector, candidate promotion, or Goal completion.
