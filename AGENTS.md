# Repository operating rules

## Scope

This repository is a private research workspace for the SAIR Modular Arithmetic Challenge. Preserve reproducibility, provenance, and the competition's trained-parameters boundary.

## Non-negotiable rules

- Treat `sources/` and `data/external/` as immutable snapshots. Update them only through a documented sync operation that records the upstream commit and new checksums.
- Do not put a hand-coded modular multiplication, long division, Barrett/Montgomery reduction, or CRT recombination routine in a submission inference path. Exact arithmetic is allowed in offline label generation and tests.
- A result is not durable until its config, seed, environment, checkpoint hash, metrics, and limitations are recorded under `experiments/runs/` and registered in `experiments/registry.yaml`.
- Project-authored knowledge belongs in `knowledge/`. Every concept document there must have YAML frontmatter with a non-empty `type`; `index.md` and `log.md` follow OKF reserved-file conventions.
- Keep the development repository separate from the minimal `submission/` artifact. Never copy credentials, training-only code, benchmark answers, or unnecessary dependencies into a submission package.
- Run `python tools/validate_project.py` and `python -m pytest` before committing.

## Evidence language

Distinguish observed results from source claims and hypotheses. A perfect public-benchmark claim is not evidence of private-seed generalization. Always report exact-answer accuracy, not only bit accuracy or arithmetic distance.
