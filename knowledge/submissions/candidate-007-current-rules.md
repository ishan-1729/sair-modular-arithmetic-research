---
type: submission-candidate
title: Candidate 007 current-rules learned-reduction adapter
description: Candidate006-derived Tier-5 artifact that consumes raw operands and passes the current rules and isolated runtime envelope.
status: published-public-current-rules
date: 2026-08-13
timestamp: "2026-08-13T06:40:01Z"
tags: [candidate007, compliance, learned-reduction, runtime-validated]
---

# Candidate 007: current-rules Candidate006 adapter

Candidate007 keeps Candidate006's frozen weights but replaces its prohibited
deterministic `a mod p` / `b mod p` front end. Ordinary code feeds raw operand
bits MSB-first on a fixed schedule with no model feedback. Candidate006's
trained transition predicts every residue state and then the product. There is
no deterministic full-width operand reduction and no output repair.

The learned routes are retained through 64-bit moduli. Wider inputs receive a
constant zero, which the current rules explicitly allow as a fallback. This
boundary is necessary because a compliant 128-bit route needs two 256-step
learned reductions plus a 128-step product pass and cannot meet the CPU limit.

## Final evidence

- Current upstream rules commit: `99cac6ef5c2f82e53105ec0ddcbb9b8d37bf6fca`.
- Current static checker: clean; preprocessing isolation and determinism pass.
- Checked 1,100-case result: 48.2% overall, Tier 5 at 91%, all tiers complete,
  153.483 inference seconds.
- Isolated official 4-CPU/8-GiB sandbox: the same 48.2% / Tier-5 result, all
  tiers complete. The durable full-container wall is 226.671 seconds, which is
  conservatively below the 300-second inference budget.
- Final weights are byte-identical to Candidate006:
  `a7d25ee4eed49c5f625bc8369447be9f51ff1be54766631d8b96ab725860553e`.

The exact three-file upload artifact is
[`experiments/runs/exp-708-candidate007-current-rules/artifact`](../../experiments/runs/exp-708-candidate007-current-rules/artifact).
Its manifest and model SHA-256 values are
`95707215a920ab2cab09fd16ad1347d5b8055626776999af71e54cc2867364d8`
and `bf4bdbdda90ef148ebc9e0b4db3c5c73d7b73e2156c868ce0c05b5c1471d8833`.

## Public Hugging Face release

- Repository: [`ishan1729/modular-arithmetic-candidate007`](https://huggingface.co/ishan1729/modular-arithmetic-candidate007)
- Immutable commit: [`f4d69a7fdd56da6422dace816d7995de7236f119`](https://huggingface.co/ishan1729/modular-arithmetic-candidate007/commit/f4d69a7fdd56da6422dace816d7995de7236f119)
- Anonymous access, remote payload byte identity, and the current static checker pass.
- The official `evaluate-hf` path completed all 1,100 cases at this exact commit on
  an independent fixed seed: 48.5% overall, Tier 5 highest, deterministic.

The competition form itself has not been submitted. The required values are the
repository ID and immutable commit above, and the `ishan1729` owner must be one
of the Hugging Face identities declared for the registered SAIR team.

## Limitation

Candidate007 is the strongest artifact in this repository with demonstrated
current-rule and runtime eligibility, but it is deliberately weaker than
Candidate006: checked accuracy falls from 66.6% to 48.2%, and the ranking tier
falls from 6 to 5. Candidate006's higher score depended on the now-prohibited
deterministic operand pre-reduction. No private-seed score or generalization
claim follows from this public evidence.
