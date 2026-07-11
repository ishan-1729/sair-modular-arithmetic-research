---
type: Open Questions Register
title: Competition Specification Questions
description: Contradictions, mutable rules, and missing operational details requiring monitoring or organizer clarification.
status: open
tags: [competition, risk, clarification]
timestamp: "2026-07-11T23:15:00+05:30"
---

# Open specification questions

| ID | Question or inconsistency | Project treatment |
|---|---|---|
| SPEC-01 | Rules say no code point sees all three inputs, while `predict_digits` receives all three encodings and shipped models use identity preprocessing. | Follow the executable interface and keep preprocessing pure. |
| SPEC-02 | Configuration asks for five primes per scored tier, while Tier 1 fixes only `{2,3,5,7}`. | Treat the generator at the pinned commit as authoritative for reproduction. |
| SPEC-03 | Prose suggests 1,233-digit operands; the public JSONL contains some 1,234-digit 4096-bit values. | Use bit geometry and actual generator tests. |
| SPEC-04 | GPU batching is recommended but the current Docker image is CPU-only. | Make CPU sandbox timing a release gate. |
| SPEC-05 | Exact CPU/RAM allocation and separate `load()` and determinism timeouts are unspecified. | Benchmark multiple constrained CPU profiles and retain headroom. |
| SPEC-06 | `highest_tier_above_90` does not state whether lower tiers must also qualify. | Test pinned scorer behavior; seek organizer confirmation before metric gaming. |
| SPEC-07 | Rules, package allowlist, sandbox, and behavioral signals remain experimental. | Pin evaluator commit and Docker digest for every durable result. |
| SPEC-08 | Public licensing expectations for the final Hugging Face artifact and training-code disclosure are unstated. | Keep provenance complete and prepare releasable training documentation. |
| SPEC-09 | Checked Tier-0 JSONL no longer matches current generator `p` values for its final five sublevels. | Treat checked files as public authority and generated sets as separately versioned. |
| SPEC-10 | Leaderboard code deduplicates by `repo_id`, while prose says per team. | Do not rely on implementation drift; follow registered-team rules. |
| SPEC-11 | Local timeout checks are cooperative and load/determinism bounds are not implemented. | Design to the stated 300-second intent with substantial headroom. |

Recheck live rules before each milestone and freeze. A change is material if it alters the interface, generator, scoring, runtime, allowlist, or compliance boundary.
