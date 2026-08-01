---
type: Research Note
title: EXP677 M004.b known-step length-schedule map
description: Page-anchored mapping of Looped Transformers for Length Generalization to Candidate005 training-only width and visit scheduling.
status: parked-calibration-runner-failure
tags: [autoresearch, M004b, length-schedule, known-step]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP677 M004.b mechanism map

Observed local source evidence:

- Page 5 Section 4.1 requires end-to-end `(x,y)` pairs, diverse problem lengths
  and a known required iteration count `T(n)` for each training sample.
- Pages 5-6 Section 4.2 and Equation 1 apply final-output loss after exactly the
  known number of shared-block iterations; the argument relies on different
  lengths supplying different step counts, not intermediate labels.
- Page 7 Table 1 distinguishes first-order prediction, NoPE, input injection
  and predefined/confidence halting. Page 9 Section 6.1.3 uses a curriculum from
  smallest length to maximum. Pages 9-10 report exact-match evaluation and make
  confidence stopping optional inference behavior.

Bounded SAIR adaptation: legal operand width is the problem length, the existing
Candidate005 dilation schedule length is the known step count, and the endpoint
target is supervised only after that final physical visit. Candidate005's
convolutional topology has no learned position embedding and already reinjects
legal input. The formal model remains the original `SharedDilatedConvGRU` with
the exact state dictionary, architecture, parameter count, call graph and
outputs. Confidence/adaptive inference, NoPE architecture changes, intermediate
targets, visit embeddings, visit shuffling and the EXP676 moving-window loss are
excluded.

The `knowledge/autoresearch/backlog.md` prose saying M004.b remained closed is a
stale EXP676 interpretation. The binding audit marks M004.b open, and EXP676's
own mechanism note says it excluded M004.b. EXP677 resolves this by treating
the component as previously open and producing the first isolated evidence.

The isolated implementation passes its nine-test focused cone, but the sole
package oracle stopped before pytest because the separate worktree did not
contain the protected immutable challenge snapshot required by project
validation. All science roots remain unopened. This records a source-faithful
implementation map and a verification park, not scientific completion of
M004.b.

T088 later authenticated the same frozen mechanism package and reused a green
validator/716-test oracle from T087. Root `20261258` opened exactly once and all
five arms materialized, but the first evaluation failed because inherited
late-depth measurement reads a `window.size` config key absent from EXP677's
frozen schema. The recovery performed no repair or retry; no calibration gate
or science result exists, and formal/confirmation roots remain unopened.
