---
type: Research Note
title: EXP678 Target-Only Supervision Mechanism Map
description: Page-authenticated Loopies target-position masking map, SAIR analogue limits, and EXP678 terminal verification park.
status: parked-pre-root-provenance
tags: [autoresearch, loopies, M044, exp678]
timestamp: "2026-07-30T01:00:00+05:30"
---

# EXP678 target-only supervision mechanism map

The immutable source is arXiv:2607.16051v2, SHA-256 `1abfc7644b0253119b53b86327011471a0d017c2904d648461fc9d6a10b074ce`. Visual and extracted-text inspection of §4.1 pp. 16-18 shows that SPT and SFT apply token-level cross-entropy only to supervised target-response tokens; prompt, context, padding, and other non-target positions are masked. Table 8 p. 38 reports the source-scale optimizer settings. These are source claims, not SAIR results.

Candidate005 is a recurrent transition model rather than an autoregressive language model. Its closest bounded analogue is to treat the final transition in each existing eight-step recovery continuation as the supervised target and earlier recovery states as context/trajectory. This translation changes only training coefficients. It neither changes the recurrent graph nor adds inference state, calls, target canvases, validity filters, answer access, repair, or programmed arithmetic.

The primary comparison separates three questions: target-position selection (`target_only` versus `shuffled_mask`), retaining trajectory labels (`target_only` versus `endpoint_weight`), and incumbent mixed supervision (`target_only` versus `mixed_trajectory`). Frozen-source and randomized-weight arms bound continuation and learned-weight dependence. Because Loopies does not provide a matched ablation and uses incomparable language-model scale, EXP678 can support only SAIR-local causal evidence for this mask analogue.

The implementation and nine focused tests are complete, but the sole package oracle stopped at validator schema checks before pytest because the two new concept files lacked required `description` fields. The bounded repair records the terminal park. No calibration, training, W128/W256 science, or confirmation root opened, so M044.a remains scientifically open.

T076 recovered the exact four canonical record fields, green validator, and first full pytest without changing the mechanism. Pre-root authentication then exposed that the accepted freeze binds the pre-description versions of this note and the preregistration. Because T076 prohibits re-freezing, no root opens; this remains provenance evidence only.
