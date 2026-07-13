---
type: Feasibility Analysis
title: EXP-632 Shared-Core Diverse-Head Feasibility
description: Exact-initialization, activation, disagreement, correction, and runtime calibration for complementary residual output heads.
status: completed
tags: [autoresearch, tier6, ensemble, output-heads, feasibility]
timestamp: "2026-07-13T15:05:00+05:30"
---

# EXP-632 shared-core diverse-head feasibility

Three zero-initialized residual bit heads sit over the frozen EXP-609 recurrent core and original local head. Each residual sees a deterministic complementary two-thirds mask of the 92 hidden channels; every channel participates in exactly two heads. At initialization all head logits and majority predictions exactly equal the source, and only 279 parameters are trainable.

Uniform-transition calibration is not discriminating: after 20 updates residual RMS reaches 0.269–0.345 logits, but 500 already source-exact training transitions show zero discrete disagreement. An initial 1.90x runtime measurement is invalid because the calibration called the recurrent core twice and is excluded.

The corrected training-only calibration uses the 153 preserved EXP-629 first-error records. After 30 bootstrap head-only updates, individual heads correct 13, 14, and 9 records; majority vote corrects 11, changes 46 source predictions, and the heads disagree on 43 cases. Thus masks and bootstrap sampling produce material learned diversity on relevant errors without changing the core.

For 128 cases, one base forward takes 1.861 wall / 7.018 aggregate-CPU seconds; one shared-core three-head forward plus vote takes 1.752 wall / 6.610 aggregate-CPU seconds. The 0.941 ratio reflects measurement noise, but rules out material overhead. A formal experiment must use new prime-disjoint selection identities and measure complete rollouts. These calibration records are training evidence only and do not authorize promotion.

