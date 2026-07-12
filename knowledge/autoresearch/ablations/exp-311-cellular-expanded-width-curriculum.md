---
type: Experiment Preregistration
title: EXP-311 Expanded-Width Cellular Curriculum
description: Continue the rescued width-independent cellular control on widths 8 through 24 and measure width-32 extrapolation.
status: preregistered
tags: [autoresearch, cellular, mixed-width, curriculum, length-transfer]
timestamp: "2026-07-12T11:05:00+05:30"
---

# EXP-311: expanded-width cellular curriculum

EXP-310's cellular control improved dramatically under mixed 8/12/16-bit training: 91.19% width-16 transition, 28.2% rollout, and 75.115% zero-shot width-32 transition. Continue that frozen checkpoint on a cyclic 8/12/16/20/24-bit hard-mined curriculum with no width-specific parameters.

Evaluate retention at width 16, learned performance at width 24, and zero-shot width 32 on fresh identities. Require at least 25% retained width-16 rollout; 90%/10% width-24 transition/rollout; and width-32 transition gain of ten points to at least 84%, at least 1% rollout, and no more than 30% invalid rollouts. Passing authorizes direct width-32 curriculum training and then a bigint width-64 transfer test. Failing closes this cellular curriculum. No checked or consumed independent answers may be read. Stop at ten wall minutes.
