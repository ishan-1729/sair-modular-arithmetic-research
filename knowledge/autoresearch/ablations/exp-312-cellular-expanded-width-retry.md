---
type: Experiment Preregistration
title: EXP-312 Expanded-Width Cellular Retry
description: Resource-corrected repeat of EXP-311 with identical training and successively halved evaluation counts.
status: preregistered
tags: [autoresearch, cellular, mixed-width, curriculum, retry]
timestamp: "2026-07-12T11:25:00+05:30"
---

# EXP-312: expanded-width cellular retry

Repeat EXP-311's exact 1,600-step 8/12/16/20/24-bit training stream from the same frozen source. Change only evaluation fidelity: 5,000/1,000 width-16 transition/rollout cases and 10,000/2,000 at widths 24 and 32. Save the trained checkpoint before evaluation so a completed training artifact is not lost to evaluator runtime.

Keep the same absolute gates: 25% retained width-16 rollout; 90%/10% width-24 transition/rollout; and width-32 transition gain of ten points to at least 84%, at least 1% rollout, and no more than 30% invalid rollouts. Passing authorizes higher-fidelity confirmation. No checked or consumed independent answers may be read. Stop at ten wall minutes.
