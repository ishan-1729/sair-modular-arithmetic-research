---
type: Experiment Preregistration
title: EXP-312 Expanded-Width Cellular Retry
description: Resource-corrected repeat of EXP-311 with identical training and successively halved evaluation counts.
status: completed
tags: [autoresearch, cellular, mixed-width, curriculum, retry]
timestamp: "2026-07-12T11:25:00+05:30"
---

# EXP-312: expanded-width cellular retry

Repeat EXP-311's exact 1,600-step 8/12/16/20/24-bit training stream from the same frozen source. Change only evaluation fidelity: 5,000/1,000 width-16 transition/rollout cases and 10,000/2,000 at widths 24 and 32. Save the trained checkpoint before evaluation so a completed training artifact is not lost to evaluator runtime.

Keep the same absolute gates: 25% retained width-16 rollout; 90%/10% width-24 transition/rollout; and width-32 transition gain of ten points to at least 84%, at least 1% rollout, and no more than 30% invalid rollouts. Passing authorizes higher-fidelity confirmation. No checked or consumed independent answers may be read. Stop at ten wall minutes.

## Outcome

Passed every gate. Width-16 transition/rollout is 99.92%/99%; width-24 is 99.88%/96.2%; zero-shot width-32 is 99.4%/85% with 0.15% invalid rollouts. Width-32 transition improves 29.97 points over the frozen source on the paired high-bit split. Freeze checkpoint `d1f2bbd22ca944aa49fb6bcedc57d1c4738fd44d4c30c27095d7794a57de1029` and continue on fresh official-numeric width-32 identities before any independent or checked evaluation.
