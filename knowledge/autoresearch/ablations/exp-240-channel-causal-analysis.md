---
type: Analysis Preregistration
title: EXP-240 Causal Channel Analysis
description: Fresh-data activation stratification and single-channel ablation of the strongest Tier-4 transition body.
status: completed
tags: [autoresearch, mechanistic, activation, ablation, Tier-4]
timestamp: "2026-07-12T12:15:00+05:30"
---

# EXP-240: causal channel analysis

Analyze frozen EXP-230 single-head control checkpoint `c26e34af9581b6b0f3d26e72715c512823b07250bee48a56ae63cf32cffe5baa`, the strongest official-like one-head continuation. Use fresh official-like prime identities and no consumed independent or checked-public rows.

1. Measure baseline exact transition accuracy on 20,000 cases.
2. Capture final normalized hidden activations for 20,000 cases, stratified by quotient class, correctness, and 30–32-bit versus shorter moduli.
3. Zero each of the 48 final hidden channels through an intervention hook and re-evaluate the same 5,000 generated cases, reporting overall and per-quotient exact drops.
4. Jointly ablate the four largest causal channels to test non-additivity.
5. Compare causal drops with local/global/quotient-head weight magnitudes.

A channel is critical if overall exact accuracy falls at least 0.5 point or quotient-2 accuracy falls at least two points. If at most eight channels are critical, design a targeted redundancy or protected-channel follow-up. If dependence is distributed, reject narrow channel surgery and prioritize a different representation/objective. This analysis cannot promote a model.

## Outcome

No single channel crossed either causal threshold. The paired unablated reference scored 99.68%; the largest individual overall drop was 0.34 point and the largest quotient-2 drop was 0.879 point. Jointly ablating the four strongest channels reduced accuracy by 1.70 points. Reject narrow channel surgery: the remaining failure mechanism is distributed. EXP-250 will instead mine rare model-dependent transition failures with uniform replay.
