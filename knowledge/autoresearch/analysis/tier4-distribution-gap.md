---
type: Experiment Analysis
title: Tier-4 Distribution and Rollout Gap
description: Causal analysis of the heldout-to-official Tier-4 generalization drop.
status: completed
tags: [autoresearch, Tier-4, distribution-shift, rollout]
timestamp: "2026-07-12T10:50:00+05:30"
---

# Tier-4 distribution and rollout gap

EXP-202's equal-per-bit heldout rollout was 94.49%, but its five-seed official-generator result was 88.2%. The official generator samples uniformly over the numeric interval before selecting the next prime, placing most probability on 30–32-bit moduli. EXP-202's corresponding rollout bands were 90.47%, 89.37%, and 84.29%.

EXP-204 causally changed the prime distribution while holding the checkpoint family, architecture, hard-mining mixture, and evaluation mechanism fixed. On fresh official-like identities, the frozen checkpoint scored 84.60% rollout and the fine-tuned checkpoint scored 89.97%, a 5.37-point gain. This supports the distribution-mismatch hypothesis, but the post-fine-tune 32-bit band remained only 87.68% and the absolute gate failed.

The remaining gap is therefore not explained by prime-identity sampling alone. First-error counts remain distributed across the full 32-step sequence, consistent with compounding transition errors. The next intervention should expose the network to its own predicted states during training and test whether recovery/correction improves exact rollout at matched heldout identities.
