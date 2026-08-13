---
type: Experiment Preregistration
title: EXP-708 Candidate007 Current-Rules Adapter
description: Replace Candidate006's prohibited operand pre-reduction with learned raw-bit reduction and retain the strongest sandbox-qualified current-rule route.
status: completed-current-rule-submission-candidate
tags: [candidate007, compliance, learned-reduction, submission]
timestamp: "2026-08-13T00:00:00Z"
---

# EXP-708 Candidate007 current-rules adapter

## Question

Can Candidate006's frozen transition experts perform the newly required
full-width operand reduction themselves when raw bits are supplied on the
rules-explicit fixed, feedback-free schedule?

## Frozen contrast

Candidate006 remains immutable evidence. Candidate007 changes only its input
adapter: ordinary code tokenizes each raw operand into MSB-first bits, while
the frozen trained transition predicts every next residue state. The existing
expert then multiplies the two learned residues. No inference code computes an
operand modulo the case modulus, repairs an output, or computes the answer.

Tier 0 and moduli wider than 64 bits receive a constant zero. The current rules
explicitly permit a constant fallback. The final boundary is width-only and
independent of batch order or input fingerprints.

## Frozen reducer continuation

The first complete current-rule host run was a diagnostic selection rung:
56.7% overall in 293.910 inference seconds, but Tier 6 reached only 87/100 and
the ranking key therefore fell to Tier 5. Tiers 1/2/3/5 reached 100/99/96/91.
That six-second host margin was not treated as sandbox qualification.

One preregistered continuation is allowed before the final run. Clone only the
Tier-6 transition checkpoint into a dedicated reducer and train it for 25
batch-16 updates on generated, disjoint 121-128-bit prime identities. Each
update backpropagates through eight hard-forward, sigmoid-backward learned
residue steps with multiplicand one and raw 256-bit prefixes. Exact bigint
arithmetic is offline supervision only. The frozen Tier-6 product expert and
40% confidence-routed commutative ensemble do not change.

The first full-objective calibration update took 15.784 wall seconds and was
already 100% step-exact with loss 0.002095. A 200-update continuation therefore
projected to 3,156.8 seconds, incompatible with the stated submission deadline.
The optional continuation is capped at 25 updates before formal execution;
this is a resource calibration made without reading a continuation result.

## Gates

Run the complete current official 1,100-case public benchmark and the isolated
4-CPU/8-GiB official sandbox. Require the current upstream static check,
preprocessing isolation, determinism, all completed tiers, at most 300 seconds,
at least 48% overall, and `highest_tier_above_90 >= 5`.

The 25-update optional Tier-6 reducer continuation was terminated after
crossing the 300-second inference boundary and is excluded. Removing the
optional 40% commutative re-evaluation was also insufficient. A 127-bit
partial route completed Tier 6 at 46/100 but took 485.2 inference seconds and
was rejected. The final artifact therefore retains learned routes only through
64-bit moduli and uses the allowed constant fallback above that boundary.

The final artifact passes the current checker and manual raw-operand audit. It
scores 48.2% overall, 91/100 on Tier 5, and completes every tier. The checked
current-rule host run takes 153.483 inference seconds. The isolated official
sandbox reports the same tier vector; its console reports 143.7 inference
seconds and the durable wrapper conservatively records 226.671 seconds for the
entire container lifecycle, below the 300-second inference ceiling. This is a
submission-eligible Tier-5 candidate, not a claim that it preserves
Candidate006's invalid Tier-6/Tier-7 score.
