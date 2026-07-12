---
type: Mechanistic Analysis
title: EXP-520 Tier-5 Recovery Delta Localization
description: Static tensor-delta audit and causal patching groups for the EXP-329 control/challenger pair.
status: preregistered-follow-up
tags: [autoresearch, mechanistic, causal-patching, Tier-5, cellular]
timestamp: "2026-07-12T23:00:00+05:30"
---

# EXP-520: Tier-5 recovery delta localization

Compare the frozen EXP-329 teacher-forced control (`b741fd...8560`) and autonomous-state recovery challenger (`050daf...76f7`). They share the 99,824-parameter cellular architecture and training start; the challenger scored 94.267% width-64 rollout versus 93.133% for control on matched selection data but missed the frozen +1.5-point causal gate.

The total L2 distance between checkpoints is 0.4962. Dominant tensor deltas are gate weights/bias (0.3702 combined), candidate weights/bias (0.2521), and quotient-head tensors (about 0.2099); stem plus state-norm and output-norm plus local-head deltas are much smaller. Relative tensor deltas are roughly 2.1–2.5% for gate/candidate weights and under 0.8% for local-head weight. Corresponding tensor cosines exceed 0.9996.

Quotient-head output is ignored by residue decoding and autonomous rollout, so transplanting only `quotient_head.*` is a preregistered negative control that should be prediction-identical to the control. Causal groups are:

- input/state: `stem.*` and `state_norm.*`;
- recurrent gate: `gate_weight`, `gate_bias`;
- recurrent candidate: `candidate_weight`, `candidate_bias`;
- recurrent core: gate plus candidate tensors;
- readout: `output_norm.*`, `local_head.*`;
- quotient negative control: `quotient_head.*`;
- endpoints: unchanged control and full challenger.

EXP-520 transplants exact challenger tensors into the control for each group, saves every hybrid before evaluation, and evaluates every arm on identical fresh width-64 cases. This is a causal weight intervention: if a subset exceeds the full challenger, omitted recovery updates were countervailing; if only the full checkpoint wins, the mechanism is distributed. Exact trained tensors remain the source of predictions, and no symbolic arithmetic is added to inference.
