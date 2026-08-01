---
type: Experiment Preregistration
title: EXP-613 Tier-6 Model-Conditioned Error Corrector
description: Matched test of uniform versus first-error-focused training for a frozen-base learned correction path.
status: completed
tags: [autoresearch, tier6, error-correction, hard-mining, causal-ablation]
timestamp: "2026-07-13T03:00:00+05:30"
---

# EXP-613 Tier-6 model-conditioned error corrector

## Question and hypothesis

Tier-6 failure is sparse: the frozen base is about 99.8% transition-exact, but isolated mistakes accumulate into roughly 13% failed rollouts, and 76.9% of first errors contain only one or two wrong bits. Generic corruption and generic residual capacity failed. Can a trained correction path learn specifically from the frozen model's real first-error and lowest-margin states?

The corrector is a 29,812-trainable-parameter, width-shared ConvGRU. It receives the six legal transition feature planes plus the frozen base's signed confidence and confidence magnitude, then adds learned local and quotient logit corrections. Zero output heads make initialization exactly identical to the base. It performs no exact arithmetic, repair rule, or confidence threshold at inference.

## Matched comparison

Both arms have the same frozen 99,824-parameter base, the same 29,812-trainable-parameter corrector, initialization, optimizer, 300 updates, batch size, width schedule, training-example count, fresh prime identities, and evaluation cases. Both share 200 uniform width-64 training records. At width 128:

- the control uses 600 uniform transitions;
- the challenger uses 300 repeated samples from unique frozen-base first-error transitions and the 300 lowest-margin records selected from one per mined trajectory.

The challenger mines 600 complete width-128 trajectories offline. Repetition is necessary because the measured yield is only 14.5%; the unique first-error count and repetition ratio will be reported. Mining compute is intentionally unmatched and disclosed. Exact bigint operations are limited to this offline label/mining stage and evaluation truth.

Before training, 512 representative cases at each width must reproduce frozen-base local and quotient logits bit-for-bit or the run aborts.

## Gates

Each frozen endpoint is evaluated on 2,000 transitions and 200 autonomous rollouts at widths 64 and 128. Advance only if the challenger reaches at least 90% width-128 rollout, improves by at least three points over the uniform corrector, retains at least 95% width-64 rollout, reaches at least 99.8% width-128 transition, and keeps invalid width-128 rollouts at or below 1%.

This is a selection smoke, not candidate promotion. A pass requires higher-fidelity fresh evidence before independent or checked evaluation. If an accurately trained targeted corrector fails the causal and absolute gates, close this representation rather than tuning its dataset ratio.

## Resource budget

Mining 200 full trajectories took 123.63 wall seconds and 476.96 aggregate CPU seconds with four threads, yielding 29 first errors. Linear projection gives 371 wall seconds for 600 trajectories. Twenty representative corrector updates took 10.64 wall seconds and 41.35 aggregate CPU seconds, projecting 319 wall seconds for two 300-update arms. Setup and the full two-arm evaluation are allocated 610 wall seconds, for a 1,300-second expected total.

The planning budget is 2,000 wall seconds (54% headroom), and the hard runaway ceiling is 3,000 wall seconds or at most 200 aggregate CPU minutes at four continuously occupied threads. Wall and CPU quantities are distinct. The mined dataset, completed checkpoints, and arm results are content-keyed and reusable. Training or evaluation fidelity will not be reduced merely to meet the projection.

## Boundaries

The checked benchmark and private answers remain unread. The corrector is trained-parameter dependent and tensor-native at inference. PyTorch supplies the required convolutional, recurrent, and optimization primitives; GNN libraries do not match this sequential local-correction intervention.

## Result and decision

Exact initialization passed at both widths. Mining 600 trajectories produced 96 unique first-error records in 406.26 wall seconds and 1,579.69 aggregate CPU seconds. The uniform corrector reached 85.5% width-128 rollout, while the first-error corrector fell to 77.5% (−8 points), raised invalid rollouts from 0.5% to 2%, and reduced width-64 rollout from 98% to 95%. Both arms scored 100% on 2,000 fresh width-128 transitions, demonstrating that sampled transition accuracy concealed autonomous damage.

Reject and close this confidence-conditioned additive-corrector representation. Repeating sparse first-error records overweights locally ambiguous corrections and shifts trajectories into new off-distribution states. The preregistered rule forbids ratio tuning after this accurate but harmful result. Total runtime was 1,290.61 wall seconds and 4,971.87 aggregate CPU seconds, within the 2,000-second planning budget and 3,000-second hard ceiling.
