---
type: Research Note
title: EXP682 Huginn Source, Topology, and Determinism Map
description: Visual source authentication and Candidate005 transfer-feasibility decision for ST-M006.a.
status: completed
tags: [autoresearch, Huginn, recurrent-depth, latent-reasoning, source-audit, topology]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP682 Huginn mechanism map

## Authenticated source

The immutable local `arxiv-2502.05171.pdf` hashes to
`96f4543683a06a470ccb0b6b0db27a98f283fde9467d175e838f03a629a76c7b`, matching
the research manifest. Pages 3, 4, 10, and 11 were rendered and visually inspected.
No web source was needed.

Pages 3-4 define the source-complete mechanism. A prelude embeds tokens as `e=P(x)`;
the initial latent is sampled as `s0~N(0,sigma^2 I)`; the shared recurrent block applies
`si=R(e,s(i-1))`; and the coda emits `p=C(sr)`. Every recurrence reinjects `e`. The core
starts with a learned `2h -> h` adapter over `concat(si,e)`. Transformer sublayers use
sandwich RMSNorm, and the recurrent block is rescaled by RMSNorm before the coda.

The training objective samples recurrence per sequence. Equations 1-2 sample
`tau~Normal(log(rbar)-sigma^2/2,sigma)` and then `r~Poisson(exp(tau))+1`, with
`sigma=0.5`; this is heavy-tailed. The main experiment backpropagates only through the
last eight recurrences. The prelude still receives gradient because `e` is reinjected in
the credited suffix. EXP682 preregistered a safety-bounded 2-12 adaptation with target
mean 5; the exact 400-draw root-20261278 stream is reproducible via native
`torch.distributions.Normal` and `Poisson`.

Pages 10-11 describe optional zero-shot adaptive exit, KV-cache sharing, continuous
CoT warm starts, and self-speculative decoding. None is part of EXP682. Formal inference
was frozen to a fixed depth and fixed-seed normal latent, so it would be deterministic and
answer-independent if the transfer gate passed.

## Candidate005 mapping and binding incompatibility

Candidate005 has a useful coarse factorization: `stem` is the prelude; one shared
dilated ConvGRU scan is the recurrent computation; and `output_norm`, `local_head`, and
`quotient_head` are the coda. Its actual microscopic topology is different:

- the recurrent update consumes `concat(GroupNorm(state), raw six-channel input)`;
- there is no independent hidden embedding reinjection or `2h -> h` adapter;
- there is no random latent boundary;
- normalization is GroupNorm, not native sandwich RMSNorm.

The executable probe copies every compatible learned boundary, uses native RMSNorm and
the source concat adapter shape, and shows that the required topology changes checkpoint
outputs. Its latent-seed and RMSNorm interventions are causal. Making those additions
inert would recover only a bypass, not a source-complete transfer, and would fail T084's
latent/component causal gates. GroupNorm and RMSNorm cannot be globally substituted as
an identity transformation because GroupNorm centers each sample whereas RMSNorm does
not.

## De-novo fallback and decision

The frozen fallback gate also fails before root allocation. EXP634 is adjacent rather
than Huginn evidence, but it is the available matched de-novo recurrent-depth feasibility
bound: after 4,000 matched updates its trained-width-12 direct recurrent model achieved
only 2/300 exact (0.667%), below its 20% viability gate. A new source-complete model with
only 400 updates cannot defensibly target T084's 95% W128 gate after that stronger small-
width rung failed. EXP643 is diagnostic context only, and EXP674 excludes M006.a.

Therefore EXP682 parks under the preregistered `no exact transfer / no viable de-novo
route` stop. Roots 20261278-20261281 remain unopened. No training, calibration,
exact-answer science, checked, independent, official/private, candidate, submission,
promotion, or completion evaluation occurred. This is not a rejection of Huginn or of a
future from-scratch Huginn program with a separately justified training budget.
