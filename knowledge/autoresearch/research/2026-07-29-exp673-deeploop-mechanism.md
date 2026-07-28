---
type: Research Note
title: EXP-673 DeepLoop mechanism and topology-feasibility map
description: Visual PDF authentication, exact source equations, Candidate005 topology mapping, exclusions, and terminal feasibility decision.
status: parked
tags: [autoresearch, exp-673, DeepLoop, LayerNorm, topology, feasibility]
timestamp: "2026-07-29T09:15:00+05:30"
---

# EXP-673 DeepLoop mechanism map

## Authenticated sources and visual review

The local DeepLoop PDF `research/papers/arxiv-2607.13491.pdf` has SHA-256
`15f83b1ad3a092e00a2e2a7afe3cea91834b11ce2913653836e868673023af4d`. Pages 4-13 and 23 were
rendered and visually reviewed. The supporting LayerNorm-dynamics PDF
`research/papers/arxiv-2607.10681.pdf` has SHA-256
`3509f28429ce0c01ebf7a9fca94d70440a33d69073da9e248ddd8b65b2150f9e`; pages 2 and 4-18 were
rendered and visually reviewed. Rendered pages and extracted text are retained under
`tmp/pdfs/exp673/`. The paper-library manifest, verification record, and checksum-list hashes are
`f2e2e386...f09c21`, `cdf94b67...f58d`, and `a4f793a7...25b8` respectively.

## Exact DeepLoop claims and equations

| Source | Claim/equation | Frozen EXP673 interpretation |
|---|---|---|
| DeepLoop p.4, Eqs. 2.2-2.6 | DeepNorm is PostNorm: `x_next = Norm(alpha*x + f(x;theta))`; the useful condition is `M(beta/alpha)^2 = O(1)`. | PostNorm placement and a real residual skip scale are load-bearing, not naming conventions. |
| pp.5-7, Eqs. 3.1-3.10 | For shared depth, `N=K*R`, `M=2N`, `alpha=(2N)^(1/2)`, `beta=(8N)^(-1/2)`, `beta/alpha=1/(4N)`, and aligned visits require `p=1/2`. | `beta` scales selected residual-branch matrices once at initialization. The paper explicitly says it is not a runtime multiplier. |
| p.6, Eqs. 3.5-3.8 | Tied perturbations contain the product of the summed output sensitivities and summed effective updates; `kappa_R` ranges from decorrelated `O(1)` to aligned `Theta(R)`. | A causal test must change the proactive parameterization, not react to a failed visit or select a depth from answers. |
| pp.8-10, Eqs. 4.1-4.7 | Hierarchical forward depth `M=2C(K_H+C_L K_L)` differs from one-step-gradient-visible `M_g=2(K_H+C_L K_L)`; H has one gradient-visible visit while aligned L revisits bind at `p=1/2`. | H/L depths apply only when the topology has separate modules and outer-state detachment. |
| pp.11-13, Table 1-3 | The paper reports matched single-seed language-model benefits for `R>=3` and an HRM ARC-AGI improvement with `M_g=24`, `N_g=12`. | Source results are claims, not SAIR evidence. |
| p.23, Fig.6 | At fixed `R=3`, training becomes reliable at `p>=0.5`; larger p is less aggressive. | No post-result p sweep was authorized. |

DeepLoop introduces no gate, learned residual scalar, or additional tuning constant (p.8). This excludes
using a zero outer gate to call an otherwise generic correction exact.

## Supporting LayerNorm-dynamics diagnostics

ST-M030.a is diagnostic only. The supporting PDF studies a different PreNorm recurrence
`h_next=Abar*h+f(h,e)`. It argues that normalization gives an inverse-activation-scale Jacobian,
distinguishes spectral radius from operator norm, and observes nonnormal transient amplification:
converged small random models can have `rho_spec(J_G)<1` while `||J_G||_2>1` (pp.8-12). Its recommended
diagnostics are residual-verified small-state full Jacobians/JVPs, spectral margin, singular gain,
nonnormal gap, transient gain, activation RMS and minimum per-token spread. Pages 15-18 further show
that nonlinear recurrence, not a constrained diagonal carry, supplies cross-channel depth memory.
These observations cannot substitute a trainable DeepLoop component.

## Candidate005 topology and exact depth accounting

The frozen checkpoint SHA-256 is
`cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`. Its authenticated topology is
`SharedDilatedConvGRU` (`experiments/exp_220_width_independent_cellular/model.py`, SHA-256
`9b271fd1...99edc`). Each visit first applies GroupNorm to the state for the gate/candidate branches,
then returns `y=(1-u)*x+u*c`. There is no normalization after the residual combination, no pair of
attention/MLP PostNorm residual sublayers, no high module, and no truncated outer gradient.

The single shared cell is fully gradient-visible on every visit. Its width-dependent schedule has
14 visits at W128 and 16 at W256. A purely numeric analogy would give:

| Width | visits treated as N | alpha | beta | beta/alpha |
|---|---:|---:|---:|---:|
| 128 | 14 | 5.2915026221 | 0.0944911183 | 1/56 |
| 256 | 16 | 5.6568542495 | 0.0883883476 | 1/64 |

This analogy is not a legal implementation: the shared tensors cannot receive both one-time
initialization gains, and selecting beta at runtime would contradict Eq.3.1's initialization semantics.
Calling the lone cell L and setting H to zero does not create the hierarchical source topology.

## Feasibility decision

There is no bounded-faithful exact-function transfer on Candidate005's existing graph:

1. Adding source PostNorm changes arbitrary pretrained ConvGRU outputs and cannot preserve them exactly.
2. Bypassing that PostNorm or wrapping it in a zero output gate preserves the function but removes the
   load-bearing source operation; DeepLoop expressly introduces no such gate.
3. Writing `alpha*x + (F_source(x)-alpha*x)` is algebraically exact but makes alpha cancel identically,
   so alpha/beta/component dependence is untestable.
4. Adding a zero-initialized correction multiplied by `beta/alpha` is generic runtime adapter scaling,
   not DeepLoop's one-time initialization, and is the excluded EXP644-like substitute.
5. Width-conditioned parameter copies or runtime weight rescaling alter the shared model identity and do
   not provide one call-identical source continuation.

The T069 stop rule therefore binds before implementation or calibration. Roots `20261242`-`20261245`,
all 400-update arms, exact labels, formal W128/W256, Jacobians, candidate timing, and confirmations remain
unopened. This is a topology-faithfulness park, not evidence against DeepLoop, LayerNorm dynamics, or a
from-scratch PostNorm recurrent architecture.

## Frozen package that was not opened

Had topology passed, the arms were frozen as full DeepLoop, alpha/beta-only, visit-alignment-only,
fixed-scale/exponent-zero, matched source continuation, frozen, trained reset, and randomized. All trained
arms would receive 400 updates with matched legal examples/optimizer/formal calls. The complete gate was
85% W128 exact rollout, zero invalids, no more than two points behind source, material trained/component
dependence plus a preregistered favorable gradient/activation/small-state full-Jacobian or nonnormal
diagnostic, exact reload/separation/determinism, W256 at least 10% and five points above every named
control, identical calls, admitted resources, and guarded candidate time at most 285 seconds on both
confirmations.
