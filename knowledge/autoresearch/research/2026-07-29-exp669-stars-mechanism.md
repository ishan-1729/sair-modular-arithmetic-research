---
type: Research Note
title: EXP-669 STARS JVP mechanism map
description: Authenticated STARS claims, native JVP proxy mapping, controls, costs, and exclusions.
status: complete
tags: [autoresearch, stars, jvp, recurrent-stability]
timestamp: "2026-07-29T04:00:00+05:30"
---
# EXP-669 STARS JVP mechanism map

The immutable STARS v1 PDF hashes to `20791f73ae2d2f58b00150acd17f106177493327898277bab4efb90080e2d8db`, matching the manifest/checksum ledger. Pages 4–8 were extracted, rendered, and visually inspected.

## M026.a claim-to-code map

Pages 6–7 define a nonlinear transition `h(t+1)=Phi(h(t))`, initialize a random vector, normalize a one-step Jacobian-vector power update, and penalize the batch mean squared norm of a subsequent JVP at the current trajectory state. `STARSCell.jvp_proxy` uses native `torch.func.jvp`: normalize `v0`, compute `Jv0`, detach and normalize it to the one-step direction `v1`, then compute differentiable `Jv1` and its per-sample squared norm. Autograd retains higher-order gradients from this proxy to recurrent parameters. The boundary is frozen immediately before the source transition's final recurrent update. This is explicitly a noisy one-step proxy, never an exact spectral radius, eigendecomposition, or contraction proof.

The primary uses paper lambda `0.1` in `(1-lambda)*base_loss + lambda*proxy`. Lambda-zero performs identical continuation and computes the same two JVPs but weights the proxy by zero. The state-norm control replaces the proxy with activation squared norm while computing a stopped proxy for cost visibility. The random/stop-gradient control uses a normalized random direction and detaches its JVP scalar. Frozen-source and randomized-weight interventions test continuation and trained-weight dependence.

All trained arms receive 400 updates, identical examples, identities, optimizer, source transition calls, and starting checkpoint bytes. STARS/lambda-zero perform two JVPs per example/update; state/random controls compute the same JVP forwards but do not carry higher-order parameter-gradient work. This unmatched backward cost is measured and disclosed.

## Exact inference boundary and exclusions

`STARSCell` inherits the frozen `SharedDilatedConvGRU.forward` without override and adds no parameters. Training helpers are unreachable from inference. Parameter keys/count, recurrent calls, logits, and quotient logits are bit-identical at initialization. Inference has no JVP, vector, proxy, target, correctness, validity filter, repair, lookup, adaptive depth, or programmed arithmetic.

M026.b random loop-count sampling is closed as the separately queued M018.a mechanism. No language-model reproduction, checked/private evaluation, exact Jacobian construction, or submission change is claimed.
