---
type: Research Note
title: EXP684 STARS and Existing-Core Random Depth Mapping
description: Authenticated reuse of M026.a STARS and M026.b count evidence on Candidate005's native visits.
status: parked-calibration-resource-admission
tags: [autoresearch, M026, STARS, JVP, loop-curriculum, factorial]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP684 STARS and existing-core random-depth mapping

EXP684 reuses the authenticated EXP669 reading of arXiv:2605.26733 pages 4-8: normalize a
random vector, apply one transition JVP, normalize that product, apply a second JVP, and
penalize its mean squared norm. This is a one-step power/JVP proxy, not an exact spectral
radius. `torch.func.jvp` in CPU PyTorch 2.5.1 is the frozen native implementation.

It reuses only EXP670's authenticated bounded deterministic count-schedule evidence from
arXiv:2604.07822. Unlike EXP670, no extra residual operator, state, parameter, reinjection,
or inference loop is added. A training count changes `SharedDilatedConvGRU`'s existing cycle
count; each cycle traverses its native power-of-two dilation visits, alternating direction
exactly as Candidate005 already does. Two cycles reproduce inherited training/inference
bit-for-bit.

Exact aggregate work is stratified by the frozen width schedule because W64/W128/W256
cycles contain 6/7/8 visits. The balanced random and within-stratum shuffled schedules each
sum to 200 cycles in every schedule position, hence 800 cycles and 5,600 visits versus the
same totals for fixed depth. Counts are sampled once per batch and never depend on examples,
labels, predictions, validity, or correctness.

The 2×2 factorial isolates STARS and random-depth main effects and their interaction.
Shuffling tests count/data association without changing histogram or work. Stop-gradient
computes the same power/JVP value but removes its parameter gradient. Formal inference is
the inherited two-cycle graph, so any scientific difference can come only from trained
weights. EXP669/670 roots remain closed historical evidence; EXP684 uses fresh roots
20261286-20261289.

Root 20261286 verified the complete mapping and equal full-route work, but resource
admission rejected before formal science: 22,655.53 expected wall seconds exceeds 21,600,
and measured variability headroom turns the identical 156-second candidate path into a
312-second guarded projection above 285. Therefore EXP684 provides no W128/W256 causal or
accuracy evidence. The exact package is parked while M026.a/b remain scientifically open.
