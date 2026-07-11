---
type: Architecture Decision Record
title: "ADR-0002: Select a learned Horner transition for the first POC"
description: Prefer a trajectory-supervised local recurrent cell over direct end-to-end digit prediction.
status: accepted
tags: [ADR, POC, TCN, learned-algorithm]
timestamp: "2026-07-11T23:15:00+05:30"
---

# ADR-0002: Select a learned Horner transition

## Context

Official direct neural baselines succeed primarily on fixed tiny primes. The supplied research identifies learned algorithm decomposition as the strongest direction, while a public external model card makes an unverified but concrete claim for a carry-aware Horner TCN.

## Decision

Build a small CPU POC whose outer binary scan is explicit and whose modular state transition is learned from true trajectories. Evaluate autonomous rollout on prime-disjoint splits and require a weight-perturbation control.

## Alternatives considered

- Direct decoder-only Transformer: simple, but the official baseline already exposes its ceiling.
- DLP/additive bottleneck: useful low-tier control, weak scaling evidence.
- ConvGRU/Neural GPU: strong follow-up, but less direct competition-specific evidence.
- Limb recurrence: potentially faster, but should follow a binary control.

## Consequences

The POC can test the core scientific hypothesis quickly and honestly. It cannot establish high-width performance, hidden-seed generalization, or organizer compliance by itself.
