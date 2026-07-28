---
type: Research Note
title: EXP-667 RSM mechanism map
description: Authenticated source claims, bounded SAIR mapping, controls, exclusions, and M016.b deferral.
status: complete
tags: [autoresearch, rsm, recursive-reasoning, mechanism-map]
timestamp: "2026-07-29T02:00:00+05:30"
---
# EXP-667 RSM mechanism map

## Source authentication

The frozen `arxiv-2603.15641.pdf` hashes to `c1cb3a00564e9bc4828c0ddd1b8161d702b32da445eb34920ffe6aab81261c17`. Pages 5–7 (§II.B–D) were extracted, rendered, and visually inspected. They specify two sequence states, one shared transition operator, nested outer/inner recursion, terminal loss, detached warm-up with predominantly one-step gradients, and independently growing H/L curricula.

## Bounded-faithful SAIR adaptation

EXP-667 freezes the exact EXP-609/Candidate-005 proposal and initializes both high and low states from it. A single zero-output convolutional operator is called for every high update, low update, and depth. The zero output gives bit-identical proposal behavior before training. Training samples the independently preregistered H/L curricula, detaches warm-up history with 0.99 probability of a strict one-step outer gradient, and applies terminal-only loss except in the declared deep-supervision control.

Inference has fixed declared H/L depths and receives no labels, verifier result, answer-dependent depth, or repair signal. The controls are one-state, deep supervision, full gradient/no detach, fixed-depth training, high reset, low reset, and randomized transition weights. Stability evidence compares depths 1/2/4 and records separate high-reset, low-reset, and high-versus-low state distances.

## Boundary

This is a convolutional analogue, not a transformer reproduction. Exact bigint arithmetic remains restricted to label generation and evaluation. M016.b RIM is explicitly deferred and receives no implementation or evaluation surface in this package.
