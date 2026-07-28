---
type: Research Note
title: EXP-674 latent-thought regularization mechanism map
description: Authenticated paper equations, Candidate005 topology map, exact exclusions, and terminal source-faithfulness decision.
status: complete
tags: [autoresearch, latent-thoughts, cosine, recurrence, topology]
timestamp: "2026-07-29T11:29:30+05:30"
---
# EXP-674 latent-thought regularization mechanism map

The immutable `arxiv-2502.17416.pdf` is arXiv v1, CC-BY-4.0, 27 pages, and hashes to
`a3aed7457446fa4c11c5f6cac89c38725ca42d0a66c44cc1935e2f8b6376882e`, matching the
manifest, verification ledger, and checksum ledger. PDF pages 2, 3, 4, and 10 were rendered and
visually inspected. Poppler was unavailable in WSL, so the installed `pypdfium2` renderer was used;
the selected PNG hashes are recorded under `tmp/pdfs/exp674/`.

## Exact source mechanism

Pages 2-4 define a `k`-layer block looped `L` times, written `(k tensor-product L)`. Page 10
Section 4, Eq. 3-4 does not regularize hidden trajectories. It starts with one ordinary untied
`L`-layer model, groups corresponding weight tensors in successive `k`-layer blocks, and adds their
cosine term to cross-entropy, averaged over parameter groups. The paper's concrete emulation uses
`L=24`, `k=4`, and `lambda_reg=10`; `lambda=0` is ordinary training. There is no external
teacher/reference model, hidden-state target, or detach instruction: both corresponding untied
weight tensors are ordinary jointly trained model parameters. The printed sign and the prose's
"closeness" wording are ambiguous, but the Candidate005 topology result below is sign-independent.

## Candidate005 topology and falsifier

The frozen EXP609 checkpoint hashes to
`cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`. Its
`SharedDilatedConvGRU` has 99,824 parameters and one `recurrent_update` tensor set reused at every
dilation visit and width. W64/W128/W256 execute 12/14/16 visits. There are no untied corresponding
depth blocks to pull together: the literal Eq. 3-4 image compares each tensor with itself.

The executable feasibility oracle measures self-cosine distance
`1.9637069748057456e-15` and maximum parameter gradient `2.220446049250313e-16`, both below the
frozen `1e-12` numerical-zero tolerance. Independent reloads reproduce the exact 16-key state
schema, all tensors, call counts, logits, and quotient logits bit-for-bit at all three widths.

A training-only untied-depth reference plus hidden-trajectory cosine would instead be
representation distillation. It invents an external reference, hidden boundary, target, and detach
semantics absent from Eq. 3-4. Treating that as "paper-faithful" would violate one-factor isolation;
adding untied parameters to formal inference would violate the checkpoint/call identity gate.

## Decision, frozen route, and exclusions

Park before training under T070's source/topology-faithfulness stop rule. The counterfactual seven
arms, lambda 10, 400 updates/6,400 examples per trained arm, identities, W128-first gates, and roots
20261246-20261249 were frozen, but every root remains unopened. Calibration, formal ETA/admission,
exact-answer evaluation, W128/W256, alignment interventions, candidate timing, and confirmations
are null. M006.a random latent/input reinjection, heavy-tailed depth, and TBPTT remain excluded.

This is not a scientific rejection of the paper, ST-M006.b on an untied architecture, or a future
separately named hidden-state distillation experiment. Candidate005 and all checked, independent,
official/private, submission, promotion, and Goal-completion evidence remain unchanged.
