---
type: Mechanism Map
title: EXP687 Compact Distillation Mechanism Map
description: Auditable Candidate005 channel-transfer, teacher-distillation, inference-boundary, and native-library decision for EXP687.
status: parked-package-verification
tags: [autoresearch, compact-model, distillation, channel-pruning, W256]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP687 compact distillation mechanism map

The source is the frozen Candidate005/EXP609 `SharedDilatedConvGRU` checkpoint
`cbc4a1bc...d4bd`. The compression map uses trained parameter values only. It does not inspect
examples, targets, correctness, benchmark identity, activations, or teacher outputs. Eleven
incident terms cover stem output, state/output norms, both gate output halves and recurrent
input columns, candidate output and recurrent input columns, local-head input, quotient hidden
rows, pooled mean/max input columns, and quotient output input columns. Per-term mean
normalization prevents a large tensor from dominating solely by element count.

The frozen ranked channel list is:

`75,49,21,85,2,33,90,55,27,80,36,39,43,46,6,71,8,63,76,57,4,15,23,51,54,83,44,10,17,87,74,67,19,30,34,7,62,89,37,82,22,86,79,28,31,16,56,69,25,41,84,11,61,47,78,88`.

Its canonical map hash is `92d31507...8cc0`; the coherently sliced initial state hash is
`f9938a97...b340`. Every source tensor is accounted for, including doubled gate rows, hidden
recurrent columns plus untouched six feature columns, pooled mean/max quotient columns, and
all biases. Both trained arms load these exact bytes.

Distillation is training-only. Both arms always execute the same frozen hidden92 teacher on the
same learner-state feature tensor, compute the same exact label loss, normalized residue-logit
MSE, and quotient KL, and make eight learner calls. Only scalar coefficients differ. Formal
candidate routing loads the compact checkpoint alone and calls the learned rollout; neither the
teacher nor a pruning operator enters inference.

Native PyTorch indexing was retained. This is one fixed graph with a single immutable map, and
`index_select` exposes every selected source axis directly, has CPU/sandbox parity with the
existing model stack, and adds no packaging dependency. There was no measured residual need
that justified a general pruning library. The real smoke validated the chosen mapping and
runtime path, but the package was parked at the unrelated full-suite fixture boundary before
resource calibration or scientific evidence.
