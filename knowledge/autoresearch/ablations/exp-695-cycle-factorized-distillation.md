---
type: Experiment
title: EXP695 C6 cycle-factorized distillation
description: Frozen C6 tournament contract and terminal complete-route resource-admission park.
status: parked
tags: [autoresearch, tournament, convgru, distillation, factorization]
timestamp: "2026-08-01T22:04:48+05:30"
---

# EXP695 C6 cycle-factorized distillation

## Frozen mechanism

EXP695 tests a deterministic Candidate005-derived h48 hard-residue student. Every learned Horner transition executes exactly sixteen visits through one shared ConvGRU core: outward dilations `[1,2,4,8,16,32,64,128]`, followed by inward dilations `[128,64,32,16,8,4,2,1]`. Native PyTorch was selected because deterministic `Conv1d`, `ModuleList`, `einsum`, optimizers, and safetensors cover the mechanism without a new dependency.

The training-only teacher has sixteen physically untied slot cores, initialized by generic tensor copying from the deterministically pruned Candidate005/EXP609 h48 base. Root 20261377 trained that teacher for the frozen 400 updates on generated exact labels, then froze every teacher parameter before either student. The teacher checkpoint is `b8ecc040647333f9415b664e1c1a1b2223a7ede05e0d12ec19dbc4e17874edbe`. No teacher parameter or call enters deployed inference.

For direction `d`, dilation identity `k`, factor rank `r`, and hidden channel `h`, the coefficient is

`c[d,k,h] = tanh(sum_r direction[d,r] * dilation[k,r] * hidden_basis[r,h])`.

The same bounded coefficient modulates the hidden cycle state immediately before and after each shared-core visit. This changes hidden state, never convolution weights, and is therefore materially distinct from C4 kernel adapters. The primary and zero-coefficient control begin from the exact same state hash `f3254f75a7b95dd1bdb44617da23f8fe33909e69dc2648758415508c5ba272e4`; both instantiate and compute identical coefficient tensors, teacher targets, per-slot trajectory losses, data/RNG, optimizer, scheduler, calls, and updates. Only coefficient access is multiplied by exact zero in the control.

Had resource admission passed, the formal package would have completed 2,400 updates per arm, 3,000-case W128/W256 transitions, paired 300-case W128 rollout, sequential W256 blocks `[200,300,450,600]`, and coefficient-zero, direction-swap, dilation-permutation, and mean-coefficient interventions. A comparison would be scientific only with control W128 transition at least 99%; mechanism support would require every intervention to erase at least half the primary-control gain.

## Baseline phase reset

Before C6, exactly two EXP694 metadata defects were normalized. Its metrics evaluation became the repository-valid `not_performed/reason/labels_or_oracle_present:false` envelope, and its run artifacts gained a pointer to the already-hashed EXP609 checkpoint. Metrics SHA-256 changed from `7e96ab0f...c957` to `0b047341...ac27`; run SHA-256 changed from `5b68565d...a587` to `b5208cf9...9619`. The failed decision, consumed root, source/config/code/test hashes, zero denominators, null science, and every evidence meaning are unchanged. Focused tests validate both files against the actual repository schemas.

## Observed verification and resource decision

The preflight found all 87 LFS paths hydrated and byte-identical to the primary workspace, the nested challenge clean at `82510bba`, the source checkpoint at `cbc4a1bc...4bd`, and roots 20261377–85 unopened. The all-mode smoke exercised teacher update, both arms, all sixteen trajectory targets, exact reload, evaluation, the four interventions, teacher-free deployed routing, and serialization in 41.6 seconds. The focused suite passed 22 tests. The exactly-once package oracle passed validation with 230 concepts and 1,100 records, then passed all 1,009 tests in 471.53 seconds.

Root 20261377 completed the 400-update teacher and both matched four-update calibration arms. Each arm had 29,084 parameters, 32 student calls, 32 frozen-teacher calls, 512 coefficient calls, 512 trajectory-slot calls, and 32 exact-loss calls. Both reloaded exactly. Five complete cold 1,100-case primary routes were required because the first three had max/median greater than 1.10. Cold wall times were `[212.907,119.931,129.489,122.158,121.830]` seconds. Outputs were deterministic, peak RSS was 2,262,302,720 bytes, and all checks except runtime passed. The binding guarded wall was `2 * 212.907 = 425.814` seconds, above the frozen 285-second ceiling.

EXP695 is therefore parked for complete-route resource inadmissibility. Root 20261378 and roots 20261379–85 remain unopened. Calibration accuracy was zero on each tiny 16-transition and 2-rollout route for both arms; those denominators are route-integrity evidence only and support no scientific or causal verdict. No full arm training, formal comparison, intervention evidence, confirmation, independent seed, candidate, promotion, or private-score claim exists. C5 remains scientifically unresolved; C7 and the final seven-member adversarial chair remain mandatory.

Canonical records are in `experiments/runs/exp-695-cycle-factorized-distillation/run.json` and `evals/results/exp-695-cycle-factorized-distillation/metrics.json`. Raw checkpoints, progress, complete calibration receipt, and MLflow run `81b94aa9dd8a4d6980b3ae1d15dfe170` remain under `.scratch/exp-695-cycle-factorized-distillation/`.
