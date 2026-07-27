---
type: research-note
title: EXP663 bounded-faithful DRM target-token mechanism map
description: Source-faithful claim-to-code, adaptation, exclusion, and rendered-page receipt for the EXP663 target-token denoising falsifier.
status: frozen-before-science
date: 2026-07-28
timestamp: "2026-07-28T18:45:00+05:30"
---

# EXP663 DRM target-token mechanism map

## Source and visual receipt

The local primary source is `research/papers/arxiv-2604.18839.pdf`, SHA-256 `6939dc9065d2eca96be7a7e1bf8a7a3fc4ad6d4ea8dd98c036a25886929c00d8` (26 pages). Text was checked with pypdf and pages 6-10, 20, 22, and 23 were rendered with bundled Poppler and visually inspected. All selected pages were legible, complete, and free of clipping or overlap. Temporary renders are deleted after this receipt is frozen.

Rendered-page SHA-256 values were:

- p6 `8a843b2772dbd8ca314d2538a7da37b772a7aef3bb49918729f942b829d510fc`
- p7 `b3c33058131c9d8d5ad5d1f7a0b97d7910757097e092d06651fcd6d6b4a7a0e3`
- p8 `ae83e4e4013083ec4c354c34b25e63a55bd692ceeb41124e3ef4c331bfd073e0`
- p9 `58e50503f95231dd04ec4babef824e63cfbe50e5ef52a414f726f4f926212baf`
- p10 `9f65296b2e4540ad68fbe928703f9f3408d78c13146550ad009be107c5c3ed12`
- p20 `728f897e0f020d268ff8c8a58520634046113a4c5f47e176856ea4fc4d14e2e9`
- p22 `7c9f8c23d1b71194281395633ea6e59d39ac75bb08fd1d88cbc40413a22776ab`
- p23 `47f7e8c1d0c9efacd0248ab27aab329ad389a73db10dfdbf0415bb530b29f7b2`

## Claim-to-code map

| Source claim | Local implementation | Classification |
|---|---|---|
| p6 Eq. 6: initialize from an embedded corrupted target and backpropagate clean likelihood through a fixed shared `k`-step window | `DRMCell.denoise_window` starts from discrete MASK/0/1 canvas embeddings and returns final clean bit logits after shared recurrent applications | Native DRM mechanism |
| p7 Eq. 8: cosine signal retention and uniformly selected masked tokens | `cosine_mask_ratio` and `corrupt_target` sample `tau` independently of identity and uniformly mask `floor(r(tau) * width)` positions | Native discrete DRM mechanism |
| pp7-8: inference begins fully masked, predicts the complete grid, and uniformly remasks a schedule-sized subset without freezing prior tokens | `generate_and_remask` uses an evaluation-answer-independent generator, full-grid threshold prediction, and uniform partial remasking down to zero masks | Native discrete DRM mechanism |
| p8: standard masked diffusion lacks recurrent weight sharing; a non-recursive masked baseline isolates recursion | `masked_nonrecursive` uses the same canvas-capable superset but one gradient-bearing step and discloses its lower block-pass/FLOP cost | Matched source control |
| p20 Fig. 7: DRM noises the clean target, uses random latent initialization/warm-up, and applies one clean final loss without TRM outer-window deep supervision | training routes use offline clean target corruption, two detached warm-up applications, four differentiable shared applications, and one final BCE | Bounded-faithful adaptation |
| pp22-23: longer gradient windows and detached warm-up steps are material; cosine-uniform masking beats linear and correlated variants in the source | frozen interventions independently switch single-step recursion, warm-up, cosine/linear/all-mask corruption, and remasking | Required causal falsifiers |
| p8 Eq. 9: SPRM injects Gaussian latent noise only at truncation boundaries | no latent Gaussian noise exists in EXP663 | Explicit exclusion; ST-M019.b remains open |

## SAIR adaptations and exclusions

- ARC colour cells become residue bit tokens; the output canvas is one width-sized next-residue bit vector.
- The transition context is the legal six-plane EXP609 state/multiplicand/modulus/input-bit/LSB/MSB representation. Exact arithmetic is used only by offline generators and evaluators.
- The frozen 99,824-parameter EXP609 decoded cellular checkpoint supplies the shared transition operator. A separate MASK/0/1 embedding plus zero-initialized projection makes the canvas extension exactly function-preserving when evaluated as one un-warmed step.
- Binary BCE replaces multi-class grid CE. Thresholding replaces colour argmax. These are explicit task adaptations, not ARC-grid reproduction claims.
- The frozen source route, clean source-initialized recurrent continuation, and non-recursive masked denoiser are controls. Stored parameter supersets match; executed block passes, analytical FLOPs, wall, process/cgroup CPU, and RSS are reported separately.
- Inference never reads targets, expected answers, puzzle identities, or confidence tuned on evaluation answers. It has no exact multiplication, division, modular reduction, lookup, repair, or answer-dependent exit.
- SPRM, Q-head selection, test-time augmentation voting, ARC task embeddings, Gaussian spatial masking, and confidence-based remasking are excluded.

## Frozen decision contract

Root `20261202` profiles every route, checkpoint roundtrip, W128/W256 evaluation, and intervention pathway. If admitted, root `20261203` gives each trained primary/control the inherited 400-update allowance over fresh W64/W128/W256 identities. The primary gate is strictly conjunctive: DRM must retain at least 80% W128 autonomous exactness and remain within five points of the clean recurrent continuation; reach at least 10% W256 autonomous exactness; beat the frozen source and every matched control by at least five points at W256; preserve valid residues; show material trained-weight, recursive-window, cosine-schedule, and remasking dependence; reload exactly; and preserve validation/test separation and admitted resources. Any miss rejects this formulation and leaves roots `20261204` and `20261205` unopened.
