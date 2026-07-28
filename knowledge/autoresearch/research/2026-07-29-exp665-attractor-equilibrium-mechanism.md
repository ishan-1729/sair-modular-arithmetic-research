---
type: Research Note
title: EXP-665 attractor and equilibrium-family mechanism map
description: Page-faithful ST-M021 claim-to-code, solver provenance, SAIR adaptations, exclusions, and frozen gates.
status: preregistered
tags: [autoresearch, exp-665, attractor, equilibrium, torchdeq, anderson]
timestamp: "2026-07-29T04:50:00+05:30"
---

# EXP-665 attractor and equilibrium-family mechanism map

## Source identity and review method

| Source | SHA-256 | Reviewed PDF pages | Role |
|---|---|---|---|
| `arxiv-2605.12466.pdf` | `c379c509c442023dceff9428e41bec4807e86fad8dedd62416d21e3cab921346` | 4-6, 9, 11-12 | Attractor proposal, persistent refinement, Anderson solve, gradient paths |
| `arxiv-2605.21488.pdf` | `986b4a6a150b08a42d2e42fc3e0f5965ff8d654e001dd18f216bf85f959783c6` | 3-9 | EqR finite recurrence, random initialization, noise/damping, depth x breadth, residual selection |
| `arxiv-2601.10679.pdf` | `91a5f399814c0bafc90cc29ad8e2485d23b2aae86fb49de4ce850c4faafdb1b1` | 3-8 | HRM fixed-point violations, spurious/rival attractors, perturbation warning |

The selected pages were extracted with `pdfplumber 0.11.10` and rendered with `pypdfium2` because WSL and the bundled runtime expose no Poppler binary. Every rendered selected page was visually inspected in three contact sheets with no clipping, overlap, missing page, or unreadable figure defect. Render hashes remain in ignored `tmp/pdfs/exp665/` until package cleanup.

## Claim-to-code map

| Source claim | Page evidence | Frozen EXP-665 code analogue | Adaptation / exclusion |
|---|---|---|---|
| A backbone proposes meaningful output-side state `y0`; a separate shared attractor refines it. | Attractor pp. 4-5, Eq. 1-2 | Frozen EXP609 ConvGRU produces `proposal`; `ProposalCorrection` maps `(z, proposal, features)` to the next hidden state. | The equilibrium is ConvGRU hidden state decoded by the frozen head, not a tied language-token embedding. This is a bounded SAIR adaptation, not native Transformer reproduction. |
| Persistent proposal injection prevents a proposal-independent fixed point. | Attractor p. 5 Eq. 2 and p. 11 Table 4 | Every correction call includes the immutable proposal and returns `proposal + bounded_delta`. | Additive output injection is used instead of token concatenation; no input or label is introduced after proposal construction. |
| Solve `A(z;z0)=T(z,z0)-z=0` with Anderson, relative residual tolerance, and bounded maximum iterations. | Attractor p. 5 Eq. 3, p. 6 Alg. 1 | TorchDEQ Anderson with `max_iter=6`, `tol=1e-3`, history 4 and regularization `1e-4`; plain Picard and fixed-depth controls share the operator. | Target-free lowest-residual selection is library behavior. Iterations are hard bounded before candidate timing. |
| Implicit differentiation is exact in principle; one-step/phantom gradients are cheaper and effective. | Attractor p. 6 Eq. 4/Alg. 1 and p. 9; pp. 11-12 ablation | Matched Anderson phantom-one-step and Anderson implicit-gradient routes; the latter uses a bounded Anderson backward solve. | No claim of exact equilibrium gradients when the bounded backward solve has residual error. |
| EqR treats repeated updates as an attractor landscape; correct and spurious attractors may coexist. | EqR pp. 3-6 | Finite EqR route and offline convergence/basin diagnostics measure residual against disjoint exact correctness. | Residual is diagnostic and target-free at inference; exactness labels never select an inference result. |
| Random initialization and per-step Gaussian noise with damping can broaden basin coverage. | EqR pp. 6-8, Eq. 2 / Alg. 1 | `eqr_finite` uses random initialization, damping `0.05`, noise `0.01`, four finite steps, and breadth 2 at evaluation. | Persistent frozen proposal conditioning remains present; breadth chooses minimum residual, not oracle-correct output. |
| Depth and breadth are complementary, and breadth only helps once useful basins exist. | EqR pp. 5, 8-9 | Fixed depth, Picard, Anderson, and EqR breadth routes disclose calls and iterations; W128 viability gates all scaling. | No unbounded or answer-dependent test-time scaling. |
| Low residual can correspond to a wrong/spurious fixed point, and arbitrary latent perturbation can destroy coherence. | HRM pp. 4-8 | Disjoint validation calibrates residual/correctness AUC; controlled state-scaled proposal perturbations measure basin return and exact recovery. | Sudoku conflict energy, relabeling, checkpoint voting, input transformations, and answer-dependent success/halting are excluded. |

## Maintained-solver provenance and compatibility

The admitted research-only dependency is `torchdeq==0.1.0`, pinned in `pyproject.toml`. Isolated Playwright-managed Chromium `1.61.1` / regular Chromium `149.0.7827.55` observed the primary [TorchDEQ repository](https://github.com/locuslab/torchdeq), [MIT license](https://github.com/locuslab/torchdeq/blob/main/LICENSE), and primary `setup.py` declaring version `0.1.0`, MIT, and a PyTorch library for DEQs. The visible latest commit is 2023-10-31, so this package treats TorchDEQ as mature and source-auditable but explicitly does **not** claim recent upstream activity. The PyPI wheel metadata declares `torch>=1.11.0` and `numpy>=1.21.5`; installation used `--no-deps`, preserving CPU PyTorch `2.5.1`. Deterministic affine-contraction probes produced bit-identical finite Anderson forward/backward outputs and gradients for phantom and implicit modes.

TorchDEQ supplies the Anderson and backward-gradient implementations; EXP-665 does not hand-roll quasi-Newton mixing. Plain finite recurrence is retained only as the source-required simple control.

## Frozen experiment identity

- Integrated base: `c042a7fa68ab143dbba0091928b475ce5de1bf65` on `codex/exp665-attractor-equilibrium-family`.
- Frozen checkpoint: `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`.
- Roots: calibration `20261210`, viability `20261211`, confirmations `20261212` and `20261213` only after a complete pass.
- Widths: W64/W128/W256; 512/128/128 disjoint prime identities per width.
- Trained routes: fixed-depth shared, plain-Picard phantom, Anderson phantom, Anderson implicit, and finite EqR; each receives 400 matched updates on schedule `[64,128,256,128]`. Proposal-only is immutable.
- Full formal counts: 3,000 transitions and 300 rollouts per test width/route; 1,000/100 validation cases; W128 primary exactness first.
- The preregistered primary is Anderson phantom, matching the source's small-data recommendation. No post-result route selection is permitted.

## Gates and exclusions

W128 requires at least 85% exact autonomous rollout, zero invalid rollouts, at least 99.8% transition exactness, no more than two points below proposal retention, material trained-weight dependence, mean relative residual at most 0.05, residual/correctness AUC at least 0.60, at least 75% perturbed-state basin return, exact reload/separation, admitted resources, and projected candidate inference under 300 seconds. Only a full W128 pass opens W256, where primary exactness must be at least 10% and exceed proposal-only and fixed-depth by at least five points. Both fresh confirmation roots must reproduce every conjunct before an intermediate advance.

Exact bigint arithmetic and exact correctness are confined to offline generation, training labels, tests, and evaluation. Inference consumes feature tensors, learned states, frozen rules, and target-free residuals only. It contains no modular multiplication, division, reduction, CRT, lookup, constructed program, identity feature, label-conditioned stopping, oracle selection, or output repair. Candidate 005, submission, checked/public-independent, official/private, and promotion surfaces are unchanged.
