---
type: Research Note
title: EXP-650 maintained Mamba-2 CPU library readiness
description: Dated visible primary-source evidence and local deterministic CPU feasibility for the ST-M014 transition package.
status: complete
tags: [autoresearch, exp-650, mamba-2, transformers, cpu, playwright]
timestamp: "2026-07-27T21:20:00+05:30"
---

# EXP-650 maintained Mamba-2 CPU library readiness

## Question and method

Does a maintained pinned implementation expose a faithful Mamba-2 operator that
can execute deterministic forward, backward, and checkpoint reload on the
project's CPU-only PyTorch environment?

All public pages below were opened read-only on 2026-07-27 with the repository's
exact `playwright==1.61.1` and managed regular Chromium `149.0.7827.55` in new
headless mode. Each URL used a fresh non-persistent context at 1280x800, service
workers were blocked, and non-read-only methods were aborted. Requested content
rendered visibly with HTTP 200 except the version-shaped Hugging Face docs URL,
which visibly returned 404; the current main docs page rendered with HTTP 200.
Every task-owned context and browser closed. Bounded ignored evidence is retained
under `.scratch/exp-650-library-backed-mamba/browser/`.

## Visible primary evidence

- [Tiny Recursive Reasoning with Mamba-2 Attention Hybrid](https://arxiv.org/html/2602.12078)
  visibly specifies `Mamba-2 -> Mamba-2 -> Attention -> MLP` for
  `TR-mamba2attn`, and describes `MLP-t` as transposed-sequence all-position
  mixing. It states that residual connections use post-norm `norm&add` and that
  pure causal Mamba was not used because cross-position bidirectional mixing is
  required.
- [Official state-spaces/mamba repository](https://github.com/state-spaces/mamba)
  visibly identifies `modules/mamba2.py` as the Mamba-2 block and lists a core
  package install without the CUDA selective-scan extension. Its visible Mamba-2
  usage example still places the module on CUDA.
- [PyPI mamba-ssm](https://pypi.org/project/mamba-ssm/) visibly reported
  `2.3.2.post1`, released 2026-05-09.
- [PyPI transformers](https://pypi.org/project/transformers/) visibly reported
  `5.14.1`, released 2026-07-16.
- [Pinned Transformers Mamba-2 source](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/mamba2/modeling_mamba2.py)
  visibly contains `Mamba2Mixer`, optional fast-path detection, and a
  `torch_forward` implementation. This is source evidence for the maintained
  fallback used here, not a performance claim.

## Local observed feasibility

The official `mamba-ssm==2.3.2.post1` source distribution did not prepare
metadata under Python 3.12 / `torch==2.5.1+cpu`: release setup warned that NVCC
was absent and raised while parsing `torch.version.cuda == None`. This makes that
release unusable as the project's pinned CPU path without patching upstream or
inventing a substitute.

The permitted maintained fallback, `transformers==5.14.1`, imported
`Mamba2Config`, `Mamba2Model`, and `Mamba2Mixer`. A deterministic CPU fixture
with shape `[2,8,16]` completed forward/backward in `0.1205` wall seconds; input
and parameter gradients were finite; a serialized/reloaded model produced
bit-identical output (`max_abs_reload_diff=0.0`). Transformers explicitly logged
that optional fast kernels were absent and selected the naive implementation.

The package therefore uses the pinned Transformers `Mamba2Mixer` CPU
`torch_forward` path. It does not claim parity with CUDA/Triton performance.

## Mechanism mapping and limitations

EXP-650 composes two distinct library `Mamba2Mixer` instances, then PyTorch
global attention and a token MLP. Every component uses residual addition followed
by `nn.RMSNorm`. The trained faithful arm is forward-causal inside its two Mamba
operators but obtains bidirectional cross-position communication from attention,
matching the paper's stated reason for the hybrid. Reverse and two-direction
averaging are frozen inference interventions. An implemented MLP-t alternative
proves the paper's other operator shape but is width-specific and is not the
scientific challenger.

CPU fallback execution is scientifically suitable for this bounded transition
test but does not establish candidate inference viability. Exact arithmetic is
used only for offline Horner labels and evaluation truth.

## T054 full-route resource observation

The hermetic verification boundary passed before calibration: 34/34 inherited
cone nodes and 405/405 complete project tests. Root `20261186` then exercised
the frozen full-fidelity CPU route without shortening its training or evaluation
contract. Linux globally OOM-killed Python during the first Mamba-hybrid arm,
after its initial checkpoint and before one arm completed. Kernel evidence
records 7,400,644 KiB anonymous RSS on a WSL instance with 7,928,476 KiB total
memory. Consequently, representative throughput, variability, ETA, and dynamic
expected/soft/hard guards could not be estimated. This observation rejects
resource admission for this exact route; it is not an accuracy or mechanism
result.
