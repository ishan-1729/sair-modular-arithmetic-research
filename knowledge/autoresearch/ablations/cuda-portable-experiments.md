---
type: ablation
title: CUDA-portable experiment adapters
description: Explicit device runtime, closed inventory, and EXP300/EXP699 replay adapter preparation.
status: active
readiness: cuda-inventory-open
timestamp: "2026-08-03T00:00:00+00:00"
---

# CUDA-portable experiment adapters

This branch preserves historical experiment runners and introduces explicit
device adapters for current hardware replays. The inventory remains open while
`adapter-required` families await later packages; this is neither full branch
preparation nor a claim that CUDA has executed successfully.

## T003 boundary

- `experiments/cuda_portability/runtime.py` resolves only CPU/CUDA devices,
  configures deterministic execution, and never changes PyTorch's global
  default device.
- `experiments/cuda_portability/exp300.py` preserves EXP300's Python-bigint
  sampler and exact-label semantics while preparing tensors on an explicit
  device.
- Exact trajectory labels are prepared before the measured recurrent model
  loop. The EXP699 replay converts rollout results to host values only after
  the loop, so semantic evidence remains available without a synchronization
  at every position.
- `experiments/cuda_portability/inventory.json` is generated from every
  `experiments/**/*.py` file. Unported torch routes remain
  `adapter-required`; they are not silently treated as ready or archived.

## Evidence and limitations

CPU parity covers EXP300 feature/loss behavior, prepared training labels, one
frozen-route training update, and a deterministic rollout comparison. Static
inspection enforces the no-transfer recurrent-loop boundary. A real GPU has
not run this code, so kernel support, CUDA memory use, numerical agreement, and
speedup are hypotheses reserved for the later authorization-gated acceptance
matrix.

Regenerate and check the manifest with:

```bash
python -m experiments.cuda_portability.inventory --write
python -m experiments.cuda_portability.inventory --check
```

No long local experiment is required by this package.
