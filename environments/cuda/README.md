# CUDA environment

This package supports Linux x86_64 fresh VMs with NVIDIA GPUs. It does not
install a system CUDA toolkit: the selected PyTorch wheel supplies its CUDA
runtime, while `nvidia-smi` proves that the host driver and GPU meet the
profile floor.

`requirements-cu121.txt` and `requirements-cu124.txt` are complete uv-generated
Python 3.12 locks. Every transitive requirement is pinned and hashed. Torch is
an HTTPS direct wheel URL under the matching PyTorch CUDA index, so PyPI cannot
substitute a CPU build.

On a fresh GPU VM:

```bash
git clone <repository-url> sair
cd sair
environments/cuda/bootstrap.sh --profile cu124
environments/cuda/run-acceptance.sh --profile cu124
```

The bootstrap verifies the lock, installs the exact uv 0.11.28 when necessary
from a hash-pinned installer, installs a uv-managed Python 3.12, syncs the lock
with `--require-hashes`, rejects an insufficient driver/GPU, and records an
environment receipt. The acceptance launcher reruns live preflight and then
executes the bounded representative-family CUDA matrix. Neither script creates
or destroys a cloud instance; provider lifecycle and billing enforcement belong
to the separate external controller.

Use `environments/providers/controller.py` from a separate local process for
budget admission and provider lifecycle. An in-guest timeout or a provider
"stop" state is not cost closure; the external receipt must confirm DELETE.

Use `--dry-run` on `bootstrap.sh` to inspect the selected paths and hardware
floor without downloading or installing anything. A real acceptance run is not
valid until its receipt reports every required test with zero skipped cases.

## Clean-worktree repository validation

The full repository suite also depends on ignored historical `.scratch`
evidence. Before the final clean-worktree test, restore the closed fixture
surface with `tools/restore_external_test_fixtures.py`, passing the development
`.scratch` and its physical EXP651 historical overlay as `--source` arguments.
The restorer links only roots whose tests permit external read-only evidence;
EXP658, EXP663, EXP675, EXP685, and MLflow are copied because their tests
authenticate local path ownership or reject linked files. It refuses divergent
destinations and records its receipt under the ignored target `.scratch`.
