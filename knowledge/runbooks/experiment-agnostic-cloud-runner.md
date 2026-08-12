---
type: Implementation Runbook
title: Run experiments on disposable cloud compute
description: Provider-neutral contract for packaging, canarying, launching, checkpointing, resuming, cleaning, and finalizing arbitrary experiments without rebuilding the control plane around each experiment.
status: active
tags: [runbook, experiments, cloud, gpu, checkpointing, reproducibility, lifecycle]
timestamp: "2026-08-12T00:00:00+05:30"
---

# Run experiments on disposable cloud compute

## Purpose

Use one deterministic lifecycle for every conforming experiment. The lifecycle owns source identity, provider resources, budgets, checkpoints, cleanup, and durable publication. The experiment owns scientific semantics: model, data, seeds, roots, training schedule, evaluation denominators, metrics, and promotion decisions.

This separation prevents a new agent from rebuilding a provider launcher around the experiment currently in view. The control plane must remain unchanged when the scientific command changes.

The public interface is:

```text
python -m tools.cloud_experiment --help
```

The optional personal `$run-cloud-experiment` skill is only an operating guide and launcher shim. It contains no provider API logic.

## Architecture

```text
experiment manifest
  -> schema and semantic validation
  -> deterministic source package + identity receipt
  -> provider doctor + worst-case budget admission
  -> one owned allocation
  -> exact hardware gate
  -> canary command
  -> ordered experiment phases
  -> immutable checkpoint generations
  -> unconditional owner-scoped cleanup
  -> verified output publication
```

The core is under `tools/cloud_experiment/`:

- `manifest.py` rejects ambiguous paths, secret-like environment fields, duplicate phases, and timeouts that consume the cleanup reserve.
- `package.py` selects declared files, records dirty included bytes, builds a deterministic archive, and independently reopens it.
- `providers.py` defines the provider protocol and a complete offline `LocalProvider` used for lifecycle proof.
- `checkpoints.py` publishes immutable generations and verifies pointer, metadata, archive, and restored-tree hashes.
- `lifecycle.py` owns authorization, admission, locking, allocation, canary, phase reuse, cleanup, and finalization.
- `cli.py` is the only supported operator surface.

Do not put scientific conditionals or provider HTTP/S3 calls in a skill, heartbeat, or experiment prompt. Scientific variation belongs in a manifest/command. Provider variation belongs behind the provider protocol.

## Manifest contract

Start from `configs/cloud_runner/manifest.example.yaml`. Validate against the exact machine schema:

```bash
python -m tools.cloud_experiment schema
python -m tools.cloud_experiment validate --manifest <path>
```

Every manifest has six boundaries:

1. `experiment` supplies only a stable id and description.
2. `package` declares source globs, exclusions, and whether selected bytes must be clean.
3. `execution` declares a canary and ordered phases. Each command supplies its own timeout, required output files, and checkpoint paths.
4. `provider` selects a driver and exact expected hardware. Options are provider configuration, never credentials.
5. `budget` sets total USD, provider lifetime, cleanup reserve, and idle ceilings.
6. `artifacts` selects separate run and publication directories inside the worktree.

The manifest must not contain API keys, S3 credentials, tokens, passwords, private keys, or shell snippets that retrieve them. The validator rejects secret-like environment names. A provider driver reads only its documented protected environment, and only after paid authorization passes.

Paths are POSIX-relative and cannot contain `..`, backslashes, absolute roots, or symlink package inputs. Run and publication directories must differ.

## Adapt an experiment once

An experiment becomes portable by accepting the generic runtime environment:

```text
SAIR_CLOUD_RUN_ID
SAIR_CLOUD_PHASE_ID
SAIR_CLOUD_CONTROL_DIR
SAIR_CLOUD_WORKSPACE
SAIR_CLOUD_CHECKPOINT_DIR
SAIR_CLOUD_MANIFEST_SHA256
SAIR_CLOUD_RESUME_ARCHIVE     # only when a verified generation exists
SAIR_CLOUD_HEARTBEAT_PATH     # touch atomically during long commands
```

The command may ignore variables it does not need. It must keep its existing scientific config authoritative. Do not copy seeds, root lists, denominators, update counts, or model settings into the cloud controller.

Commands shorter than `budget.max_idle_seconds` need no explicit heartbeat. Longer commands must atomically touch `SAIR_CLOUD_HEARTBEAT_PATH` before that interval elapses. Provider drivers enforce the same idle contract remotely; silence is treated as a failed phase and triggers checkpoint/cleanup handling.

For a phase that can run longer than the acceptable loss window, make the experiment write reloadable state under declared checkpoint paths and periodically call:

```bash
python -m tools.cloud_experiment checkpoint publish \
  --path checkpoints \
  --path progress.json
```

The controller automatically publishes declared paths after successful phases and, when `checkpoint_on_failure: true`, after failures when files exist. The experiment is responsible for making each published state internally resumable. The controller is responsible for immutability and byte identity.

Make phase boundaries meaningful. A phase receipt is reusable only when it represents a coherent completed unit. Do not create thousands of one-update phases to simulate checkpointing.

## Provider driver contract

A provider driver is a `module:Class` implementing:

- `doctor()` — return availability, paid classification, finite hourly/fixed cost, and non-secret diagnostics without mutation;
- `inventory(owner_nonce=None)` — return all live resources when `None`, or resources for one owner;
- `allocate(owner_nonce, run_id, deadline_epoch_seconds)` — create exactly one lease and return observed hardware/rate facts;
- `run(lease, argv, cwd, environment, timeout_seconds)` — execute one command and return bounded stdout/stderr plus timing;
- `cleanup(lease)` — idempotently delete the exact known lease and prove absence;
- `cleanup_owner(owner_nonce)` — reconcile response-loss/uncertain creation and delete every resource carrying the exact owner nonce.

The driver must fail closed on incomplete provider facts. It must never substitute guessed GPU, region, price, interruptibility, or volume facts. Provider-specific API drift is repaired in the driver and its contract tests, not in an experiment.

Before enabling a paid driver, prove the same lifecycle with injected/fake responses for at least:

- empty and non-empty inventory;
- allocation success;
- response loss after resource creation;
- wrong hardware;
- canary failure;
- command timeout;
- checkpoint publication and corrupted restore;
- controller interruption;
- cleanup retry and exact absence;
- completion publication.

Then run one short paid canary. Do not use a scientific run as the first provider integration test.

## Standard workflow

### 1. Validate and inspect

```bash
python -m tools.cloud_experiment validate --manifest <manifest>
python -m tools.cloud_experiment doctor --manifest <manifest>
```

`doctor` proves importability and finite cost admission; it does not prove allocation or hardware.

### 2. Package once

```bash
python -m tools.cloud_experiment package --manifest <manifest>
python -m tools.cloud_experiment verify-package --manifest <manifest>
```

The receipt binds the normalized manifest, repository commit, every selected source file, dirty selected paths, and deterministic archive. Rebuild when any bound byte changes. Never edit a package receipt or swap an archive underneath it.

### 3. Authorize paid work

Unpaid providers reject authorization arguments. Paid providers require an exact seven-field authorization receipt before the driver is loaded:

```text
schema_version
run_id
manifest_sha256
provider_driver
maximum_total_cost_usd
maximum_provider_lifetime_seconds
expires_at_utc
```

Use `authorization-template` to show the exact fields, but the human grants authorization. Do not infer approval from a prior manifest, account balance, or conversation about another run.

### 4. Canary before science

```bash
python -m tools.cloud_experiment canary \
  --manifest <manifest> \
  --authorization <receipt-if-paid>
```

The canary allocates once, checks observed hardware, runs the declared command, and cleans up. It must exercise the actual image/dependency/runtime path required by science. Keep it short, but do not replace exact hardware acceptance with a provider catalog claim.

### 5. Launch exactly once

```bash
python -m tools.cloud_experiment launch \
  --manifest <manifest> \
  --authorization <receipt-if-paid>
```

The run-directory lock prevents another controller. The provider inventory must be empty before allocation. The owner nonce is durable and every resource must carry it. The controller reruns the canary on the scientific lease, executes ordered phases, and always enters cleanup.

### 6. Observe events

```bash
python -m tools.cloud_experiment status --manifest <manifest>
```

`status` reads local receipts without provider access. Report only new phase/checkpoint/terminal/cleanup evidence. Do not poll provider APIs or full logs on a fixed short cadence. Use the phase timeout, checkpoint interval, and completion marker to choose the next observation.

### 7. Resume only from verified state

```bash
python -m tools.cloud_experiment resume \
  --manifest <manifest> \
  --authorization <new-current-receipt-if-paid>
```

Resume requires:

- the same normalized manifest hash;
- the same package hash and repository commit;
- deletion-confirmed prior resources;
- a readable workspace;
- a latest checkpoint whose pointer, metadata, archive, and restored files all match;
- completed phase receipts with the same manifest/package identity.

It restores the newest generation, reruns the canary on the new lease, skips completed phases, and continues at the first incomplete phase. Changing code, configuration, dependencies, hardware requirements, or phase semantics creates a new manifest/package/run identity; it is not a resume.

### 8. Clean up and finalize

Cleanup is automatic after completion, failure, timeout, or interruption. If external termination prevented the controller's `finally` block:

```bash
python -m tools.cloud_experiment cleanup --manifest <manifest>
```

Cleanup is idempotent and owner-scoped. Never delete a resource lacking the saved owner identity.

After lifecycle status is `completed` and cleanup proves exact absence:

```bash
python -m tools.cloud_experiment finalize --manifest <manifest>
```

Finalization verifies every phase receipt and required output, copies outputs into a new publication directory, and binds the manifest, package, repository, lifecycle, checkpoint, and output hashes. Register scientific results through the experiment's own declared phase or repository publication procedure; the generic controller does not invent metrics or registry status.

## Budget and billing rules

- Admit worst-case cost before allocation: hourly rate times full provider lifetime plus fixed/storage upper bounds must fit the manifest ceiling.
- Reserve cleanup time inside the provider lifetime. Command timeout totals must fit the remaining compute window.
- Treat an account balance as capacity, not authorization.
- Treat a controller timeout as a cleanup trigger, not permission to start a replacement.
- Charge uncertain lifecycles conservatively until the provider supplies terminal billing evidence.
- Run one paid resource set at a time unless a separate manifest and explicit policy prove disjoint ownership and budget.
- Dependency installation belongs in a prebuilt image or preloaded immutable artifact. A paid canary may verify dependencies; it should not spend most of the lease discovering or building them.

## Failure taxonomy

Classify the first failed boundary accurately:

- **validation/package** — no provider access; repair manifest or source identity;
- **admission** — no allocation; repair budget, authorization, or provider read-only facts;
- **allocation** — reconcile uncertain creation by owner, clean up, then decide whether a new attempt is warranted;
- **hardware/canary** — resource existed but science did not start; fix image, driver facts, or environment;
- **phase failure** — science/evaluation started; preserve newest checkpoint and exact logs;
- **cleanup failure** — do not launch again while absence is unproven;
- **finalization failure** — resources should already be gone; repair local output/provenance only.

Do not call setup work scientific progress. Do not call completed training completed evaluation. Do not call an output durable until finalization binds its bytes.

## Anti-loop rules

- One provider-driver repair generation should consolidate all known response-shape findings. Do not create a new paid run after every field-level patch.
- Keep one short provider canary fixture outside the paid path. Provider API parsing should fail in tests, not after GPU allocation.
- Build dependencies once. Reuse a hash-pinned image/artifact across attempts.
- Never turn a current run id, GPU id, region, model, seed, or checkpoint into reusable policy.
- When two consecutive control-plane repairs produce no canary or experiment progress, perform the implementation-process phase reset: consolidate, execute one decisive canary, or park the provider.
- A skill orchestrates this CLI; it must never become a second implementation of it.

## Completion checklist

- Manifest schema and semantic validation pass.
- Package independently reopens and all bound hashes match.
- Authorization, if paid, matches the exact current bytes and ceilings.
- Observed hardware and canary pass on the allocated lease.
- Every required phase is completed or truthfully failed.
- Latest checkpoint is immutable and restore-verified.
- No duplicate controller or provider resource exists.
- Cleanup proves exact absence.
- Finalization binds every declared output.
- Experiment-specific run/config/seed/environment/checkpoint/metrics/limitations are persisted and registered by the experiment's own contract.
