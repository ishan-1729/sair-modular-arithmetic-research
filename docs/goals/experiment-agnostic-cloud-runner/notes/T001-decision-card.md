# T001 decision card

```yaml
hypothesis_or_requirement: "One manifest and public CLI can safely drive arbitrary experiment commands without knowing their scientific semantics."
measured_failure_or_need: "Prior cloud runs duplicated provider/scaffold logic around individual experiments and repeatedly lost time at setup, identity, checkpoint, and cleanup boundaries."
baseline_and_identity: "Branch codex/experiment-agnostic-cloud-runner from cf0dd4cd2f9931a291d06749ddbdbc0a9394da60; no provider or credential access."
changed_variable: "Introduce a provider-neutral lifecycle contract and a thin skill; leave all existing experiment implementations unchanged."
public_route: "validate -> doctor -> package -> canary -> launch/resume -> immutable checkpoint -> status -> finalize -> cleanup"
smallest_falsifier: "A synthetic two-phase experiment interrupted after an immutable checkpoint must resume without repeating the completed phase and must leave no owned resource."
primary_oracle: "Focused end-to-end and adversarial tests through python -m tools.cloud_experiment."
promotion_gate: "Generic lifecycle, skill validation, project validation, full pytest, and final PM audit all pass."
futility_or_stop_rule: "Stop if live provider access is needed, the trained-parameters boundary weakens, or implementation requires experiment-specific code."
retention_and_safety_gates: "No secrets in manifests/receipts, exact source/config/checkpoint hashes, finite budgets, one owner, fail-closed resume identity, idempotent cleanup."
cost_model: "Local-only implementation and tests; no paid compute. Full suite is the only L3 gate."
evidence_outputs: "Focused test receipt, skill validation, runbook, final diff, board receipt."
allowed_files: "tools/cloud_experiment/**; configs/cloud_runner/**; tests/test_cloud_experiment_runner.py; runbook/goal files; personal run-cloud-experiment skill."
```
