# External GPU provider controller

This package controls rented resources from outside the guest. It supports a
side-effect-free `plan` mode and an explicitly gated `run` mode for RunPod Pods
and Vast.ai instances. Tests use an injected fake transport; repository
verification never contacts either provider.

The endpoint shapes follow the current official lifecycle documentation:

- RunPod creates, observes, and deletes Pods with Bearer-authenticated REST
  calls. A stopped Pod can retain billable volume storage, so the controller
  requires DELETE confirmation rather than accepting stop/exit as cost closure:
  https://docs.runpod.io/pods/manage-pods
- Vast creates an instance by accepting an offer, observes its contract, and
  destroys it with DELETE. Vast explicitly documents that stopped instances
  continue storage charges:
  https://docs.vast.ai/api-reference/instances/create-instance
  and https://docs.vast.ai/guides/reference/faq/billing

Always plan with the current offer's compute and storage rates:

```bash
python -m environments.providers.controller \
  --config configs/cuda_provider_control.yaml plan \
  --provider runpod \
  --resource-selector 'NVIDIA GeForce RTX 4090' \
  --image '<verified-image>' \
  --hourly-rate-usd 0.69 \
  --storage-hourly-usd 0.006945 \
  --max-total-usd 1.00 \
  --max-provider-seconds 1800 \
  --port '8000/http' \
  --command 'git clone ... && environments/cuda/bootstrap.sh --profile cu124 && environments/cuda/run-acceptance.sh --profile cu124'
```

For a large private-source bootstrap payload, pass `--command-file PATH`
instead of `--command`. The controller reads the file locally, records only
the workload hash, and sends the command to the provider without placing it in
the host process argument list. `--port PORT/http` may be repeated for bounded
receipt retrieval; no port is exposed by default.

Use `--environment-file PREFIX=PATH` for a source archive or other large input.
It base64-encodes the file into numbered 60,000-character environment chunks,
keeping each value below Linux's per-string process limit. Receipts contain
only chunk names, sizes, and hashes; the values themselves remain redacted.
`--environment-value-file NAME=PATH` reads one small non-secret value, such as
an ephemeral SSH public key, while likewise recording only its size and hash.
For RunPod payloads that exceed its direct Pod-create gateway, place the same
private configuration in a temporary template and pass `--template-id ID`.
The Pod request then contains only scheduling fields; the caller remains
responsible for deleting the temporary template after the Pod is destroyed.
For official images that initialize SSH/Jupyter themselves, pass
`--use-image-command`; this submits empty ENTRYPOINT/CMD overrides so RunPod
preserves the image defaults while retaining the controller's lifecycle cap.
Set `--cuda-runtime any` only when the image plus the guest's fail-closed
environment verifier already enforce the required driver/runtime compatibility.

Plan mode does not read a credential. It reserves 15% of the total cap and
charges the remaining upper bound from the instant create is requested until
DELETE is confirmed, including provisioning and storage. It prints only the
workload-command hash.

Live mode additionally requires the exact `SAIR_CLOUD_LAUNCH_APPROVED` value
recorded in the config and the provider token in `RUNPOD_API_KEY` or
`VAST_API_KEY`. Those values remain in the external Authorization header and
are rejected if found in the guest command/payload. The resource id is durably
written immediately after creation, and SIGINT, SIGTERM, guest exit, rate
change, timeout, and ordinary exceptions all enter a retrying DELETE path.

Client-side controls cannot survive every laptop power/network/process failure.
Before any authorized launch, use prepaid credit, disable automatic recharge,
and set provider account alerts/limits. The receipt's billed-lifetime field is
a controller-observed create-to-delete upper bound, not a provider invoice; the
actual invoice must be attached to the later real-GPU receipt.
