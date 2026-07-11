---
type: Release Runbook
title: Release a Hugging Face Submission Candidate
description: Clean-room checklist for staging, sandboxing, hashing, publishing, and locking the ranked artifact.
status: draft
tags: [runbook, release, hugging-face, submission]
timestamp: "2026-07-11T23:15:00+05:30"
---

# Release a submission candidate

1. Freeze the source commit, evaluator commit, Docker image digest, configuration, and checkpoint.
2. Create a fresh directory under `submission/candidates/<candidate-id>/`.
3. Copy only `manifest.json`, inference `model.py`, weights, and required inference assets.
4. Confirm `model_description` and `training_description` match the immutable run record.
5. Scan every submitted Python file with the official static checker.
6. Run manifest, load, preprocess-isolation, determinism, and output-contract checks.
7. Run the full public benchmark through the official sandbox on CPU and record per-tier timing.
8. Confirm total artifact bytes and at least 25% timing headroom where possible.
9. Reinitialize or perturb weights and record the expected capability collapse outside the candidate directory.
10. Audit the candidate for secrets, benchmark answers, training utilities, caches, and undeclared files.
11. Create or update the Hugging Face repository privately, verify by immutable commit, then make it public only when ready to submit.
12. Re-run remote accessibility and immutable-hash evaluation, submit `repo_id` plus commit hash, and retain the receipt.

Publication changes external state. The final repository visibility change and submission action require the project owner's explicit execution decision at release time.
