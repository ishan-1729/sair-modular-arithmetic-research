# Candidate 001 reproduction

The retained source trains the exact 85,021-parameter width-9 transition family from random initialization. Run from the repository root and keep generated artifacts in the ignored scratch area:

```powershell
python experiments/runs/candidate-001-horner-tcn9/train.py `
  --output .scratch/reproductions/candidate-001-horner-tcn9

python experiments/runs/candidate-001-horner-tcn9/evaluate_checkpoint.py `
  .scratch/reproductions/candidate-001-horner-tcn9/horner_tcn9.pt `
  --output .scratch/reproductions/candidate-001-horner-tcn9/diagnostics.json
```

The measured reference run used seed `20260712`, 2,400 steps, batch size 768, and four CPU threads. Exact commands for official-contract evaluation are in the [candidate runbook](../../../knowledge/runbooks/evaluate-candidate.md).

`model.py` is the retained training architecture. The separate, inference-only official adapter is staged at [`submission/`](../../../submission/).
