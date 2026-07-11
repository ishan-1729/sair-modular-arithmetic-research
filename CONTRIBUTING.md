# Contributing

This is a private competition research workspace. Keep changes small, reproducible, and easy to audit.

1. Read [AGENTS.md](AGENTS.md) and the [competition compliance guide](knowledge/competition/compliance.md).
2. Create or update a machine-readable run record for any changed result or checkpoint.
3. Keep official benchmark data evaluation-only and split synthetic data by prime identity.
4. Put durable research interpretation in the `knowledge/` OKF bundle.
5. Run:

   ```powershell
   python tools/validate_project.py
   python -m pytest -m "not slow"
   ```

6. Never place credentials, benchmark answers, trainers, or exploratory files in `submission/`.

Negative results are first-class evidence. Record the hypothesis, exact configuration, observed failure, and what decision it changes.
