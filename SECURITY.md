# Security and disclosure

Do not commit GitHub, Hugging Face, SAIR, or cloud credentials. Store local secrets only in ignored environment files or operating-system credential stores.

The competition sandbox and static checker are defense-in-depth tools, not targets. Do not exploit package, timeout, seed, filesystem, or network discrepancies found during the source audit. Report upstream security issues privately to the competition organizers.

Before any external release:

- inspect every staged file;
- confirm the package allowlist and current sandbox image;
- scan for tokens, private paths, benchmark answers, and unintended metadata;
- publish only the clean-room candidate under `submission/`;
- retain immutable hashes and provenance outside the released artifact.
