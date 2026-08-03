#!/usr/bin/env python3
"""Materialize the ignored research fixtures required by the full test suite.

The repository intentionally does not version ``.scratch``.  A clean worktree can
therefore run the complete regression suite only when an operator supplies the
existing, authenticated research evidence.  Integrity-sensitive roots are
independent byte copies.  Large roots whose tests do not authenticate local
path ownership are explicit read-only directory links.  The tool never creates
hard-link aliases and never overwrites a divergent file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable


# This is the closed fixture surface exercised by the repository test suite at
# the cuda-prepared-hardware-unvalidated gate.  New fixture-dependent tests must
# extend this list explicitly instead of silently importing all of .scratch.
REQUIRED_ROOTS = (
    "exp-100-width16-shared-context-smoke",
    "exp-105-width-transfer-smoke",
    "exp-106-width-transfer-1200",
    "exp-110-quotient2-hard-mining",
    "exp-111-hard-mining-2400",
    "exp-200-width32-transfer-ab",
    "exp-201-width32-full-budget",
    "exp-202-width32-2400",
    "exp-204-official-distribution-finetune",
    "exp-210-rollout-aware",
    "exp-220-width-independent-cellular",
    "exp-230-quotient-routed-experts",
    "exp-250-online-failure-mining",
    "exp-251-online-failure-mining-smoke",
    "exp-252-replay-heavy-failure-mining",
    "exp-260-low-rate-continuation",
    "exp-300-width64-bit-tensor-transfer",
    "exp-301-width64-hard-mining",
    "exp-310-chunk-shared-recurrent",
    "exp-312-cellular-expanded-width-retry",
    "exp-314-cellular-width32-smoke",
    "exp-320-cellular-width64-curriculum",
    "exp-322-cellular-width64-smoke",
    "exp-323-cellular-width64-diverse",
    "exp-324-cellular-width64-second-stage",
    "exp-326-cellular-width64-soup-retry",
    "exp-327-cellular-width64-logit-ensemble",
    "exp-328-cellular-width64-logit-ensemble-retry",
    "exp-329-cellular-width64-autonomous-state-recovery",
    "exp-400-graph-message-passing-smoke",
    "exp-401-pyg-graph-message-passing",
    "exp-500-cellular-grokking-probe",
    "exp-510-multiarchitecture-grokking-feasibility",
    "exp-520-tier5-recovery-delta-patching",
    "exp-600-tier6-cellular-curriculum",
    "exp-601-tier6-autonomous-state-recovery",
    "exp-602-tier6-recovery-full-budget",
    "exp-603-tier6-recovery-continuation",
    "exp-604-tier6-long-horizon-recovery",
    "exp-605-tier6-long-horizon-retention",
    "exp-606-tier6-checkpoint-complementarity",
    "exp-607-tier6-differentiable-soft-state",
    "exp-608-tier6-straight-through-state",
    "exp-609-tier6-straight-through-confirmation",
    "exp-610-tier6-state-corruption-robustness",
    "exp-611-tier6-function-preserving-widening",
    "exp-612-tier6-adapter-activation",
    "exp-613-tier6-model-conditioned-error-corrector",
    "exp-614-tier6-global-recurrent-refinement",
    "exp-615-tier6-casewise-exactness-loss",
    "exp-616-tier6-trajectory-tail-cvar",
    "exp-617-tier6-recurrent-core-lora",
    "exp-618-tier6-continuous-state-inference",
    "exp-619-tier6-confidence-augmented-state",
    "exp-620-persistent-latent-neural-gpu",
    "exp-621-tier6-learned-beam-decoding",
    "exp-622-tier6-width4-beam-coverage",
    "exp-623-tier6-staged-beam-runtime",
    "exp-624-tier6-width3-beam-coverage",
    "exp-625-tier6-confidence-routed-width4-search",
    "exp-626-tier6-minimum-margin-routing-confirmation",
    "exp-627-procedural-smoke",
    "exp-627-procedural-smoke-resume",
    "exp-627-tier6-trajectory-feature-reranker",
    "exp-629-tier6-balanced-first-error-replay",
    "exp-630-radix4-cellular-transition",
    "exp-632-tier6-shared-core-diverse-output-heads",
    "exp-633-hierarchical-learned-composition-feasibility",
    "exp-634-direct-relative-attention-feasibility",
    "exp-635-independent-cellular-diversity-feasibility",
    "exp-636-independent-cellular-tier5-bootstrap",
    "exp-637-teacher-distilled-macro-transition",
    "exp-638-full-depth-soft-boundary-diagnostic",
    "exp-639-full-depth-soft-boundary-adaptation",
    "exp-640-commutative-operand-swap-diversity",
    "exp-641-conditional-commutative-swap-routing",
    "exp-642-fresh-conditional-commutative-confirmation",
    "exp-643-tier7-recurrent-depth-stability-diagnostic",
    "exp-647-operand-swap-consistency",
    "exp-648-structured-trajectory-consistency",
    "exp-649-stochastic-trajectory-breadth",
    "exp-650-library-backed-mamba",
    "exp-651-visit-dilation-adapters",
    "exp-654-neural-ram-executor-viability",
    "exp-658-joint-quotient-carry-transducer",
    "exp-660-symbol-equivariant-formal",
    "exp-663-drm-target-token-denoising",
    "exp-665-attractor-equilibrium-family",
    "exp-675-rrt-svd-teacher-uptraining",
    "exp-677-length-schedule",
    "exp-685-validity-constrained-attractor",
    "mlflow",
)

MUTABLE_OVERLAY_EXCLUSIONS = {Path("mlflow/mlflow.db")}
PHYSICAL_ROOTS = {
    "exp-658-joint-quotient-carry-transducer",
    "exp-663-drm-target-token-denoising",
    "exp-675-rrt-svd-teacher-uptraining",
    "exp-685-validity-constrained-attractor",
    "mlflow",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_paths(repository: Path, sources: Iterable[Path]) -> tuple[Path, list[Path]]:
    repository = repository.resolve()
    if not (repository / ".git").exists():
        raise ValueError(f"repository is not a Git worktree: {repository}")
    target = (repository / ".scratch").resolve(strict=False)
    resolved_sources: list[Path] = []
    for source in sources:
        resolved = source.resolve(strict=True)
        if not resolved.is_dir() or resolved.name != ".scratch":
            raise ValueError(f"fixture source must be an existing .scratch directory: {resolved}")
        if resolved == target or _inside(resolved, repository):
            raise ValueError(f"fixture source must be outside the target worktree: {resolved}")
        resolved_sources.append(resolved)
    if not resolved_sources:
        raise ValueError("at least one fixture source is required")
    return target, resolved_sources


def _materialize_tree(
    source_root: Path,
    target_root: Path,
    relative_root: Path,
    counters: dict[str, int],
) -> None:
    source = source_root / relative_root
    if not source.exists():
        return
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"fixture root must be a physical directory: {source}")
    for current, directory_names, file_names in os.walk(source, followlinks=False):
        current_path = Path(current)
        for name in tuple(directory_names):
            candidate = current_path / name
            if candidate.is_symlink():
                raise ValueError(f"fixture source contains a directory symlink: {candidate}")
        destination_dir = target_root / relative_root / current_path.relative_to(source)
        destination_dir.mkdir(parents=True, exist_ok=True)
        for name in file_names:
            candidate = current_path / name
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"fixture source is not a physical file: {candidate}")
            relative = relative_root / candidate.relative_to(source)
            destination = target_root / relative
            if destination.exists() or destination.is_symlink():
                if destination.is_symlink() or not destination.is_file():
                    raise ValueError(f"fixture destination is not a physical file: {destination}")
                if relative in MUTABLE_OVERLAY_EXCLUSIONS:
                    counters["mutable_existing"] += 1
                    continue
                if sha256(candidate) != sha256(destination):
                    raise ValueError(f"refusing to overwrite divergent fixture: {destination}")
                counters["verified_existing"] += 1
                continue
            shutil.copy2(candidate, destination)
            counters["copied"] += 1


def restore(
    repository: Path,
    sources: Iterable[Path],
    *,
    replace_root_symlinks: bool = False,
) -> dict[str, object]:
    target, resolved_sources = validate_paths(repository, sources)
    target.mkdir(parents=True, exist_ok=True)
    counters = {
        "copied": 0,
        "verified_existing": 0,
        "mutable_existing": 0,
        "removed_root_symlinks": 0,
        "linked_roots": 0,
        "verified_root_links": 0,
    }
    restored_roots: list[str] = []
    for name in REQUIRED_ROOTS:
        relative_root = Path(name)
        destination = target / relative_root
        available = [
            source
            for source in resolved_sources
            if (source / relative_root).is_dir()
            and not (source / relative_root).is_symlink()
        ]
        if not available:
            raise FileNotFoundError(f"required fixture root is absent from all sources: {name}")
        if name not in PHYSICAL_ROOTS:
            expected = available[0] / relative_root
            if destination.is_symlink():
                if destination.resolve(strict=True) == expected.resolve(strict=True):
                    counters["verified_root_links"] += 1
                    restored_roots.append(name)
                    continue
                if not replace_root_symlinks:
                    raise ValueError(
                        f"fixture destination root has the wrong target; rerun with --replace-root-symlinks: {destination}"
                    )
                destination.unlink()
                counters["removed_root_symlinks"] += 1
            elif destination.exists():
                raise ValueError(f"linkable fixture destination already exists physically: {destination}")
            destination.symlink_to(expected, target_is_directory=True)
            counters["linked_roots"] += 1
            restored_roots.append(name)
            continue
        if destination.is_symlink():
            if not replace_root_symlinks:
                raise ValueError(
                    f"fixture destination root is a symlink; rerun with --replace-root-symlinks: {destination}"
                )
            destination.unlink()
            counters["removed_root_symlinks"] += 1
        for source in available:
            _materialize_tree(source, target, relative_root, counters)
        restored_roots.append(name)

    payload: dict[str, object] = {
        "schema_version": "external-test-fixture-restore-v1",
        "repository": str(repository.resolve()),
        "sources": [str(path) for path in resolved_sources],
        "roots": restored_roots,
        **counters,
    }
    receipt = target / "external-test-fixture-restore.json"
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        required=True,
        help="external .scratch directory; repeat for physical historical overlays",
    )
    parser.add_argument("--replace-root-symlinks", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = restore(
        args.repository,
        args.source,
        replace_root_symlinks=args.replace_root_symlinks,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
