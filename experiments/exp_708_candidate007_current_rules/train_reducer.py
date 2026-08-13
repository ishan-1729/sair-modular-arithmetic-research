"""Train the bounded Candidate007 Tier-6 learned streaming reducer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path

import mlflow
import torch
import yaml
from safetensors.torch import load_file, save_file
from torch import nn

from tools.autoresearch_mlflow import configure_mlflow, tracking_uri


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_708_candidate007_current_rules.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp708-current-rules/tier6-reducer",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    cfg = config["training"]
    source = (ROOT / cfg["source_checkpoint"]).resolve()
    if sha256(source) != cfg["source_checkpoint_sha256"]:
        raise RuntimeError("Tier-6 reducer source hash mismatch")

    exp300 = load_module(
        "exp300_for_exp708",
        ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py",
    )
    cellular = load_module(
        "cellular_for_exp708",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    seed = int(cfg["seed"])
    width = int(cfg["width"])
    raw_bits = int(cfg["raw_operand_bits"])
    steps = int(cfg["steps"])
    batch = int(cfg["batch_size"])
    unroll = int(cfg["unroll_steps"])
    split_config = {
        "data": {
            "prime_bits_min": 121,
            "prime_bits_max": 128,
            "train_prime_identities": 512,
            "test_prime_identities": 128,
        }
    }
    split = exp300.bigint_prime_split(split_config, seed + 70800)
    split_hash = hashlib.sha256(
        json.dumps(split, sort_keys=True).encode()
    ).hexdigest()

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = random.Random(seed + 1)
    model = cellular.SharedDilatedConvGRU(width=width, hidden=92, cycles=2)
    model.load_state_dict(load_file(source))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=0.00002,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=float(cfg["minimum_learning_rate"]),
    )

    def soft_features(
        state_plane: torch.Tensor,
        moduli: list[int],
        flags: list[int],
    ) -> torch.Tensor:
        flag_plane = torch.tensor(flags, dtype=torch.float32)[:, None].expand(
            len(flags), width
        )
        lsb = torch.zeros(len(flags), width)
        msb = torch.zeros(len(flags), width)
        lsb[:, 0] = 1
        msb[:, -1] = 1
        return torch.stack(
            (
                state_plane,
                exp300.bits([1] * len(flags), width),
                exp300.bits(moduli, width),
                flag_plane,
                lsb,
                msb,
            ),
            dim=1,
        ).mul(2).sub(1)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    started = time.perf_counter()
    # Retain compatibility with old configs that store tracking as a mapping.
    if isinstance(config.get("tracking"), dict):
        experiment_name = config["tracking"]["experiment_name"]
        experiment_id = configure_mlflow(experiment_name)
    else:
        experiment_name = "sair-modular-arithmetic-wsl"
        experiment_id = configure_mlflow(experiment_name)

    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags(
            {
                "canonical_run_id": config["run_id"],
                "status": "running",
                "role": "candidate007-tier6-learned-reducer",
            }
        )
        mlflow.log_params(
            {
                "seed": seed,
                "steps": steps,
                "batch_size": batch,
                "unroll_steps": unroll,
                "source_checkpoint_sha256": sha256(source),
                "split_sha256": split_hash,
            }
        )
        model.train()
        for step in range(1, steps + 1):
            moduli = [rng.choice(split["train"]) for _ in range(batch)]
            operands = [
                rng.getrandbits(raw_bits - 1) | (1 << (raw_bits - 1))
                for _ in range(batch)
            ]
            starts = [rng.randrange(0, raw_bits - unroll + 1) for _ in range(batch)]
            teacher: list[int] = []
            for operand, modulus, start in zip(operands, moduli, starts, strict=True):
                state = 0
                for position in range(start):
                    flag = (operand >> (raw_bits - 1 - position)) & 1
                    state = (2 * state + flag) % modulus
                teacher.append(state)
            state_plane = exp300.bits(teacher, width)
            losses = []
            exact = []
            for offset in range(unroll):
                positions = [start + offset for start in starts]
                flags = [
                    (operand >> (raw_bits - 1 - position)) & 1
                    for operand, position in zip(operands, positions, strict=True)
                ]
                targets = [
                    (2 * state + flag) % modulus
                    for state, flag, modulus in zip(
                        teacher, flags, moduli, strict=True
                    )
                ]
                quotients = torch.tensor(
                    [
                        (2 * state + flag) // modulus
                        for state, flag, modulus in zip(
                            teacher, flags, moduli, strict=True
                        )
                    ],
                    dtype=torch.long,
                )
                logits, quotient_logits = model(
                    soft_features(state_plane, moduli, flags)
                )
                loss, stats = exp300.transition_loss(
                    logits,
                    quotient_logits,
                    targets,
                    quotients,
                    width,
                    0.45,
                    0.0,
                )
                losses.append(loss)
                exact.append(stats["batch_exact"])
                probabilities = torch.sigmoid(logits)
                state_plane = (
                    logits.ge(0).float()
                    + probabilities
                    - probabilities.detach()
                )
                teacher = targets
            objective = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            gradient = float(nn.utils.clip_grad_norm_(model.parameters(), 2.0))
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 25 == 0 or step == steps:
                record = {
                    "step": step,
                    "loss": float(objective.detach()),
                    "batch_step_exact": sum(exact) / len(exact),
                    "gradient_norm": gradient,
                    "wall_seconds": time.perf_counter() - started,
                }
                history.append(record)
                mlflow.log_metrics(record, step=step)
                print(json.dumps(record), flush=True)

        model.eval()
        checkpoint = output / "weights.safetensors"
        save_file(
            {
                key: value.detach().contiguous()
                for key, value in model.state_dict().items()
            },
            checkpoint,
        )
        summary = {
            "run_id": config["run_id"],
            "status": "completed",
            "component": "tier6-learned-reducer",
            "source_checkpoint_sha256": sha256(source),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "split_sha256": split_hash,
            "history": history,
            "runtime_seconds": time.perf_counter() - started,
            "mlflow": {
                "tracking_uri": tracking_uri(),
                "experiment_name": experiment_name,
                "experiment_id": experiment_id,
                "run_id": run.info.run_id,
            },
            "limitations": [
                "Generated training identities only; benchmark rows are not training data.",
                "Exact bigint operations occur only in offline supervision.",
            ],
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_artifact(str(checkpoint), artifact_path="candidate007-reducer")
        mlflow.log_artifact(str(output / "summary.json"), artifact_path="candidate007-reducer")
        mlflow.set_tags({"status": "completed"})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
