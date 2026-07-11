"""Train the small CPU reference model from true transition trajectories."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import POCConfig, load_config
from .data import TrajectorySampler, split_primes, training_moduli
from .model import HornerTCN, ints_to_bits, make_features


def transition_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: POCConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_bits = ints_to_bits(targets, config.width)
    element_loss = F.binary_cross_entropy_with_logits(logits, target_bits, reduction="none")
    mean_bce = element_loss.mean()
    worst_bit_bce = element_loss.max(dim=1).values.mean()
    loss = mean_bce + config.worst_bit_weight * worst_bit_bce
    with torch.no_grad():
        predicted = logits >= 0
        expected = target_bits.bool()
        exact = predicted.eq(expected).all(dim=1).float().mean()
        bit_accuracy = predicted.eq(expected).float().mean()
    return loss, {
        "loss": float(loss.detach()),
        "mean_bce": float(mean_bce.detach()),
        "worst_bit_bce": float(worst_bit_bce.detach()),
        "batch_exact_accuracy": float(exact),
        "batch_bit_accuracy": float(bit_accuracy),
    }


def train(config: POCConfig, run_dir: Path) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")

    train_primes, heldout_primes = split_primes(config)
    train_values = training_moduli(config, train_primes, heldout_primes)
    model = HornerTCN(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.steps,
        eta_min=config.learning_rate * 0.08,
    )
    full_sampler = TrajectorySampler(train_values, config, device, config.seed + 1)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, config.steps + 1):
        curriculum_steps = max(1, int(config.steps * 0.15))
        if step <= curriculum_steps:
            max_bits = min(
                config.width,
                config.min_prime_bits + step * 2 // curriculum_steps,
            )
            eligible = [value for value in train_values if value.bit_length() <= max_bits]
            sampler = TrajectorySampler(
                eligible,
                config,
                device,
                config.seed + 1000 + step,
            )
        else:
            sampler = full_sampler
        state, multiplicand, modulus, bit, target = sampler.sample(config.batch_size)
        logits = model(make_features(state, multiplicand, modulus, bit, config.width))
        loss, statistics = transition_loss(logits, target, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0 or step == config.steps:
            record: dict[str, float | int] = {
                "step": step,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.perf_counter() - started,
                **statistics,
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "horner_tcn.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.as_dict(),
            "train_moduli": train_values,
            "train_primes": train_primes,
            "heldout_primes": heldout_primes,
            "training_seconds": time.perf_counter() - started,
            "history": history,
        },
        checkpoint,
    )
    print(f"saved checkpoint: {checkpoint}")
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = train(load_config(args.config), args.run_dir)
    print("evaluate with:")
    print(
        f"python -m pocs.learned_horner_tcn.evaluate --config {args.config} "
        f"--checkpoint {checkpoint} --output {args.run_dir / 'metrics.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
