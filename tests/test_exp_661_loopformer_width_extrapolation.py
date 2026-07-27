from __future__ import annotations

import ast
import inspect
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

from experiments.exp_661_loopformer_width_extrapolation.model import (
    ConditionedLoopBlock,
    DecodedCellularControl,
    LoopFormer,
    PlainLoopedTransformer,
    sinusoidal_scalar,
)
from experiments.exp_661_loopformer_width_extrapolation.run import (
    ARM_NAMES,
    calibration_admission,
    endpoint_targets,
    feature_tensor,
    loss_weights,
    scientific_gate,
    short_schedule,
)

ROOT = Path(__file__).resolve().parents[1]


def small_features(batch: int = 3, width: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(661)
    return torch.randint(0, 2, (batch, 5, width), generator=generator).float().mul(2).sub(1)


def small_model(**kwargs) -> LoopFormer:
    return LoopFormer(hidden=12, heads=3, ff_hidden=24, condition=8, max_width=32, default_budget=8, **kwargs)


def test_primary_uses_shared_loop_block_normalized_time_and_step() -> None:
    model = small_model()
    assert isinstance(model.block, ConditionedLoopBlock)
    encoded = sinusoidal_scalar(torch.tensor([0.0, 0.5]), 8)
    assert encoded.shape == (2, 8)
    features = small_features()
    output = model.loop(features, schedule=[0.25, 0.25, 0.5])
    assert output.hidden.shape == (3, 8, 12)
    assert output.logits.shape == (3, 8)
    assert output.call_count == 3
    assert len(output.hidden_curve) == len(output.logit_curve) == 3
    first = model.trajectory_condition(0.0, 0.25, 3, features.device)
    second = model.trajectory_condition(0.25, 0.25, 3, features.device)
    assert not torch.equal(first, second)


def test_rmsnorm_conditioning_and_shortcut_switches_are_live() -> None:
    torch.manual_seed(2)
    model = small_model().eval()
    assert isinstance(model.block.norm_attention, torch.nn.RMSNorm)
    assert isinstance(model.block.norm_ffn, torch.nn.RMSNorm)
    features = small_features(batch=2)
    with torch.inference_mode():
        baseline = model.loop(features, schedule=[0.5, 0.5]).logits
        no_time = model.loop(features, schedule=[0.5, 0.5], use_time_step=False).logits
        no_shortcut = model.loop(features, schedule=[0.5, 0.5], use_shortcut=False).logits
    assert not torch.equal(baseline, no_time)
    assert not torch.equal(baseline, no_shortcut)


def test_full_short_endpoint_and_stop_gradient_hidden_consistency() -> None:
    model = small_model()
    features = small_features(batch=2)
    targets = torch.zeros(2, 8)
    full = model.loop(features, schedule=model.uniform_schedule(8))
    short = model.loop(features, schedule=[0.5, 0.5])
    full.hidden.retain_grad()
    short.hidden.retain_grad()
    loss = F.binary_cross_entropy_with_logits(full.logits, targets)
    loss += 0.1 * F.binary_cross_entropy_with_logits(short.logits, targets)
    consistency = F.mse_loss(short.hidden, full.hidden.detach())
    consistency.backward(retain_graph=True)
    assert short.hidden.grad is not None
    assert full.hidden.grad is None
    assert loss_weights("loopformer", yaml.safe_load((ROOT / "configs/exp_661_loopformer_width_extrapolation.yaml").read_text())) == (0.1, 0.1)


def test_discrete_shortcuts_and_fixed_user_budgets() -> None:
    for index in range(14):
        schedule = short_schedule(index, 20261195, 8)
        assert 1 <= len(schedule) < 8
        assert all(step > 0 for step in schedule)
        assert abs(sum(schedule) - 1.0) < 1e-9
    model = small_model().eval()
    features = small_features(batch=2)
    with torch.inference_mode():
        for budget in (1, 2, 4, 8, 16):
            assert model(features, budget=budget).shape == (2, 8)
    assert not any("halt" in name.lower() or "exit" in name.lower() for name, _ in model.named_modules())


def test_controls_match_parameters_and_recurrent_calls() -> None:
    models = [small_model(), PlainLoopedTransformer(hidden=12, heads=3, ff_hidden=24, condition=8, max_width=32, default_budget=8), DecodedCellularControl(hidden=12, heads=3, ff_hidden=24, condition=8, max_width=32, default_budget=8)]
    assert len({model.parameter_count() for model in models}) == 1
    calls = []
    for model in models:
        count = 0

        def increment(_module, _inputs):
            nonlocal count
            count += 1

        hook = model.block.register_forward_pre_hook(increment)
        output = model.loop(small_features(batch=2), schedule=[0.25] * 4)
        hook.remove()
        calls.append((count, output.call_count))
    assert calls == [(4, 4)] * 3


def test_legal_inputs_labels_and_checkpoint_reload(tmp_path: Path) -> None:
    cases = [{"left": 11, "right": 7, "modulus": 13}, {"left": 5, "right": 9, "modulus": 11}]
    features, targets = feature_tensor(cases, 4), endpoint_targets(cases, 4)
    assert features.shape == (2, 5, 4)
    assert set(features.unique().tolist()) <= {-1.0, 1.0}
    assert targets.shape == (2, 4)
    model = small_model().eval()
    path = tmp_path / "model.safetensors"
    save_file({key: value.contiguous() for key, value in model.state_dict().items()}, path)
    loaded = small_model().eval()
    loaded.load_state_dict(load_file(path), strict=True)
    with torch.inference_mode():
        assert torch.equal(model(features, budget=4), loaded(features, budget=4))


def test_forward_paths_contain_no_prohibited_exact_arithmetic() -> None:
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            ConditionedLoopBlock.forward,
            LoopFormer.trajectory_condition,
            LoopFormer.loop,
            LoopFormer.forward,
        )
    )
    tree = ast.parse(inspect.cleandoc(source))
    forbidden = {ast.Mod, ast.FloorDiv, ast.LShift, ast.RShift, ast.BitAnd, ast.BitXor}
    observed = {type(node.op) for node in ast.walk(tree) if isinstance(node, (ast.BinOp, ast.AugAssign))}
    assert forbidden.isdisjoint(observed)
    assert "puzzle" not in source.lower()
    assert "evaluation_id" not in source.lower()


def test_dynamic_calibration_profiles_complete_route_without_static_cap() -> None:
    config = yaml.safe_load((ROOT / "configs/exp_661_loopformer_width_extrapolation.yaml").read_text())
    arms = {
        name: {
            "resources": {
                "training_wall_seconds": 2.0,
                "training_process_cpu_seconds": 4.0,
                "training_cgroup_cpu_seconds": 3.0,
                "update_wall_seconds": [0.9, 1.1],
                "peak_rss_bytes": 1024,
            },
            "reload_exact": True,
        }
        for name in ARM_NAMES
    }
    admission = calibration_admission(arms, 20.0, 40.0, 35.0, config)
    assert admission["admitted"]
    assert admission["checks"]["all_route_classes"]
    route = admission["projected_formal_route"]
    assert route["wall_seconds"]["hard"] > route["wall_seconds"]["soft"] > route["wall_seconds"]["eta"]
    assert admission["arbitrary_static_wall_cap"] is None


def test_frozen_roots_components_and_gates_are_complete() -> None:
    config = yaml.safe_load((ROOT / "configs/exp_661_loopformer_width_extrapolation.yaml").read_text())
    assert config["roots"] == {"calibration": 20261194, "viability": 20261195, "confirmations": [20261196, 20261197]}
    assert config["evaluation"]["fixed_budgets"] == [1, 2, 4, 8, 16]
    assert set(config["calibration"]["route_classes"]) == set(ARM_NAMES)
    assert set(config["confirmation"]["component_controls"]) == {"no_time_step", "no_shortcut", "no_short_loss", "no_consistency"}
    assert config["gates"]["minimum_primary_w16_exact_accuracy"] >= 0.80


def test_scientific_gate_is_conjunctive_across_width_depth_controls_and_components() -> None:
    config = yaml.safe_load((ROOT / "configs/exp_661_loopformer_width_extrapolation.yaml").read_text())

    def arm(w16: float, w24: float, w32: float):
        evaluation = {}
        for width, value in (("16", w16), ("24", w24), ("32", w32)):
            evaluation[width] = {str(budget): {"exact_accuracy": value, "invalid_rate": 0.0} for budget in (1, 2, 4, 8, 16)}
        return {"parameters": 100, "reload_exact": True, "training": {"optimizer_updates": 10, "trajectory_examples": 320, "recurrent_block_calls": 1000}, "evaluation": evaluation}

    arms = {
        "loopformer": arm(0.90, 0.20, 0.20),
        "plain_looped_transformer": arm(0.60, 0.10, 0.10),
        "decoded_cellular_control": arm(0.60, 0.10, 0.10),
        "no_time_step": arm(0.60, 0.10, 0.10),
        "no_shortcut": arm(0.60, 0.10, 0.10),
        "no_short_loss": arm(0.60, 0.10, 0.10),
        "no_consistency": arm(0.60, 0.10, 0.10),
    }
    diagnostics = {"randomized_weights": {"exact_accuracy": 0.0}, "inference_interventions": {"no_time_step": {"exact_accuracy": 0.60}, "no_shortcut": {"exact_accuracy": 0.60}}}
    assert scientific_gate(arms, diagnostics, config)["passed"]
    arms["loopformer"]["evaluation"]["32"]["8"]["exact_accuracy"] = 0.0
    assert not scientific_gate(arms, diagnostics, config)["passed"]
