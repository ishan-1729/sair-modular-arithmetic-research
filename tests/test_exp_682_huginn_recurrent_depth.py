from __future__ import annotations

import inspect

import torch
from safetensors.torch import load_file, save_file

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU
from experiments.exp_682_huginn_recurrent_depth.model import (
    HuginnTopologyProbe,
    sample_bounded_lognormal_poisson_depths,
    structural_transfer_receipt,
)


def models() -> tuple[SharedDilatedConvGRU, HuginnTopologyProbe]:
    torch.manual_seed(682)
    source = SharedDilatedConvGRU(width=64, hidden=12, cycles=2)
    return source, HuginnTopologyProbe(source, release_seed=68205171)


def test_source_boundaries_and_required_native_topology_are_explicit() -> None:
    source, probe = models()
    assert torch.equal(source.stem.weight, probe.prelude.weight)
    assert torch.equal(source.gate_weight, probe.gate_weight)
    assert torch.equal(source.local_head.weight, probe.coda_local.weight)
    assert isinstance(probe.pre_norm, torch.nn.RMSNorm)
    assert isinstance(probe.post_norm, torch.nn.RMSNorm)
    assert probe.concat_adapter.in_channels == 2 * source.hidden
    assert probe.concat_adapter.out_channels == source.hidden


def test_eq1_eq2_distribution_is_replayable_bounded_and_heavy_tailed() -> None:
    first = sample_bounded_lognormal_poisson_depths(400, seed=20261278)
    second = sample_bounded_lognormal_poisson_depths(400, seed=20261278)
    assert torch.equal(first, second)
    assert int(first.min()) >= 2 and int(first.max()) <= 12
    assert int(first.max()) > int(torch.median(first))
    assert torch.unique(first).numel() >= 5


def test_random_zero_and_deterministic_latent_release_semantics() -> None:
    _, probe = models()
    features = torch.randn(3, 6, 32)
    zero = probe.latent(features, "zero")
    deterministic_a = probe.latent(features, "deterministic")
    deterministic_b = probe.latent(features, "deterministic")
    random_a = probe.latent(features, "random", seed=1)
    random_b = probe.latent(features, "random", seed=2)
    assert torch.count_nonzero(zero) == 0
    assert torch.equal(deterministic_a, deterministic_b)
    assert not torch.equal(random_a, random_b)


def test_tbptt_depth_and_core_call_accounting() -> None:
    _, probe = models()
    features = torch.randn(2, 6, 32, requires_grad=True)
    logits, quotient, trace = probe(features, depth=12, tbptt_last=8, return_trace=True)
    (logits.square().mean() + quotient.square().mean()).backward()
    assert trace.prefix_without_credit == 4
    assert trace.suffix_with_credit == 8
    assert trace.core_calls == 12 * len(probe.dilation_schedule(32))
    assert probe.prelude.weight.grad is not None
    assert probe.gate_weight.grad is not None


def test_full_bptt_and_truncated_routes_match_forward_values() -> None:
    _, probe = models()
    features = torch.randn(2, 6, 16)
    truncated = probe(features, depth=4, tbptt_last=2)
    full = probe(features, depth=4, full_bptt=True)
    assert torch.equal(truncated[0], full[0])
    assert torch.equal(truncated[1], full[1])


def test_component_and_state_interventions_are_causal() -> None:
    _, probe = models()
    features = torch.randn(3, 6, 32)
    ordinary = probe(features, depth=2, latent_mode="random", latent_seed=7)
    variants = [
        probe(features, depth=2, latent_mode="random", latent_seed=8),
        probe(features, depth=2, latent_mode="random", latent_seed=7, reinjection=False),
        probe(features, depth=2, latent_mode="random", latent_seed=7, concat=False),
        probe(features, depth=2, latent_mode="random", latent_seed=7, rmsnorm=False),
        probe(features, depth=2, latent_mode="random", latent_seed=7, reset=True),
        probe(features, depth=2, latent_mode="random", latent_seed=7, shuffled=True),
    ]
    assert all(not torch.equal(ordinary[0], variant[0]) for variant in variants)


def test_required_causal_topology_is_not_exact_checkpoint_transfer() -> None:
    source, probe = models()
    receipt = structural_transfer_receipt(source, probe, torch.randn(4, 6, 32))
    assert receipt["source_norm_type"] == "GroupNorm"
    assert receipt["required_norm_type"] == "RMSNorm"
    assert receipt["source_has_concat_adapter"] is False
    assert receipt["source_has_random_latent"] is False
    assert receipt["causal_required_components"] is True
    assert receipt["exact_checkpoint_transfer"] is False
    assert receipt["source_vs_required_max_delta"] > 0


def test_reload_and_deterministic_release_are_exact(tmp_path) -> None:
    source, probe = models()
    features = torch.randn(2, 6, 32)
    expected = probe.eval()(features)
    path = tmp_path / "probe.safetensors"
    save_file({name: value.detach().contiguous() for name, value in probe.state_dict().items()}, path)
    restored = HuginnTopologyProbe(source)
    restored.load_state_dict(load_file(path), strict=True)
    observed = restored.eval()(features)
    assert torch.equal(expected[0], observed[0])
    assert torch.equal(expected[1], observed[1])
    assert torch.equal(observed[0], restored(features)[0])


def test_surface_has_no_answers_or_programmed_arithmetic_and_fails_closed() -> None:
    source = inspect.getsource(HuginnTopologyProbe.forward).lower()
    for forbidden in ("answer", "correctness", "barrett", "montgomery", "crt", "long_division"):
        assert forbidden not in source
    _, probe = models()
    for bad in (torch.randn(2, 5, 32), torch.full((2, 6, 32), float("nan"))):
        try:
            probe(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed features must fail closed")
    for depth in (1, 13):
        try:
            probe(torch.randn(2, 6, 32), depth=depth)
        except ValueError:
            pass
        else:
            raise AssertionError("unfrozen depth must fail closed")
