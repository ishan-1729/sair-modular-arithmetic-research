from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU
from experiments.exp_681_abbie_recurrence.model import AbbieConvGRU, randomize_body, source_identity_receipt


ROOT = Path(__file__).resolve().parents[1]


def models(form: str = "C", *, recurrence: int = 2, untied: bool = False):
    torch.manual_seed(681)
    source = SharedDilatedConvGRU(width=128, hidden=12, cycles=2)
    return source, AbbieConvGRU(source, form=form, default_recurrence=recurrence, untied=untied)


def test_form_c_recurrence_one_is_bit_exact_source() -> None:
    source, abbie = models(recurrence=1)
    assert source_identity_receipt(source, abbie, torch.randn(4, 6, 128)) == {
        "bit_logits_equal": True, "quotient_logits_equal": True, "maximum_absolute_delta": 0.0
    }


def test_head_body_tail_store_only_expected_parameters() -> None:
    source, abbie = models()
    assert set(dict(abbie.named_parameters())) == {
        "head.weight", "head.bias", "bodies.0.gate_weight", "bodies.0.gate_bias",
        "bodies.0.candidate_weight", "bodies.0.candidate_bias", "bodies.0.state_norm.weight",
        "bodies.0.state_norm.bias", "tail_norm.weight", "tail_norm.bias", "tail_local.weight",
        "tail_local.bias", "tail_quotient.0.weight", "tail_quotient.0.bias",
        "tail_quotient.2.weight", "tail_quotient.2.bias",
    }
    assert abbie.accounting(128, 2)["stored_parameters"] == sum(p.numel() for p in source.parameters())


def test_historical_permanent_park_record_is_unchanged() -> None:
    run = json.loads((ROOT / "experiments/runs/exp-681-abbie-recurrence/run.json").read_text())
    metrics = json.loads((ROOT / "evals/results/exp-681-abbie-recurrence/metrics.json").read_text())
    verification = json.loads(
        (ROOT / "experiments/runs/exp-681-abbie-recurrence/verification.json").read_text()
    )
    parked_roots = {
        str(root): "unopened-permanent-verification-park"
        for root in range(20261274, 20261278)
    }
    assert run["status"] == "failed"
    assert metrics["status"] == "permanent-focused-verification-park"
    assert run["decision"] == metrics["decision"] == "permanent-park-focused-topology-schema-failure"
    assert run["roots"] == metrics["roots"] == parked_roots
    assert verification["focused"] == {
        "command": "python -m pytest -q tests/test_exp_681_abbie_recurrence.py",
        "result": "failed",
        "passed": 9,
        "failed": 1,
        "wall_seconds": 4.18,
        "process_cpu_seconds": 9.83,
        "peak_rss_kib": 251992,
        "log_sha256": "97f28a172b5d6bb41687c50852b49cc3c5b59eebdbc2e565ec265e09f614f385",
    }
    assert verification["package_oracle"] == {
        "launched": False,
        "reason": "Focused gate failed; T082 prohibits post-launch repair and requires terminal park.",
    }


def test_recurrence_counts_cover_frozen_test_grid() -> None:
    _, abbie = models()
    features = torch.randn(2, 6, 64)
    schedule = len(abbie.dilation_schedule(64))
    for recurrence in (1, 2, 4, 8):
        trace = abbie(features, recurrence=recurrence, return_trace=True)[2]
        assert trace["body_calls"] == recurrence * schedule
        assert len(trace["state_norms"]) == recurrence


def test_form_d_is_global_residual_not_input_chunk_recurrence() -> None:
    _, form_c = models("C", recurrence=1)
    _, form_d = models("D", recurrence=1)
    features = torch.randn(3, 6, 64)
    c_state = form_c(features, return_trace=True)[2]["final_state"]
    d_state = form_d(features, return_trace=True)[2]["final_state"]
    assert torch.allclose(d_state, c_state + form_c.head(features))


def test_default_training_recurrence_is_two_and_body_has_gradients() -> None:
    _, abbie = models()
    logits, quotient, trace = abbie(torch.randn(3, 6, 64), return_trace=True)
    (logits.square().mean() + quotient.square().mean()).backward()
    assert trace["recurrence"] == 2
    assert sum(float(p.grad.norm()) for p in abbie.body_parameters() if p.grad is not None) > 0


def test_state_and_component_interventions_change_or_zero_outputs() -> None:
    _, abbie = models()
    features = torch.randn(4, 6, 64)
    ordinary = abbie(features)
    assert not torch.equal(ordinary[0], abbie(features, reset_state=True)[0])
    assert not torch.equal(ordinary[0], abbie(features, shuffle_state=True)[0])
    assert not torch.equal(ordinary[0], abbie(features, ablate_head=True)[0])
    assert not torch.equal(ordinary[0], abbie(features, ablate_body=True)[0])
    assert torch.count_nonzero(abbie(features, ablate_tail=True)[0]) == 0


def test_untied_control_has_two_distinct_body_copies() -> None:
    _, tied = models()
    _, untied = models(untied=True)
    assert untied.accounting(128, 2)["stored_bodies"] == 2
    assert untied.accounting(128, 2)["body_parameters"] == 2 * tied.accounting(128, 2)["body_parameters"]
    assert untied.bodies[0].gate_weight.data_ptr() != untied.bodies[1].gate_weight.data_ptr()


def test_randomized_body_changes_weights_without_touching_head_tail() -> None:
    _, abbie = models()
    head = abbie.head.weight.detach().clone(); body = abbie.bodies[0].gate_weight.detach().clone()
    randomize_body(abbie, 681)
    assert torch.equal(head, abbie.head.weight) and not torch.equal(body, abbie.bodies[0].gate_weight)


def test_reload_and_cpu_repeat_are_exact(tmp_path) -> None:
    _, abbie = models(); features = torch.randn(2, 6, 64); expected = abbie.eval()(features)
    path = tmp_path / "abbie.safetensors"
    save_file({key: value.detach().contiguous() for key, value in abbie.state_dict().items()}, path)
    _, restored = models(); restored.load_state_dict(load_file(path)); observed = restored.eval()(features)
    assert torch.equal(expected[0], observed[0]) and torch.equal(expected[1], observed[1])
    assert torch.equal(observed[0], restored(features)[0])


def test_surface_has_no_answers_or_programmed_arithmetic_and_fails_closed() -> None:
    assert set(inspect.signature(AbbieConvGRU.forward).parameters) == {
        "self", "features", "recurrence", "reset_state", "shuffle_state",
        "ablate_head", "ablate_body", "ablate_tail", "return_trace",
    }
    source = inspect.getsource(AbbieConvGRU.forward).lower()
    for forbidden in ("answer", "correctness", "barrett", "montgomery", "crt", "long_division"):
        assert forbidden not in source
    _, abbie = models()
    for bad in (torch.randn(2, 5, 64), torch.full((2, 6, 64), float("nan"))):
        try: abbie(bad)
        except ValueError: pass
        else: raise AssertionError("malformed features must fail")
    try: abbie(torch.randn(2, 6, 64), recurrence=3)
    except ValueError: pass
    else: raise AssertionError("unfrozen recurrence must fail")
