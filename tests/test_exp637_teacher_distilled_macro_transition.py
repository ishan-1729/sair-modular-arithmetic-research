from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_macro_transition_is_two_radix2_steps() -> None:
    data = load_module(
        "exp637_data_exact_test",
        ROOT / "experiments/exp_637_teacher_distilled_macro_transition/data.py",
    )
    for modulus in (17, 29):
        for state in range(modulus):
            for right in range(modulus):
                for digit in range(4):
                    intermediate, q1, final, q2 = data.exact_macro_transition(
                        state, right, modulus, digit
                    )
                    first = 2 * state + (digit >> 1) * right
                    second = 2 * intermediate + (digit & 1) * right
                    assert (intermediate, q1) == (first % modulus, first // modulus)
                    assert (final, q2) == (second % modulus, second // modulus)
                    assert final == (4 * state + digit * right) % modulus


def test_macro_features_route_high_then_low_digit_bits() -> None:
    data = load_module(
        "exp637_data_feature_test",
        ROOT / "experiments/exp_637_teacher_distilled_macro_transition/data.py",
    )
    first, second = data.macro_features(
        [0, 0, 0, 0], [1] * 4, [17] * 4, [0, 1, 2, 3], 5
    )
    assert first.shape == second.shape == (4, 6, 5)
    assert first[:, 3, 0].tolist() == [-1.0, -1.0, 1.0, 1.0]
    assert second[:, 3, 0].tolist() == [-1.0, 1.0, -1.0, 1.0]
    assert second[:, 0].eq(-1.0).all()


def test_macro_rollout_matches_product_mod_prime() -> None:
    data = load_module(
        "exp637_data_rollout_test",
        ROOT / "experiments/exp_637_teacher_distilled_macro_transition/data.py",
    )
    for left in range(16):
        for right in range(16):
            assert data.teacher_rollout(left, right, 17, 4) == (left * right) % 17


def test_teacher_transfer_duplicates_parameters_but_not_storage() -> None:
    teacher_module = load_module(
        "exp637_teacher_model_test",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    student_module = load_module(
        "exp637_student_model_transfer_test",
        ROOT / "experiments/exp_637_teacher_distilled_macro_transition/model.py",
    )
    torch.manual_seed(9)
    teacher = teacher_module.SharedDilatedConvGRU(width=8, hidden=8, cycles=2)
    student = student_module.TeacherDistilledMacroTransition(hidden=8)
    report = student_module.transfer_teacher_state(student, teacher.state_dict())
    for key, value in teacher.state_dict().items():
        assert torch.equal(student.phase1.state_dict()[key], value)
        assert torch.equal(student.phase2.state_dict()[key], value)
    assert student.phase1.gate_weight.data_ptr() != student.phase2.gate_weight.data_ptr()
    assert student.parameter_count() == 2 * teacher.parameter_count()
    assert report["parameters_untied_after_initialization"] is True


def test_final_loss_backpropagates_through_continuous_intermediate() -> None:
    student_module = load_module(
        "exp637_student_model_gradient_test",
        ROOT / "experiments/exp_637_teacher_distilled_macro_transition/model.py",
    )
    torch.manual_seed(17)
    student = student_module.TeacherDistilledMacroTransition(hidden=8)
    first = torch.randn(3, 6, 8)
    second = torch.randn(3, 6, 8)
    final_logits, intermediate_logits, q1_logits, q2_logits = student(first, second)
    assert final_logits.shape == intermediate_logits.shape == (3, 8)
    assert q1_logits.shape == q2_logits.shape == (3, 3)
    final_logits.square().mean().backward()
    assert student.phase1.local_head.weight.grad is not None
    assert float(student.phase1.local_head.weight.grad.abs().sum()) > 0.0


def test_masked_distillation_ignores_incorrect_teacher_cases() -> None:
    runner = load_module(
        "exp637_distillation_test",
        ROOT / "experiments/exp_637_teacher_distilled_macro_transition/run.py",
    )
    student = torch.tensor([[0.2, -0.3], [2.0, -2.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    mask = torch.tensor([True, False])
    loss = runner.masked_binary_distillation(student, teacher, mask, 2.0)
    loss.backward()
    assert student.grad is not None
    assert float(student.grad[0].abs().sum()) > 0.0
    assert torch.equal(student.grad[1], torch.zeros_like(student.grad[1]))
