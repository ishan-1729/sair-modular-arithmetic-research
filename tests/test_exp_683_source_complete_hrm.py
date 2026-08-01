from __future__ import annotations
import inspect
import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file
from experiments.exp_683_source_complete_hrm.model import ROUTES,SourceCompleteHRM

def model():
    torch.manual_seed(683); return SourceCompleteHRM(hidden=16,heads=2,feedforward=24,high_cycles=2,low_steps=2,maximum_segments=4)
def sample(batch=3,width=8): return torch.randn(batch,6,width)

def test_source_topology_has_distinct_carried_h_and_l():
    m=model(); out=m(sample(),route="full_hrm")
    assert m.high is not m.low and len(out.state_trace)==4
    assert out.low_updates==16 and out.high_updates==8
    assert not torch.equal(out.high_state,out.low_state)
def test_unequal_and_equal_frequency_have_matched_low_but_distinct_high_calls():
    m=model(); x=sample(); unequal=m(x,route="full_hrm"); equal=m(x,route="equal_frequency")
    assert unequal.low_updates==equal.low_updates==16 and unequal.high_updates==8 and equal.high_updates==16
    assert not torch.equal(unequal.logits,equal.logits)
def test_reset_carry_and_one_state_interventions_change_outputs():
    m=model(); x=sample(); base=m(x)
    for route in ("high_reset","low_reset","one_state"):
        assert not torch.equal(base.segment_logits[-1],m(x,route=route).segment_logits[-1])
    carried=m.forward_train(x,route="full_hrm"); carried.segment_logits[-1].square().mean().backward(); detached_grad=m.input.weight.grad.detach().clone();m.zero_grad(set_to_none=True)
    linked=m.forward_train(x,route="segment_detach_disabled");linked.segment_logits[-1].square().mean().backward()
    # Exact one-step local credit already severs the earlier segment graph; this
    # ablation is intentionally forward- and gradient-redundant unless full BPTT
    # is also enabled, which is covered by the separate full_gradient control.
    assert torch.equal(carried.segment_logits[-1],linked.segment_logits[-1]) and torch.equal(detached_grad,m.input.weight.grad)
def test_one_step_and_full_gradient_match_forward_but_change_gradient_graph():
    m=model(); x=sample(); one=m.forward_train(x,route="full_hrm"); full=m.forward_train(x,route="full_gradient")
    assert torch.allclose(one.logits,full.logits)
    one.logits.square().mean().backward(); one_grad=float(m.low.qkv.weight.grad.norm()); m.zero_grad(set_to_none=True)
    full.logits.square().mean().backward(); full_grad=float(m.low.qkv.weight.grad.norm())
    assert one_grad>0 and full_grad>0 and one_grad!=full_grad
def test_detached_deep_supervision_and_q_targets_have_gradients():
    m=model(); out=m.forward_train(sample(),route="full_hrm"); truth=torch.zeros_like(out.logits)
    prediction=sum(F.binary_cross_entropy_with_logits(x,truth) for x in out.segment_logits)
    correctness=torch.tensor([[1.,0.]]*truth.shape[0]); qloss=sum(F.binary_cross_entropy(q,correctness) for q in out.segment_q)
    (prediction+qloss).backward()
    assert m.high.qkv.weight.grad is not None and m.low.qkv.weight.grad is not None and m.q_head.weight.grad is not None
def test_act_is_deterministic_target_free_and_bounded():
    m=model().eval(); x=sample(); a=m(x); b=m(x)
    assert torch.equal(a.logits,b.logits) and torch.equal(a.segments,b.segments)
    assert int(a.segments.min())>=1 and int(a.segments.max())<=4
    assert "target" not in inspect.signature(SourceCompleteHRM.forward).parameters
def test_fixed_random_q_disabled_and_q_shuffle_controls_exist():
    m=model(); x=sample(batch=4)
    outputs={r:m(x,route=r) for r in ("fixed_depth","equal_average_act","random_act","q_disabled","q_shuffled")}
    assert int(outputs["fixed_depth"].segments[0])==4
    assert int(outputs["equal_average_act"].segments[0])==2
    assert all(r in ROUTES for r in outputs)
def test_reload_is_bit_exact(tmp_path):
    m=model().eval(); x=sample(); expected=m(x); path=tmp_path/"hrm.safetensors"; save_file({k:v.detach().contiguous() for k,v in m.state_dict().items()},path)
    restored=model(); restored.load_state_dict(load_file(path)); observed=restored.eval()(x)
    assert torch.equal(expected.logits,observed.logits) and torch.equal(expected.q_values,observed.q_values)
def test_randomized_weights_change_outputs():
    m=model(); x=sample(); before=m(x).logits; m.randomized_intervention(683); assert not torch.equal(before,m(x).logits)
def test_surface_fails_closed_and_has_no_programmed_arithmetic():
    source=inspect.getsource(SourceCompleteHRM.forward).lower()
    for word in ("answer","correctness","barrett","montgomery","crt","long_division"): assert word not in source
    m=model()
    for bad in (torch.randn(2,5,8),torch.full((2,6,8),float("nan"))):
        try:m(bad)
        except ValueError:pass
        else:raise AssertionError("must fail closed")
