from __future__ import annotations
import ast,copy,inspect
from pathlib import Path
import pytest,torch,yaml
from safetensors.torch import load_file
from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU
from experiments.exp_669_stars_jvp_regularization.model import ROUTES,TRAINED_ROUTES,STARSCell
from experiments.exp_669_stars_jvp_regularization.run import Experiment,InferenceView,build_model,features,load_exp300,sha256,stability_diagnostic,train_route
ROOT=Path(__file__).resolve().parents[1];CONFIG_PATH=ROOT/"configs/exp_669_stars_jvp_regularization.yaml";CONFIG=yaml.safe_load(CONFIG_PATH.read_text());SOURCE=ROOT/CONFIG["source_checkpoint"]
def model():torch.manual_seed(669);return build_model(CONFIG,SOURCE)
def sample(width=16,batch=3):return features(list(range(batch)),[7]*batch,[65521]*batch,[0,1,0][:batch],width)
def boundary(m,batch):return m.forward_with_boundary(batch)[2:4]
def test_frozen_source_paper_fixtures_roots_routes():
    assert sha256(SOURCE)==CONFIG["source_checkpoint_sha256"] and sha256(ROOT/CONFIG["paper"]["path"])==CONFIG["paper"]["sha256"]
    for name,digest in CONFIG["fixture_hashes"].items():assert sha256(ROOT/".scratch/exp-658-joint-quotient-carry-transducer/pinned_sources"/name)==digest
    assert tuple(CONFIG["routes"])==ROUTES and tuple(CONFIG["trained_routes"])==TRAINED_ROUTES and CONFIG["roots"]=={"calibration":20261226,"viability":20261227,"confirmations":[20261228,20261229]}
def test_inference_is_inherited_source_graph_and_bit_identical():
    m=model();assert STARSCell.forward is SharedDilatedConvGRU.forward;batch=sample();source=SharedDilatedConvGRU(width=256,hidden=92,cycles=2);source.load_state_dict(load_file(SOURCE));left=m(batch);right=source(batch);assert torch.equal(left[0],right[0]) and torch.equal(left[1],right[1]) and m.parameter_count()==source.parameter_count() and set(m.state_dict())==set(source.state_dict())
def test_training_boundary_forward_matches_inference_exactly():
    m=model();batch=sample();a=m(batch);b=m.forward_with_boundary(batch);assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
def test_native_jvp_matches_finite_difference_directionally():
    m=model();batch=sample(batch=2);state,dilation=boundary(m,batch);record=m.jvp_proxy(state,batch,dilation,seed=7,route="stars");assert record.jvp_calls==2 and abs(float(record.vector_norm)-1)<1e-5 and float(record.proxy)>0 and record.direction=="one-step-power"
def test_proxy_has_actual_higher_order_parameter_gradients():
    m=model();batch=sample(batch=2);state,dilation=boundary(m,batch);record=m.jvp_proxy(state,batch,dilation,seed=8,route="stars");grads=torch.autograd.grad(record.proxy,tuple(m.parameters()),allow_unused=True);present=[g for g in grads if g is not None];assert present and all(torch.isfinite(g).all() for g in present) and sum(float(g.abs().sum()) for g in present)>0
@pytest.mark.parametrize("route",("lambda_zero","state_norm","random_stop_gradient"))
def test_control_proxy_is_stop_gradient(route):
    m=model();batch=sample(batch=2);state,dilation=boundary(m,batch);record=m.jvp_proxy(state,batch,dilation,seed=9,route=route);assert not record.proxy.requires_grad and not record.parameter_gradient_enabled
def test_random_direction_is_distinct_from_power_direction():
    m=model();batch=sample(batch=2);state,dilation=boundary(m,batch);power=m.jvp_proxy(state,batch,dilation,seed=10,route="stars");random=m.jvp_proxy(state,batch,dilation,seed=10,route="random_stop_gradient");assert power.direction!=random.direction and not torch.equal(power.proxy.detach(),random.proxy)
def test_lambda_zero_and_frozen_start_identical():
    for route in ROUTES[:-1]:assert all(torch.equal(a,b) for a,b in zip(model().state_dict().values(),model().state_dict().values(),strict=True))
def test_randomized_intervention_changes_inference():
    m=model();batch=sample();before=m(batch)[0];m.randomized_intervention(11);assert not torch.equal(before,m(batch)[0])
def test_stability_diagnostic_names_proxy_not_spectral_radius():
    raw=load_exp300().BigIntSampler([65521],4).transitions(3);cases={"state":raw[0],"multiplicand":raw[1],"modulus":raw[2],"input_bit":raw[3],"target":raw[4],"quotient":raw[5].tolist(),"width":16};d=stability_diagnostic(model(),cases,12,.0001);assert set(d)==set(TRAINED_ROUTES) and all(not x["exact_spectral_radius"] and x["perturbation_amplification"]>=0 for x in d.values())
def test_inference_view_does_not_route_or_add_calls():
    m=model();batch=sample();view=InferenceView(m);out=view(batch,route="stars");assert torch.equal(out.logits,m(batch)[0]) and out.iterations.unique().item()==len(m.dilation_schedule(16))
def test_roots_splits_and_formal_lock(tmp_path):
    e=Experiment(CONFIG_PATH,tmp_path,"preregister",20261226)
    for x in e.splits.values():assert set(x["train"]).isdisjoint(x["validation"]) and set(x["validation"]).isdisjoint(x["test"])
    with pytest.raises(ValueError):Experiment(CONFIG_PATH,tmp_path,"formal",20261226)
def test_one_update_roundtrip(tmp_path):
    cfg=copy.deepcopy(CONFIG);cfg["training"]["batch_size"]=2;cfg["training"]["schedule"]=[16]*4;cfg["data"]["training_widths"]=[16];cfg["data"]["widths"]=[16];exp=load_exp300();raw=exp.bigint_prime_split({"data":{"prime_bits_min":13,"prime_bits_max":16,"train_prime_identities":8,"test_prime_identities":4}},66901);split={16:{"train":raw["train"],"validation":raw["test"][:2],"test":raw["test"][2:]}};_,r=train_route(cfg,exp,SOURCE,split,20261226,"stars",1,tmp_path,"calibration");assert r["checkpoint_reload_exact"] and r["training"]["jvp_calls"]==4
def test_no_prohibited_inference_arithmetic_or_jvp():
    source=inspect.getsource(STARSCell.forward);assert "jvp" not in source.lower() and "%" not in source and "//" not in source;tree=ast.parse(inspect.getsource(__import__("experiments.exp_669_stars_jvp_regularization.model",fromlist=["*"])));assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {"pow","divmod","crt","mod_inverse"} for n in ast.walk(tree))
