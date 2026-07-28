from __future__ import annotations
import ast,copy,inspect,textwrap
from pathlib import Path
import pytest,torch,yaml
from safetensors.torch import load_file
from experiments.exp_668_h2l3_magicnorm_credit.model import H2L3Cell,MagicNormModule,ROUTES,TRAINED_ROUTES,parameterless_rms_norm
from experiments.exp_668_h2l3_magicnorm_credit.run import Experiment,bits,build_model,credit_diagnostic,features,load_exp300,sha256,train_route
ROOT=Path(__file__).resolve().parents[1];CONFIG_PATH=ROOT/"configs/exp_668_h2l3_magicnorm_credit.yaml";CONFIG=yaml.safe_load(CONFIG_PATH.read_text());SOURCE=ROOT/CONFIG["source_checkpoint"]
def model():torch.manual_seed(668);return build_model(CONFIG,SOURCE)
def sample(width=16,batch=3):return features(list(range(batch)),[7]*batch,[65521]*batch,[0,1,0][:batch],width)
def test_frozen_sources_fixtures_roots_routes():
    assert sha256(ROOT/CONFIG["paper"]["path"])==CONFIG["paper"]["sha256"] and sha256(SOURCE)==CONFIG["source_checkpoint_sha256"]
    for name,digest in CONFIG["fixture_hashes"].items():assert sha256(ROOT/".scratch/exp-658-joint-quotient-carry-transducer/pinned_sources"/name)==digest
    assert tuple(CONFIG["routes"])==ROUTES and tuple(CONFIG["trained_routes"])==TRAINED_ROUTES and CONFIG["roots"]=={"calibration":20261222,"viability":20261223,"confirmations":[20261224,20261225]}
def test_proposal_bit_identical_and_frozen():
    m=model();batch=sample();out=m(batch,route="proposal_only");direct=m.base(batch);assert torch.equal(out.logits,direct[0]) and torch.equal(out.quotient_logits,direct[1]) and all(not p.requires_grad for p in m.base.parameters())
@pytest.mark.parametrize("route",ROUTES[1:-1])
def test_exact_initialization(route):
    m=model();batch=sample();assert torch.equal(m(batch,route=route).logits,m(batch,route="proposal_only").logits)
def test_h2l3_schedule_separate_modules_and_states():
    m=model();assert isinstance(m.high_module,MagicNormModule) and isinstance(m.low_module,MagicNormModule) and m.high_module is not m.low_module;out=m(sample());assert out.iterations.unique().item()==8 and len(out.state_trace)==8 and out.high_state.shape==out.low_state.shape and not torch.equal(out.high_state,out.low_state)
def test_magicnorm_parameterless_and_bounded():
    x=torch.randn(4,92,16)*10;y=parameterless_rms_norm(x);assert not list(parameterless_rms_norm.__annotations__.values())==[] and torch.allclose(y.square().mean(1),torch.ones_like(y[:,0]),atol=1e-4)
def test_warmup_credit_horizon_progresses_two_to_five():
    m=model();assert [m.horizon(x) for x in (0.,.34,.67,1.)]==[2,3,4,5];assert m.forward_train(sample(),progress=0.).credit_horizon==2 and m.forward_train(sample(),progress=1.).credit_horizon==5
def test_component_routes_are_separable_after_intervention():
    m=model();batch=sample()
    with torch.no_grad():
        m.high_module.down.weight.fill_(.01);m.low_module.down.weight.fill_(.01)
    states={r:m(batch,route=r).state for r in ("full","hierarchy_only","credit_only","flat_compute_matched","no_warmup_raw_gradient","high_reset","low_reset")};assert not torch.equal(states["full"],states["hierarchy_only"]);assert not torch.equal(states["full"],states["credit_only"]);assert not torch.equal(states["high_reset"],states["low_reset"])
@pytest.mark.parametrize("route",TRAINED_ROUTES)
def test_each_trained_route_finite_nonzero_gradient(route):
    m=model();out=m.forward_train(sample(),route=route,progress=.5);sum(x.square().mean() for x in out.supervision_logits).backward();g=[p.grad for p in m.parameters() if p.requires_grad and p.grad is not None];assert g and all(torch.isfinite(x).all() for x in g) and sum(float(x.abs().sum()) for x in g)>0
def test_randomization_changes_state():
    m=model();batch=sample();before=m(batch).state;m.randomized_intervention(9);assert not torch.equal(before,m(batch,route="randomized").state)
def test_credit_diagnostic_separates_horizons():
    raw=load_exp300().BigIntSampler([65521],4).transitions(4);cases={"state":raw[0],"multiplicand":raw[1],"modulus":raw[2],"input_bit":raw[3],"target":raw[4],"quotient":raw[5].tolist(),"width":16};d=credit_diagnostic(model(),cases);assert d["full-p0"]["horizon"]==2 and d["full-p1"]["horizon"]==5 and d["no_warmup_raw_gradient-p1"]["horizon"]==8 and all(x["finite"] for x in d.values())
def test_target_free_fixed_schedule_and_prohibited_arithmetic():
    sig=inspect.signature(H2L3Cell.forward);assert all(x not in sig.parameters for x in ("target","answer","modulus"));src=inspect.getsource(H2L3Cell.forward);assert "%" not in src and "//" not in src;tree=ast.parse(inspect.getsource(__import__("experiments.exp_668_h2l3_magicnorm_credit.model",fromlist=["*"])));assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {"pow","divmod","crt","mod_inverse"} for n in ast.walk(tree))
    with pytest.raises(ValueError):model()(sample(),high_depth=3)
def test_roots_splits_and_formal_lock(tmp_path):
    e=Experiment(CONFIG_PATH,tmp_path,"preregister",20261222)
    for x in e.splits.values():assert set(x["train"]).isdisjoint(x["validation"]) and set(x["validation"]).isdisjoint(x["test"])
    with pytest.raises(ValueError):Experiment(CONFIG_PATH,tmp_path,"formal",20261222)
def test_one_update_checkpoint_roundtrip(tmp_path):
    cfg=copy.deepcopy(CONFIG);cfg["training"]["batch_size"]=2;cfg["training"]["schedule"]=[16,16,16,16];cfg["data"]["training_widths"]=[16];cfg["data"]["widths"]=[16];exp=load_exp300();raw=exp.bigint_prime_split({"data":{"prime_bits_min":13,"prime_bits_max":16,"train_prime_identities":8,"test_prime_identities":4}},66801);split={16:{"train":raw["train"],"validation":raw["test"][:2],"test":raw["test"][2:]}};_,r=train_route(cfg,exp,SOURCE,split,20261222,"full",1,tmp_path,"calibration");assert r["checkpoint_reload_exact"]
def test_checkpoint_geometry_and_parameter_matching():
    m=model();assert set(load_file(SOURCE))==set(m.base.state_dict()) and m.trainable_parameter_count()>0 and m.high_module.up.weight.numel()==m.low_module.up.weight.numel()
