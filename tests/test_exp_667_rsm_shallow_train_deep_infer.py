from __future__ import annotations
import ast,copy,inspect,textwrap
from pathlib import Path
import pytest,torch,yaml
from safetensors.torch import load_file
from experiments.exp_667_rsm_shallow_train_deep_infer.model import ROUTES,TRAINED_ROUTES,RSMCell
from experiments.exp_667_rsm_shallow_train_deep_infer.run import Experiment,bits,build_model,features,load_exp300,stability_diagnostic,train_route
ROOT=Path(__file__).resolve().parents[1]; CONFIG_PATH=ROOT/"configs/exp_667_rsm_shallow_train_deep_infer.yaml"; CONFIG=yaml.safe_load(CONFIG_PATH.read_text()); SOURCE=ROOT/CONFIG["source_checkpoint"]
def model(): torch.manual_seed(667); return build_model(CONFIG,SOURCE)
def sample(width=16,batch=3): return features(list(range(batch)),[7]*batch,[65521]*batch,[0,1,0][:batch],width)

def test_frozen_contract():
    assert CONFIG["paper"]["sha256"]=="c1cb3a00564e9bc4828c0ddd1b8161d702b32da445eb34920ffe6aab81261c17"
    assert CONFIG["source_checkpoint_sha256"]=="cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd"
    assert tuple(CONFIG["routes"])==ROUTES and tuple(CONFIG["trained_routes"])==TRAINED_ROUTES
    assert CONFIG["roots"]=={"calibration":20261218,"viability":20261219,"confirmations":[20261220,20261221]}

def test_proposal_identical_and_frozen():
    m=model(); batch=sample(); result=m(batch,route="proposal_only"); direct=m.base(batch)
    assert torch.equal(result.logits,direct[0]) and torch.equal(result.quotient_logits,direct[1]); assert all(not p.requires_grad for p in m.base.parameters())

@pytest.mark.parametrize("route",ROUTES[1:-1])
def test_exact_initialized_routes(route):
    m=model(); batch=sample(); assert torch.equal(m(batch,route=route).logits,m(batch,route="proposal_only").logits)

def test_one_operator_two_states_and_depth_agnostic_sharing():
    m=model(); assert sum(1 for _ in m.modules() if type(_).__name__=="SharedRSMOperator")==1
    shallow=m(sample(),high_depth=2,low_depth=1); deep=m(sample(),high_depth=4,low_depth=3)
    assert shallow.iterations.unique().item()==4 and deep.iterations.unique().item()==16
    assert shallow.high_state.shape==shallow.low_state.shape

def test_target_free_inference_and_detached_warmup():
    sig=inspect.signature(RSMCell.forward); assert all(x not in sig.parameters for x in ("target","answer","modulus"))
    source=inspect.getsource(RSMCell._run); assert ".detach()" in source
    tree=ast.parse(textwrap.dedent(inspect.getsource(RSMCell.forward))); assert {x.id for x in ast.walk(tree) if isinstance(x,ast.Name)}.isdisjoint({"target","answer","modulus"})

def test_independent_depth_curricula_and_controls():
    assert CONFIG["training"]["high_curriculum"]!=CONFIG["training"]["low_curriculum"]
    m=model(); batch=sample();
    with torch.no_grad(): m.operator.output.weight.fill_(.01)
    full=m(batch,route="rsm_full",high_depth=3,low_depth=2); assert not torch.equal(full.high_state,full.low_state)
    assert not torch.equal(full.state,m(batch,route="high_reset",high_depth=3,low_depth=2).state)
    assert not torch.equal(full.state,m(batch,route="low_reset",high_depth=3,low_depth=2).state)

@pytest.mark.parametrize("route",TRAINED_ROUTES)
def test_finite_nonzero_gradient(route):
    m=model(); out=m.forward_train(sample(),route=route,high_depth=2,low_depth=2,generator=torch.Generator().manual_seed(4)); sum(x.square().mean() for x in out.supervision_logits).backward(); grads=[p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]; assert grads and sum(float(x.abs().sum()) for x in grads)>0 and all(torch.isfinite(x).all() for x in grads)

def test_randomization_changes_state():
    m=model(); batch=sample(); before=m(batch).state; m.randomized_states(7); assert not torch.equal(before,m(batch,route="randomized_states").state)

def test_stability_diagnostic_records_depths_and_resets():
    raw=load_exp300().BigIntSampler([65521],4).transitions(4); cases={"state":raw[0],"multiplicand":raw[1],"modulus":raw[2],"input_bit":raw[3],"target":raw[4],"quotient":raw[5].tolist(),"width":16}; record=stability_diagnostic(model(),cases); assert set(record["depths"])=={"1","2","4"} and record["fixed_depth_target_free"]

def test_roots_and_splits(tmp_path):
    exp=Experiment(CONFIG_PATH,tmp_path,"preregister",20261218)
    for x in exp.splits.values(): assert set(x["train"]).isdisjoint(x["validation"]) and set(x["validation"]).isdisjoint(x["test"])
    with pytest.raises(ValueError): Experiment(CONFIG_PATH,tmp_path,"formal",20261218)

def test_one_update_roundtrip(tmp_path):
    cfg=copy.deepcopy(CONFIG); cfg["training"]["batch_size"]=2; cfg["training"]["schedule"]=[16]; cfg["training"]["high_curriculum"]=[2]; cfg["training"]["low_curriculum"]=[1]; cfg["data"]["training_widths"]=[16]; cfg["data"]["widths"]=[16]; exp300=load_exp300(); raw=exp300.bigint_prime_split({"data":{"prime_bits_min":13,"prime_bits_max":16,"train_prime_identities":8,"test_prime_identities":4}},66701); split={16:{"train":raw["train"],"validation":raw["test"][:2],"test":raw["test"][2:]}}; _,record=train_route(cfg,exp300,SOURCE,split,20261218,"rsm_full",1,tmp_path,"calibration"); assert record["checkpoint_reload_exact"]

def test_no_prohibited_inference_arithmetic():
    source=inspect.getsource(__import__("experiments.exp_667_rsm_shallow_train_deep_infer.model",fromlist=["*"])); tree=ast.parse(source); assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {"pow","divmod","crt","mod_inverse"} for n in ast.walk(tree)); inference=inspect.getsource(RSMCell.forward); assert "%" not in inference and "//" not in inference

def test_checkpoint_geometry():
    m=model(); assert set(load_file(SOURCE))==set(m.base.state_dict()) and m.trainable_parameter_count()>0
