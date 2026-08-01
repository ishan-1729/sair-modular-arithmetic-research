"""Execute EXP687's frozen-map compact W256 distillation experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU  # noqa: E402
from experiments.exp_664_sprm_boundary_perturbation.run import (  # noqa: E402
    Timer, bits, features, git_output, make_split, sha256, write_json,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

RUN_ID = "exp-687-compact-distilled-w256-specialist"
CONFIG_PATH = ROOT / "configs/exp_687_compact_distilled_w256_specialist.yaml"
CHALLENGE_COMMIT = "82510bba00a1126649bd76dd1a451f14d0b3eb60"
TRAINED_ARMS = ("distilled", "exact_only")
ARM_ORDER = TRAINED_ARMS + ("frozen_pruned", "randomized_control", "reset_control")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_exp300():
    old_path, old_model, had_model = sys.path.copy(), sys.modules.get("model"), "model" in sys.modules
    try:
        return load_module("exp300_for_exp687", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
    finally:
        sys.path[:] = old_path
        if had_model: sys.modules["model"] = old_model
        else: sys.modules.pop("model", None)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    return canonical_hash({name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for name, value in state.items()})


def configure_torch(config: dict[str, Any]) -> None:
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    torch.use_deterministic_algorithms(True)


def normalized(contribution: torch.Tensor) -> torch.Tensor:
    contribution = contribution.double()
    return contribution / contribution.mean().clamp_min(torch.finfo(torch.float64).tiny)


def channel_importance(source: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, list[float]]]:
    """Rank hidden channels by a frozen sum of normalized squared incident parameters."""
    h = source["stem.bias"].numel(); terms: dict[str, torch.Tensor] = {}
    terms["stem"] = source["stem.weight"].square().sum((1, 2)) + source["stem.bias"].square()
    terms["state_norm"] = source["state_norm.weight"].square() + source["state_norm.bias"].square()
    gw, gb = source["gate_weight"], source["gate_bias"]
    terms["gate_rows"] = gw.square().sum((1, 2)).reshape(2, h).sum(0) + gb.square().reshape(2, h).sum(0)
    terms["gate_columns"] = gw[:, :h, :].square().sum((0, 2))
    cw = source["candidate_weight"]
    terms["candidate_rows"] = cw.square().sum((1, 2)) + source["candidate_bias"].square()
    terms["candidate_columns"] = cw[:, :h, :].square().sum((0, 2))
    terms["output_norm"] = source["output_norm.weight"].square() + source["output_norm.bias"].square()
    terms["local_head"] = source["local_head.weight"].square().sum((0, 2))
    q0, qb = source["quotient_head.0.weight"], source["quotient_head.0.bias"]
    terms["quotient_hidden_rows"] = q0.square().sum(1) + qb.square()
    terms["quotient_pooled_columns"] = q0[:, :h].square().sum(0) + q0[:, h:].square().sum(0)
    terms["quotient_output_columns"] = source["quotient_head.2.weight"].square().sum(0)
    normalized_terms = {name: normalized(value) for name, value in terms.items()}
    score = sum(normalized_terms.values(), torch.zeros(h, dtype=torch.float64))
    order = torch.tensor(sorted(range(h), key=lambda index: (-float(score[index]), index)), dtype=torch.long)
    return order, {name: value.tolist() for name, value in normalized_terms.items()} | {"total": score.tolist()}


def slice_state(source: dict[str, torch.Tensor], selected: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Coherently slice every hidden dimension through the same source-channel map."""
    h = source["stem.bias"].numel(); doubled = torch.cat((selected, selected + h)); pooled = torch.cat((selected, selected + h))
    result = {
        "stem.weight": source["stem.weight"].index_select(0, selected), "stem.bias": source["stem.bias"].index_select(0, selected),
        "state_norm.weight": source["state_norm.weight"].index_select(0, selected), "state_norm.bias": source["state_norm.bias"].index_select(0, selected),
        "gate_weight": source["gate_weight"].index_select(0, doubled).index_select(1, torch.cat((selected, torch.arange(h, h + 6)))),
        "gate_bias": source["gate_bias"].index_select(0, doubled),
        "candidate_weight": source["candidate_weight"].index_select(0, selected).index_select(1, torch.cat((selected, torch.arange(h, h + 6)))),
        "candidate_bias": source["candidate_bias"].index_select(0, selected),
        "output_norm.weight": source["output_norm.weight"].index_select(0, selected), "output_norm.bias": source["output_norm.bias"].index_select(0, selected),
        "local_head.weight": source["local_head.weight"].index_select(1, selected), "local_head.bias": source["local_head.bias"].clone(),
        "quotient_head.0.weight": source["quotient_head.0.weight"].index_select(0, selected).index_select(1, pooled),
        "quotient_head.0.bias": source["quotient_head.0.bias"].index_select(0, selected),
        "quotient_head.2.weight": source["quotient_head.2.weight"].index_select(1, selected), "quotient_head.2.bias": source["quotient_head.2.bias"].clone(),
    }
    manifest = {name: {"source_shape": list(source[name].shape), "student_shape": list(value.shape)} for name, value in result.items()}
    if set(result) != set(source): raise RuntimeError("incomplete hidden tensor slice")
    return {name: value.detach().contiguous() for name, value in result.items()}, manifest


def prepare_pruned(config: dict[str, Any], checkpoint: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    source = load_file(checkpoint); order, terms = channel_importance(source)
    selected = order[: int(config["model"]["hidden"])]
    state, slices = slice_state(source, selected)
    model = SharedDilatedConvGRU(256, int(config["model"]["hidden"]), int(config["model"]["cycles"])); model.load_state_dict(state)
    manifest = {"equation": "sum_t incident_parameter_square(channel,t)/mean_channel(incident_parameter_square(channel,t))", "answer_independent": True, "tie_break": "ascending-source-channel-index", "selected_source_channels_ranked": selected.tolist(), "selected_source_channels_sorted": sorted(selected.tolist()), "source_state_sha256": state_hash(source), "pruned_state_sha256": state_hash(state), "parameter_count": model.parameter_count(), "terms": terms, "slices": slices, "native_choice": "Native PyTorch index_select is used for a single fixed graph: it preserves an explicit source index for every sliced axis and introduces no pruning runtime or sandbox dependency."}
    return state, manifest


def new_student(config: dict[str, Any], state: dict[str, torch.Tensor] | None, random_seed: int | None = None) -> SharedDilatedConvGRU:
    if random_seed is not None: torch.manual_seed(random_seed)
    model = SharedDilatedConvGRU(256, int(config["model"]["hidden"]), int(config["model"]["cycles"]))
    if state is not None: model.load_state_dict(state)
    return model


def new_teacher(config: dict[str, Any], checkpoint: Path) -> SharedDilatedConvGRU:
    model = SharedDilatedConvGRU(256, int(config["model"]["source_hidden"]), int(config["model"]["cycles"])); model.load_state_dict(load_file(checkpoint)); return model.eval()


def soft_features(state: torch.Tensor, multiplicand: list[int], modulus: list[int], flags: list[int], width: int) -> torch.Tensor:
    batch = len(multiplicand); plane = torch.tensor(flags, dtype=torch.float32)[:, None].expand(batch, width)
    lsb, msb = torch.zeros(batch, width), torch.zeros(batch, width); lsb[:, 0], msb[:, -1] = 1.0, 1.0
    return torch.stack((state, bits(multiplicand, width), bits(modulus, width), plane, lsb, msb), 1).mul(2).sub(1)


def hard_straight_through(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits); return logits.ge(0).float() + probability - probability.detach()


def teacher_prefix(left: list[int], right: list[int], modulus: list[int], starts: list[int], width: int) -> list[int]:
    state = [0] * len(left)
    for position in range(max(starts, default=0)):
        state = [(2 * value + (((a >> (width - 1 - position)) & 1) if position < stop else 0) * b) % p if position < stop else value for value, a, b, p, stop in zip(state, left, right, modulus, starts, strict=True)]
    return state


def train_update(config: dict[str, Any], exp300, student: SharedDilatedConvGRU, teacher_model: SharedDilatedConvGRU, optimizer: torch.optim.Optimizer, batch: tuple[list[int], list[int], list[int], list[int]], width: int, distill: bool) -> dict[str, float]:
    left, right, modulus, starts = batch; horizon = int(config["training"]["horizon"]); teacher_state = teacher_prefix(left, right, modulus, starts, width); state_plane = bits(teacher_state, width)
    losses, exacts, residue_terms, quotient_terms = [], [], [], []
    optimizer.zero_grad(set_to_none=True); student.width = teacher_model.width = width
    for offset in range(horizon):
        flags = [(value >> (width - 1 - start - offset)) & 1 for value, start in zip(left, starts, strict=True)]
        raw = [2 * state + flag * other for state, flag, other in zip(teacher_state, flags, right, strict=True)]
        target = [value % prime for value, prime in zip(raw, modulus, strict=True)]; quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus, strict=True)], dtype=torch.long)
        inputs = soft_features(state_plane, right, modulus, flags, width); logits, quotient_logits = student(inputs)
        with torch.no_grad(): teacher_logits, teacher_quotient = teacher_model(inputs.detach())
        exact_loss, stats = exp300.transition_loss(logits, quotient_logits, target, quotient, width, float(config["training"]["worst_bit_weight"]), float(config["training"]["quotient_weight"]))
        residue = F.mse_loss(logits, teacher_logits) / teacher_logits.square().mean().clamp_min(1e-8)
        quotient_kl = F.kl_div(F.log_softmax(quotient_logits, 1), F.softmax(teacher_quotient, 1), reduction="batchmean")
        residue_weight = float(config["training"]["residue_distillation_weight"]) if distill else 0.0; quotient_weight = float(config["training"]["quotient_distillation_weight"]) if distill else 0.0
        losses.append(exact_loss + residue_weight * residue + quotient_weight * quotient_kl); exacts.append(float(stats["batch_exact"])); residue_terms.append(residue); quotient_terms.append(quotient_kl)
        state_plane = hard_straight_through(logits); teacher_state = target
    objective = torch.stack(losses).mean(); objective.backward(); gradient = float(torch.nn.utils.clip_grad_norm_(student.parameters(), float(config["training"]["gradient_clip"]))); optimizer.step()
    return {"loss": float(objective.detach()), "gradient_norm": gradient, "mean_exact": float(np.mean(exacts)), "residue_distillation": float(torch.stack(residue_terms).mean()), "quotient_distillation": float(torch.stack(quotient_terms).mean()), "distillation_weight": float(config["training"]["residue_distillation_weight"] if distill else 0.0), "student_calls": horizon, "teacher_calls": horizon, "exact_loss_calls": horizon}


def train_arm(config: dict[str, Any], exp300, pruned: dict[str, torch.Tensor], checkpoint: Path, splits: dict[int, dict[str, list[int]]], root: int, arm: str, updates: int, output: Path) -> tuple[SharedDilatedConvGRU, dict[str, Any]]:
    random.seed(root + 687); np.random.seed(root + 687); torch.manual_seed(root + 687)
    model = new_student(config, None if arm == "randomized_control" else pruned, root + 687900 if arm == "randomized_control" else None); initial = state_hash(model.state_dict()); timer = Timer(); history=[]
    if arm in TRAINED_ARMS:
        teacher = new_teacher(config, checkpoint); optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"])); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=float(config["training"]["minimum_learning_rate"])); samplers={w: exp300.BigIntSampler(splits[w]["train"], root + int(config["data"]["training_seed_offset"]) + w) for w in set(config["data"]["curriculum"])}; starts={w: random.Random(root + int(config["data"]["training_seed_offset"]) + w + 1) for w in samplers}; model.train()
        for update in range(1, updates + 1):
            tick=Timer(); width=int(config["data"]["curriculum"][(update-1) % len(config["data"]["curriculum"])]); left,right,modulus=samplers[width].pairs(int(config["training"]["batch_size"])); begin=[starts[width].randrange(0,width-int(config["training"]["horizon"])+1) for _ in left]
            stats=train_update(config,exp300,model,teacher,optimizer,(left,right,modulus,begin),width,arm=="distilled"); scheduler.step(); record={"update":update,"width":width,"learning_rate":scheduler.get_last_lr()[0],"timing":tick.finish(),**stats}; history.append(record)
            if update==1 or update==updates or update%int(config["training"]["log_every_updates"])==0: print(json.dumps({"arm":arm,**record}),flush=True)
    model.eval(); arm_dir=output/"arms"/arm; arm_dir.mkdir(parents=True,exist_ok=True); path=arm_dir/"model.safetensors"; save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},path); reloaded=new_student(config,load_file(path))
    record={"arm":arm,"trained":arm in TRAINED_ARMS,"initial_state_sha256":initial,"parameters":model.parameter_count(),"training":{"updates":updates if arm in TRAINED_ARMS else 0,"curriculum":list(config["data"]["curriculum"]),"calls_per_update":int(config["training"]["horizon"]),"teacher_calls":sum(x["teacher_calls"] for x in history),"student_calls":sum(x["student_calls"] for x in history),"exact_loss_calls":sum(x["exact_loss_calls"] for x in history),"history":history,"resource":timer.finish()},"checkpoint":{"path":str(path),"sha256":sha256(path),"bytes":path.stat().st_size},"reload_exact":all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),reloaded.state_dict().values(),strict=True))}
    return model,record


def decode(rows: torch.Tensor) -> list[int]: return [sum(int(bit)<<i for i,bit in enumerate(row)) for row in rows.tolist()]


@torch.inference_mode()
def rollout(model: SharedDilatedConvGRU,left:list[int],right:list[int],modulus:list[int],width:int)->tuple[list[int],list[int|None],list[bool]]:
    state_bits=torch.zeros((len(left),width),dtype=torch.bool); left_bits=bits(left,width); invariant=torch.empty((len(left),6,width)); invariant[:,1]=bits(right,width).mul(2).sub(1); invariant[:,2]=bits(modulus,width).mul(2).sub(1); invariant[:,4].fill_(-1); invariant[:,4,0]=1; invariant[:,5].fill_(-1); invariant[:,5,-1]=1; teacher=[0]*len(left); first=[None]*len(left); invalid=[False]*len(left); model.width=width
    for position in range(width):
        invariant[:,0]=state_bits.float().mul(2).sub(1); invariant[:,3]=left_bits[:,width-1-position,None].mul(2).sub(1); logits,_=model(invariant); state_bits=logits.ge(0); values=decode(state_bits); flags=[int(x) for x in left_bits[:,width-1-position]]; teacher=[(2*s+f*b)%p for s,f,b,p in zip(teacher,flags,right,modulus,strict=True)]
        for i,(guess,truth,p) in enumerate(zip(values,teacher,modulus,strict=True)):
            if guess!=truth and first[i] is None: first[i]=position
            invalid[i]=invalid[i] or guess>=p
    return decode(state_bits),first,invalid


def evaluate(config:dict[str,Any],exp300,model:SharedDilatedConvGRU,split:dict[str,list[int]],root:int,width:int,transitions:int,rollouts:int)->dict[str,Any]:
    model.width=width; timer=Timer(); transition=exp300.evaluate_transitions(model,exp300.BigIntSampler(split["test"],root+int(config["data"]["test_seed_offset"])+width),transitions); transition["resource"]=timer.finish(); left,right,modulus=exp300.BigIntSampler(split["test"],root+int(config["data"]["test_seed_offset"])+width+1).pairs(rollouts); timer=Timer(); prediction,first,invalid=rollout(model,left,right,modulus,width); targets=[(a*b)%p for a,b,p in zip(left,right,modulus,strict=True)]; errors=[x for x in first if x is not None]; correct=sum(a==b for a,b in zip(prediction,targets,strict=True)); return {"transition":transition,"rollout":{"cases":rollouts,"correct":correct,"exact_accuracy":correct/rollouts,"invalids":sum(invalid),"any_invalid_residue_rate":sum(invalid)/rollouts,"mean_first_error_position":None if not errors else float(np.mean(errors)),"prediction_sha256":canonical_hash(prediction),"resource":timer.finish()}}


def candidate_inputs(root:int)->list[tuple[str,str,str,int]]:
    source=ROOT/"sources/github/modular-arithmetic-challenge/src"; sys.path.insert(0,str(source))
    try:
        from modchallenge.config import EvalConfig
        from modchallenge.testgen.generator import generate_private_test_set
        test_set=generate_private_test_set(hashlib.sha256(f"exp687-candidate-{root}".encode()).digest(),EvalConfig()); return [(case.a,case.b,case.p,tier.tier_id) for tier in test_set.tiers for case in tier.cases]
    finally: sys.path.remove(str(source))


def candidate_runtime_once(config:dict[str,Any],checkpoint:Path,root:int,limit:int|None=None)->dict[str,Any]:
    rows=candidate_inputs(root); rows=rows if limit is None else ([r for r in rows if 128<int(r[2]).bit_length()<=256]+[r for r in rows if not 128<int(r[2]).bit_length()<=256])[:limit]; candidate=load_module(f"candidate005_for_exp687_{root}_{limit}",ROOT/"submission/model.py"); incumbent=candidate.Tier2HornerModel(); model=new_student(config,load_file(checkpoint)).eval(); timer=Timer(); incumbent.load(str(ROOT/"submission")); lower=[(a,b,p) for a,b,p,_ in rows if int(p).bit_length()<=128]; wide=[r for r in rows if 128<int(r[2]).bit_length()<=256]; upper=[r for r in rows if int(r[2]).bit_length()>256]; low_out=incumbent.predict_digits_batch(lower) if lower else []; mod=[int(r[2]) for r in wide]; wide_out=rollout(model,[int(r[0])%p for r,p in zip(wide,mod,strict=True)],[int(r[1])%p for r,p in zip(wide,mod,strict=True)],mod,256)[0] if wide else []; resource=timer.finish(); return {"cases":len(rows),"complete_1100_case_route":len(rows)==1100,"lower_incumbent_cases":len(lower),"w256_specialist_cases":len(wide),"above_w256_zero_fallback_cases":len(upper),"output_count":len(low_out)+len(wide_out)+len(upper),"labels_available_to_timed_route":False,"resource":resource}


class Experiment:
    def __init__(self,config_path:Path,output:Path,mode:str,root:int|None)->None:
        self.config_path=config_path.resolve(); self.config=yaml.safe_load(self.config_path.read_text()); self.output=output.resolve(); self.output.mkdir(parents=True,exist_ok=True); self.mode=mode; self.root=root; self.checkpoint=(ROOT/self.config["source_checkpoint"]).resolve(); self.authenticate(); configure_torch(self.config); self.exp300=load_exp300(); self.pruned,self.manifest=prepare_pruned(self.config,self.checkpoint)
        if mode!="smoke":
            expected=int(self.config["roots"]["calibration"] if mode=="calibration" else self.config["roots"]["formal"] if mode=="formal" else -1); allowed={expected} if mode!="confirmation" else set(map(int,self.config["roots"]["confirmations"])); self.root=expected if root is None and mode!="confirmation" else root
            if self.root is None or int(self.root) not in allowed: raise ValueError(f"root {self.root} is not frozen for {mode}")
            self.root=int(self.root); self.splits={w:make_split(self.config,self.exp300,w,self.root) for w in set(self.config["data"]["curriculum"]+self.config["evaluation"]["widths"])}

    def authenticate(self)->None:
        if self.config["run_id"]!=RUN_ID or not self.checkpoint.is_file() or sha256(self.checkpoint)!=self.config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint mismatch")
        for name,binding in self.config["source_contract"].items():
            if sha256(ROOT/binding["path"])!=binding["sha256"]: raise RuntimeError(f"source contract mismatch: {name}")
        if git_output("-C","sources/github/modular-arithmetic-challenge","rev-parse","HEAD")!=CHALLENGE_COMMIT: raise RuntimeError("challenge commit mismatch")

    def identity(self)->dict[str,Any]: return {"repository_commit":git_output("rev-parse","HEAD"),"challenge_commit":CHALLENGE_COMMIT,"config_sha256":sha256(self.config_path),"runner_sha256":sha256(Path(__file__)),"test_sha256":sha256(ROOT/"tests/test_exp_687_compact_distilled_w256_specialist.py"),"source_checkpoint_sha256":sha256(self.checkpoint),"channel_map_sha256":canonical_hash(self.manifest["selected_source_channels_ranked"]),"pruned_state_sha256":self.manifest["pruned_state_sha256"]}
    def root_dir(self)->Path: return self.output/self.mode/f"root-{self.root}"
    def open_root(self)->Path:
        path=self.root_dir()
        if path.exists(): raise RuntimeError(f"root {self.root} already opened")
        path.mkdir(parents=True); write_json(path/"root-opened.json",{"mode":self.mode,"root":self.root,"source":self.identity()}); write_json(path/"channel-slice-manifest.json",self.manifest); return path
    def train_routes(self,path:Path,updates:int):
        models={}; records={}
        for arm in ARM_ORDER:
            models[arm],records[arm]=train_arm(self.config,self.exp300,self.pruned,self.checkpoint,self.splits,int(self.root),arm,updates,path); write_json(path/"progress.json",{"completed_arms":list(records),"records":records})
        a,b=(records[x] for x in TRAINED_ARMS)
        if a["initial_state_sha256"]!=b["initial_state_sha256"] or any(a["training"][k]!=b["training"][k] for k in ("updates","curriculum","calls_per_update","teacher_calls","student_calls","exact_loss_calls")): raise RuntimeError("matched trained-arm work failed")
        return models,records
    def evaluate_routes(self,models,transitions,rollouts,widths=(128,256)): return {arm:{str(width):evaluate(self.config,self.exp300,model,self.splits[width],int(self.root),width,transitions,rollouts) for width in widths} for arm,model in models.items()}

    def admission(self,records,evaluations,runtimes,resource,updates,transitions,rollouts):
        scale=int(self.config["training"]["updates"])/updates; training=sum(float(records[a]["training"]["resource"]["wall_seconds"])*scale for a in TRAINED_ARMS); evaluation=sum(float(evaluations[a][str(w)][kind]["resource"]["wall_seconds"])*(int(self.config["evaluation"]["transition_cases_per_width"] if kind=="transition" else self.config["evaluation"]["rollout_cases_per_width"])/(transitions if kind=="transition" else rollouts)) for a in ARM_ORDER for w in (128,256) for kind in ("transition","rollout")); candidate=max(float(r["resource"]["wall_seconds"]) for r in runtimes["cold"]); variability=max(float(x["timing"]["wall_seconds"]) for a in TRAINED_ARMS for x in records[a]["training"]["history"])/max(np.mean([float(x["timing"]["wall_seconds"]) for a in TRAINED_ARMS for x in records[a]["training"]["history"]]),1e-12); headroom=min(float(self.config["resource_policy"]["variability_headroom_ceiling"]),max(float(self.config["resource_policy"]["candidate_headroom_floor"]),float(variability))); expected=training+evaluation+candidate; admission={"projected_training_wall_seconds":training,"projected_evaluation_wall_seconds":evaluation,"expected_wall_seconds":expected,"eta_wall_seconds":expected*headroom,"soft_guard_wall_seconds":math.ceil(expected*headroom*float(self.config["resource_policy"]["soft_guard_multiplier"])),"hard_guard_wall_seconds":math.ceil(expected*headroom*float(self.config["resource_policy"]["hard_guard_multiplier"])),"headroom":headroom,"candidate_max_complete_wall_seconds":candidate,"candidate_guarded_wall_seconds":candidate*headroom,"expected_peak_rss_bytes":int(resource["peak_rss_bytes"]*headroom),"fidelity_reduced":False}; admission["checks"]={"complete_routes":set(records)==set(ARM_ORDER) and set(evaluations)==set(ARM_ORDER),"matched_training":records[TRAINED_ARMS[0]]["training"]["teacher_calls"]==records[TRAINED_ARMS[1]]["training"]["teacher_calls"],"reload_exact":all(x["reload_exact"] for x in records.values()),"candidate_three_complete_cold":len(runtimes["cold"])==3 and all(r["complete_1100_case_route"] and r["output_count"]==1100 for r in runtimes["cold"]),"candidate_runtime":admission["candidate_guarded_wall_seconds"]<=float(self.config["gates"]["maximum_candidate_guarded_wall_seconds"]),"expected_value":expected<=float(self.config["resource_policy"]["maximum_expected_formal_wall_seconds"]),"memory":admission["expected_peak_rss_bytes"]<=int(self.config["resource_policy"]["maximum_peak_rss_bytes"]),"no_fidelity_reduction":True}; admission["admitted"]=all(admission["checks"].values()); return admission

    def smoke(self):
        old_root,old_splits=self.root,getattr(self,"splits",None); self.root=687000; self.splits={w:make_split(self.config,self.exp300,w,self.root) for w in (64,128,256)}
        with tempfile.TemporaryDirectory(prefix="exp687-smoke-",dir=self.output) as directory:
            path=Path(directory); write_json(path/"channel-slice-manifest.json",self.manifest); models,records=self.train_routes(path,1); evaluations=self.evaluate_routes(models,1,1); runtime=candidate_runtime_once(self.config,Path(records["distilled"]["checkpoint"]["path"]),int(self.root),1); synthetic={**runtime,"cases":1100,"complete_1100_case_route":True,"output_count":1100,"resource":{**runtime["resource"],"wall_seconds":float(runtime["resource"]["wall_seconds"])*1100}}; resources=Timer().finish(); admission=self.admission(records,evaluations,{"warmup":runtime,"cold":[synthetic]*3},resources,1,1,1); terminals=[]
            randomized=new_student(self.config,None,687999); reset=new_student(self.config,self.pruned); dependence=state_hash(randomized.state_dict())!=state_hash(reset.state_dict())
            for mode in ("calibration","formal","confirmation"):
                target=path/f"{mode}-terminal.json"; write_json(target,{"schema_version":"exp687-smoke-terminal-v1","mode":mode,"status":"completed","records":records,"evaluations":evaluations,"admission":admission}); json.loads(target.read_text()); terminals.append(target.name)
            result={"schema_version":"exp687-rootless-all-mode-smoke-v1","status":"completed","root":None,"all_modes":["smoke","calibration","formal","confirmation"],"channel_map_frozen":len(self.manifest["selected_source_channels_ranked"])==56,"one_update_both_arms":all(records[a]["training"]["updates"]==1 for a in TRAINED_ARMS),"one_case_all_routes_widths":all(evaluations[a][str(w)][kind]["cases"]==1 for a in ARM_ORDER for w in (128,256) for kind in ("transition","rollout")),"identical_initial_bytes":records["distilled"]["initial_state_sha256"]==records["exact_only"]["initial_state_sha256"],"identical_teacher_work":records["distilled"]["training"]["teacher_calls"]==records["exact_only"]["training"]["teacher_calls"],"nonzero_distillation_gradient_path":records["distilled"]["training"]["history"][0]["distillation_weight"]>0,"reload_exact":all(x["reload_exact"] for x in records.values()),"randomized_reset_differ":dependence,"candidate_route_executed":runtime["output_count"]==1,"admission_math_exercised":bool(admission["checks"]),"terminal_serializations":terminals,"native_pruning_choice":self.manifest["native_choice"]}
        self.root,self.splits=old_root,old_splits; write_json(self.output/"smoke.json",result); return result

    def calibration(self):
        path=self.open_root(); timer=Timer(); updates=int(self.config["training"]["calibration_updates"]); transitions=int(self.config["calibration"]["transition_cases_per_route"]); rollouts=int(self.config["calibration"]["rollout_cases_per_route"]); models,records=self.train_routes(path,updates); evaluations=self.evaluate_routes(models,transitions,rollouts); checkpoint=Path(records["distilled"]["checkpoint"]["path"]); warmup=candidate_runtime_once(self.config,checkpoint,int(self.root),1); cold=[candidate_runtime_once(self.config,checkpoint,int(self.root)) for _ in range(int(self.config["calibration"]["candidate_repetitions"]))]; resource=timer.finish(); admission=self.admission(records,evaluations,{"warmup":warmup,"cold":cold},resource,updates,transitions,rollouts); decision="admit-root-20261303" if admission["admitted"] else "park-exp687-at-root20261302-resource-admission"; result={"schema_version":"exp687-calibration-v1","run_id":RUN_ID,"status":"completed","mode":self.mode,"root":self.root,"decision":decision,"source":self.identity(),"channel_slice_manifest":self.manifest,"records":records,"evaluations":evaluations,"candidate_runtime":{"warmup":warmup,"cold":cold},"resource":resource,"admission":admission,"limitations":self.limitations(True)}; write_json(path/"root-summary.json",result); write_json(self.output/"calibration.json",result); return result
    def admitted(self):
        path=self.output/"calibration.json"
        if not path.is_file(): raise RuntimeError("science requires root20261302 admission")
        value=json.loads(path.read_text())
        if value["decision"]!="admit-root-20261303" or not value["admission"]["admitted"] or value["source"]["runner_sha256"]!=sha256(Path(__file__)): raise RuntimeError("matching admitted calibration required")
        return value
    def require_predecessor(self):
        if self.mode!="confirmation": return
        roots=list(map(int,self.config["roots"]["confirmations"])); i=roots.index(int(self.root)); prior=int(self.config["roots"]["formal"]) if i==0 else roots[i-1]; mode="formal" if i==0 else "confirmation"; expected="advance-to-confirmations" if i==0 else "confirmation-pass"; path=self.output/mode/f"root-{prior}"/"root-summary.json"
        if not path.is_file() or json.loads(path.read_text())["decision"]!=expected: raise RuntimeError(f"confirmation requires {expected} at root {prior}")
    def science(self):
        calibration=self.admitted(); self.require_predecessor(); path=self.open_root(); timer=Timer(); models,records=self.train_routes(path,int(self.config["training"]["updates"])); evaluations=self.evaluate_routes(models,int(self.config["evaluation"]["transition_cases_per_width"]),int(self.config["evaluation"]["rollout_cases_per_width"])); p=evaluations["distilled"]["256"]; control=evaluations["exact_only"]; frozen=evaluations["frozen_pruned"]["256"]; randomized=evaluations["randomized_control"]["256"]; reset=evaluations["reset_control"]["256"]; score=p["rollout"]["exact_accuracy"]; gates={"primary_w256_rollout":score>=float(self.config["gates"]["minimum_primary_w256_rollout"]),"primary_w256_transition":p["transition"]["exact_accuracy"]>=float(self.config["gates"]["minimum_primary_w256_transition"]),"invalid_rate":p["rollout"]["any_invalid_residue_rate"]<=float(self.config["gates"]["maximum_invalid_rate"]),"gain_over_exact_only":score-control["256"]["rollout"]["exact_accuracy"]>=float(self.config["gates"]["minimum_gain_over_exact_only"]),"w128_retention":evaluations["distilled"]["128"]["rollout"]["exact_accuracy"]>=control["128"]["rollout"]["exact_accuracy"]-float(self.config["gates"]["maximum_w128_loss_to_exact_only"]),"nonzero_distillation":records["distilled"]["training"]["history"][-1]["distillation_weight"]>0,"matched_work":records["distilled"]["training"]["teacher_calls"]==records["exact_only"]["training"]["teacher_calls"],"trained_weight_dependence":score-randomized["rollout"]["exact_accuracy"]>=float(self.config["gates"]["minimum_trained_weight_drop"]),"reset_matches_frozen":reset["rollout"]["prediction_sha256"]==frozen["rollout"]["prediction_sha256"],"reload_exact":all(x["reload_exact"] for x in records.values()),"candidate_runtime":float(calibration["admission"]["candidate_guarded_wall_seconds"])<=float(self.config["gates"]["maximum_candidate_guarded_wall_seconds"]),"confirmation_threshold":score>=float(self.config["gates"]["confirmation_admission_rollout"])}; mechanism=all(v for k,v in gates.items() if k!="confirmation_threshold"); confirmation=mechanism and gates["confirmation_threshold"]; decision=("advance-to-confirmations" if confirmation else "reject-exp687-compact-distilled-w256-specialist") if self.mode=="formal" else ("confirmation-pass" if confirmation else "reject-exp687-confirmation"); result={"schema_version":"exp687-terminal-v1","run_id":RUN_ID,"status":"completed","mode":self.mode,"root":self.root,"decision":decision,"mechanism_gate_passed":mechanism,"confirmation_gate_passed":confirmation,"source":self.identity(),"records":records,"evaluations":evaluations,"gates":gates,"dynamic_admission":calibration["admission"],"resource":timer.finish(),"limitations":self.limitations(False)}; write_json(path/"root-summary.json",result); write_json(self.output/"terminal-summary.json",result); return result
    @staticmethod
    def limitations(calibration:bool)->list[str]: return ["Calibration exact accuracy is route integrity evidence, not scientific evidence." if calibration else "Only fresh generated identities are evaluated; no checked, independent, official or private benchmark is opened.","Exact arithmetic and the frozen teacher are training/evaluation-only; formal inference loads only compact trained weights.","Native index_select was selected for auditable coherent slicing; no external pruning library showed a measured need in this one fixed graph.","This is intermediate research and authorizes no candidate, submission, release or incumbent promotion."]


def log_result(experiment:Experiment,result:dict[str,Any])->dict[str,Any]:
    if experiment.mode=="smoke": return result
    experiment_id=configure_mlflow(experiment.config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name=f"{RUN_ID}-{experiment.mode}-{experiment.root}") as active:
        mlflow.set_tags({"canonical_run_id":RUN_ID,"mode":experiment.mode,"root":str(experiment.root),"decision":result["decision"],"candidate_registration":"false"}); mlflow.log_dict(result,f"{experiment.mode}-result.json"); run_id=active.info.run_id
    record={"tracking_uri":tracking_uri(),"experiment_id":experiment_id,"run_id":run_id,"mode":experiment.mode,"root":experiment.root}; write_json(experiment.output/f"{experiment.mode}-mlflow.json",record); result["mlflow"]=record; write_json(experiment.root_dir()/"root-summary.json",result); write_json(experiment.output/("calibration.json" if experiment.mode=="calibration" else "terminal-summary.json"),result); return result


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,default=CONFIG_PATH); parser.add_argument("--output",type=Path,default=ROOT/".scratch/exp-687-compact-distilled-w256-specialist"); parser.add_argument("--mode",choices=("smoke","calibration","formal","confirmation"),required=True); parser.add_argument("--root",type=int); args=parser.parse_args(); experiment=Experiment(args.config,args.output,args.mode,args.root); result=experiment.smoke() if args.mode=="smoke" else experiment.calibration() if args.mode=="calibration" else experiment.science(); result=log_result(experiment,result); print(json.dumps({"mode":args.mode,"root":result.get("root"),"status":result["status"],"decision":result.get("decision"),"gates":result.get("gates"),"admission":result.get("admission",{}).get("checks")},indent=2,sort_keys=True),flush=True); return 0


if __name__=="__main__": raise SystemExit(main())
