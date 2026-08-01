"""Matched random versus transferred width-64 TCN using offline Python-bigint labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file
from sympy import nextprime
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
EXP100_DIR = ROOT / "experiments/exp_100_width_scaling"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(EXP100_DIR))
import model as exp100_model  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str) -> str: return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def bigint_prime_split(config: dict, seed: int) -> dict[str, list[int]]:
    rng = random.Random(seed); low = 1 << (config["data"]["prime_bits_min"] - 1); high = 1 << config["data"]["prime_bits_max"]; total = config["data"]["train_prime_identities"] + config["data"]["test_prime_identities"]; values = []; seen = set()
    while len(values) < total:
        prime = int(nextprime(rng.randrange(low, high)))
        if prime < high and prime not in seen: seen.add(prime); values.append(prime)
    count = config["data"]["train_prime_identities"]
    return {"train": values[:count], "test": values[count:]}


def bits(values: list[int], width: int) -> torch.Tensor:
    return torch.tensor([[(value >> position) & 1 for position in range(width)] for value in values], dtype=torch.float32)


def features(state: list[int], multiplicand: list[int], modulus: list[int], bit: list[int], width: int) -> torch.Tensor:
    batch = len(state); bit_plane = torch.tensor(bit, dtype=torch.float32)[:, None].expand(batch, width); lsb = torch.zeros(batch, width); msb = torch.zeros(batch, width); lsb[:, 0] = 1.0; msb[:, -1] = 1.0
    return torch.stack((bits(state, width), bits(multiplicand, width), bits(modulus, width), bit_plane, lsb, msb), dim=1).mul(2.0).sub(1.0)


class BigIntSampler:
    def __init__(self, primes: list[int], seed: int): self.primes = primes; self.rng = random.Random(seed)
    def transitions(self, size: int):
        states=[]; multiplicands=[]; moduli=[]; input_bits=[]; targets=[]; quotients=[]
        for _ in range(size):
            modulus=self.rng.choice(self.primes); state=self.rng.randrange(modulus); multiplicand=self.rng.randrange(modulus); bit=self.rng.randrange(2); raw=2*state+bit*multiplicand; states.append(state); multiplicands.append(multiplicand); moduli.append(modulus); input_bits.append(bit); targets.append(raw % modulus); quotients.append(raw // modulus)
        return states,multiplicands,moduli,input_bits,targets,torch.tensor(quotients,dtype=torch.long)
    def pairs(self, size: int):
        moduli=[self.rng.choice(self.primes) for _ in range(size)]; return [self.rng.randrange(p) for p in moduli],[self.rng.randrange(p) for p in moduli],moduli


def transition_loss(logits, quotient_logits, targets: list[int], quotients, width: int, worst_weight: float, quotient_weight: float):
    target = bits(targets, width); per_bit = F.binary_cross_entropy_with_logits(logits, target, reduction="none"); mean = per_bit.mean(); worst = per_bit.mean(0).max(); quotient = F.cross_entropy(quotient_logits, quotients); loss = mean + worst_weight * worst + quotient_weight * quotient; exact = (logits.ge(0) == target.bool()).all(1).float().mean(); return loss,{"loss":float(loss.detach()),"batch_exact":float(exact),"mean_bce":float(mean.detach()),"worst_bit_bce":float(worst.detach()),"quotient_ce":float(quotient.detach())}


@torch.inference_mode()
def evaluate_transitions(model, sampler, cases: int) -> dict:
    seen=correct=valid=0
    while seen < cases:
        size=min(512,cases-seen); state,multiplicand,modulus,input_bit,target,_q=sampler.transitions(size); logits,_=model(features(state,multiplicand,modulus,input_bit,model.width)); predicted=logits.ge(0); target_tensor=bits(target,model.width).bool(); correct += int((predicted == target_tensor).all(1).sum()); predicted_values=[sum(int(row[position]) << position for position in range(model.width)) for row in predicted.tolist()]; valid += sum(value < p for value,p in zip(predicted_values,modulus)); seen += size
    return {"cases":seen,"correct":correct,"exact_accuracy":correct/seen,"valid_residue_rate":valid/seen}


@torch.inference_mode()
def evaluate_rollouts(model, sampler, cases: int) -> dict:
    width=model.width; seen=correct=all_steps=invalid=0
    while seen < cases:
        size=min(128,cases-seen); multiplier,multiplicand,modulus=sampler.pairs(size); state=[0]*size; teacher=[0]*size; every=[True]*size; bad=[False]*size
        for position in range(width):
            input_bit=[(value >> (width-1-position)) & 1 for value in multiplier]; logits,_=model(features(state,multiplicand,modulus,input_bit,width)); rows=logits.ge(0).tolist(); state=[sum(int(row[index]) << index for index in range(width)) for row in rows]; teacher=[(2*value+bit*other)%p for value,bit,other,p in zip(teacher,input_bit,multiplicand,modulus)]; every=[prior and predicted==expected for prior,predicted,expected in zip(every,state,teacher)]; bad=[prior or predicted>=p for prior,predicted,p in zip(bad,state,modulus)]
        expected=[(a*b)%p for a,b,p in zip(multiplier,multiplicand,modulus)]; correct += sum(a==b for a,b in zip(state,expected)); all_steps += sum(every); invalid += sum(bad); seen += size
    return {"cases":seen,"correct":correct,"exact_accuracy":correct/seen,"all_steps_exact_accuracy":all_steps/seen,"any_invalid_residue_rate":invalid/seen}


def train_arm(name: str, transferred: bool, config: dict, split: dict, source: Path, exp105, output: Path) -> dict:
    random.seed(config["seed"]); torch.manual_seed(config["seed"]); model=exp100_model.SharedContextHornerTCN(**config["model"]); transfer=None
    if transferred: transfer=exp105.transfer_width9_to_width16(model,source)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["training"]["learning_rate"],weight_decay=config["training"]["weight_decay"]); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config["training"]["steps"],eta_min=config["training"]["minimum_learning_rate"]); sampler=BigIntSampler(split["train"],config["seed"]+3001); started=time.perf_counter(); history=[]; model.train()
    for step in range(1,config["training"]["steps"]+1):
        state,multiplicand,modulus,input_bit,target,quotient=sampler.transitions(config["training"]["batch_size"]); logits,quotient_logits=model(features(state,multiplicand,modulus,input_bit,model.width)); loss,stats=transition_loss(logits,quotient_logits,target,quotient,model.width,config["training"]["worst_bit_weight"],config["training"]["quotient_weight"]); optimizer.zero_grad(set_to_none=True); loss.backward(); grad=float(nn.utils.clip_grad_norm_(model.parameters(),config["training"]["gradient_clip"])); optimizer.step(); scheduler.step()
        if step==1 or step%50==0 or step==config["training"]["steps"]:
            record={"step":step,"elapsed_seconds":time.perf_counter()-started,"gradient_norm":grad,**stats}; history.append(record); mlflow.log_metrics({"train_loss":stats["loss"],"train_batch_exact":stats["batch_exact"],"gradient_norm":grad},step=step); print(json.dumps({"arm":name,**record}),flush=True)
    training_seconds=time.perf_counter()-started; model.eval(); transition=evaluate_transitions(model,BigIntSampler(split["test"],config["seed"]+3050),config["evaluation"]["transition_cases"]); rollout=evaluate_rollouts(model,BigIntSampler(split["test"],config["seed"]+3051),config["evaluation"]["rollout_cases"]); output.mkdir(parents=True,exist_ok=True); checkpoint=output/"weights.safetensors"; save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},checkpoint); return {"arm":name,"transferred":transferred,"transfer":transfer,"parameters":model.parameter_count(),"training":{"steps":config["training"]["steps"],"seconds":training_seconds,"history":history},"evaluation":{"transition":transition,"rollout":rollout},"checkpoint":{"path":str(checkpoint.relative_to(ROOT)),"sha256":sha256(checkpoint),"bytes":checkpoint.stat().st_size}}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,default=ROOT/"configs/exp_300_width64_bit_tensor_transfer.yaml"); parser.add_argument("--output",type=Path,default=ROOT/".scratch/exp-300-width64-bit-tensor-transfer"); args=parser.parse_args(); config_path=args.config.resolve(); config=yaml.safe_load(config_path.read_text()); source=(ROOT/config["source_checkpoint"]).resolve()
    if sha256(source)!=config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
    exp105=load_module("exp105_for_exp300",ROOT/"experiments/exp_105_width_transfer/run.py"); split=bigint_prime_split(config,config["seed"]+300); torch.set_num_threads(config["training"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); experiment_id=configure_mlflow(config["tracking"]["experiment_name"]); output=args.output.resolve(); started=time.perf_counter(); repository_commit=git_output("rev-parse","HEAD"); results={}
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        parent_id=parent.info.run_id; mlflow.set_tags({"canonical_run_id":config["run_id"],"status":"running","role":"width64-bit-tensor-transfer-ab","repository_commit":repository_commit,"repository_dirty":str(bool(git_output("status","--porcelain"))).lower()}); mlflow.log_params({"seed":config["seed"],"steps":config["training"]["steps"],"source_checkpoint_sha256":sha256(source),"config_sha256":sha256(config_path),"runner_sha256":sha256(Path(__file__))}); mlflow.log_artifact(str(config_path),artifact_path="preregistration"); mlflow.log_artifact(str(ROOT/"knowledge/autoresearch/ablations/exp-300-width64-bit-tensor-transfer.md"),artifact_path="preregistration"); run_ids={}
        for name,transferred in (("width64-random",False),("width64-transfer",True)):
            with mlflow.start_run(run_name=name,nested=True) as nested:
                run_ids[name]=nested.info.run_id; results[name]=train_arm(name,transferred,config,split,source,exp105,output/name); mlflow.log_metrics({"transition_exact":results[name]["evaluation"]["transition"]["exact_accuracy"],"rollout_exact":results[name]["evaluation"]["rollout"]["exact_accuracy"],"invalid_rollout":results[name]["evaluation"]["rollout"]["any_invalid_residue_rate"]})
        control=results["width64-random"]; challenger=results["width64-transfer"]; comparison={"transition_gain":challenger["evaluation"]["transition"]["exact_accuracy"]-control["evaluation"]["transition"]["exact_accuracy"],"rollout_gain":challenger["evaluation"]["rollout"]["exact_accuracy"]-control["evaluation"]["rollout"]["exact_accuracy"]}; gates=config["gates"]; passed=comparison["transition_gain"]>=gates["minimum_transfer_transition_gain"] and comparison["rollout_gain"]>=gates["minimum_transfer_rollout_gain"] and challenger["evaluation"]["transition"]["exact_accuracy"]>=gates["minimum_transfer_transition"] and challenger["evaluation"]["rollout"]["any_invalid_residue_rate"]<=gates["maximum_invalid_rollout_rate"]; decision="advance-width64-full-budget" if passed else "reject-width64-transfer-smoke"; summary={"run_id":config["run_id"],"status":"completed","mlflow":{"tracking_uri":tracking_uri(),"experiment_name":config["tracking"]["experiment_name"],"experiment_id":experiment_id,"parent_run_id":parent_id,"arm_run_ids":run_ids},"source":{"repository_commit":repository_commit,"dirty":bool(git_output("status","--porcelain")),"challenge_commit":"82510bba00a1126649bd76dd1a451f14d0b3eb60","config_sha256":sha256(config_path),"runner_sha256":sha256(Path(__file__)),"source_checkpoint_sha256":sha256(source)},"arms":results,"comparison":comparison,"gates":{**gates,"passed":passed},"decision":decision,"resource":{"wall_seconds":time.perf_counter()-started,"cpu_minute_cap":config["resource_cap"]["cpu_minutes"]},"limitations":["Python bigint arithmetic is confined to offline sampling, labels, and evaluation.","The learned model receives only bit tensors and produces bit logits.","Fresh generated identities only; no checked or consumed independent answer was read."]}; summary_path=output/"summary.json"; summary_path.write_text(json.dumps(summary,indent=2)+"\n"); mlflow.log_metrics({**comparison,"wall_seconds":summary["resource"]["wall_seconds"]}); mlflow.log_artifact(str(summary_path),artifact_path="canonical-preview"); mlflow.set_tags({"status":"completed","decision":decision})
    print(json.dumps({"decision":decision,"comparison":comparison,"arms":{k:{"transition":v["evaluation"]["transition"],"rollout":v["evaluation"]["rollout"],"checkpoint":v["checkpoint"],"parameters":v["parameters"]} for k,v in results.items()},"mlflow":summary["mlflow"]},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
