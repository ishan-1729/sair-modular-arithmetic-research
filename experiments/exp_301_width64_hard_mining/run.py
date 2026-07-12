"""Hard-mined continuation of the transferred width-64 bit-tensor model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import torch
import yaml
from safetensors.torch import load_file, save_file
from torch import nn

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def git_output(*args): return subprocess.run(["git",*args],cwd=ROOT,check=True,capture_output=True,text=True,timeout=15).stdout.strip()


def mine(values,size,config,generator):
    state,multiplicand,modulus,input_bit,target,quotient=values; available=torch.ones(len(state),dtype=torch.bool); raw=[2*s+b*m for s,b,m in zip(state,input_bit,multiplicand)]; boundary=torch.tensor([min(abs(value-p),abs(value-2*p))/p <= config["training"]["boundary_relative_margin"] for value,p in zip(raw,modulus)]); selected=[]
    def draw(mask,count):
        indices=torch.nonzero(mask & available,as_tuple=False).flatten(); order=torch.randperm(len(indices),generator=generator); chosen=indices[order[:min(count,len(indices))]]; available[chosen]=False; return chosen
    selected.append(draw(quotient.eq(2),round(size*config["training"]["quotient2_fraction"]))); selected.append(draw(boundary,round(size*config["training"]["boundary_fraction"]))); selected.append(draw(available,size-sum(len(x) for x in selected))); indices=torch.cat(selected); indices=indices[torch.randperm(len(indices),generator=generator)]; return ([state[i] for i in indices],[multiplicand[i] for i in indices],[modulus[i] for i in indices],[input_bit[i] for i in indices],[target[i] for i in indices],quotient[indices])


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,default=ROOT/"configs/exp_301_width64_hard_mining.yaml"); parser.add_argument("--output",type=Path,default=ROOT/".scratch/exp-301-width64-hard-mining"); args=parser.parse_args(); config_path=args.config.resolve(); config=yaml.safe_load(config_path.read_text()); source=(ROOT/config["source_checkpoint"]).resolve()
    if sha256(source)!=config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
    exp300=load_module("exp300_for_exp301",ROOT/"experiments/exp_300_width64_bit_tensor_transfer/run.py"); split=exp300.bigint_prime_split(config,config["seed"]+301); torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); experiment_id=configure_mlflow(config["tracking"]["experiment_name"]); model=exp300.exp100_model.SharedContextHornerTCN(**config["model"]); model.load_state_dict(load_file(source)); model.eval(); source_transition=exp300.evaluate_transitions(model,exp300.BigIntSampler(split["test"],config["seed"]+3150),config["evaluation"]["transition_cases"]); optimizer=torch.optim.AdamW(model.parameters(),lr=config["training"]["learning_rate"],weight_decay=config["training"]["weight_decay"]); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config["training"]["steps"],eta_min=config["training"]["minimum_learning_rate"]); sampler=exp300.BigIntSampler(split["train"],config["seed"]+3101); selector=torch.Generator().manual_seed(config["seed"]+3102); started=time.perf_counter(); history=[]; pool_size=config["training"]["batch_size"]*config["training"]["candidate_pool_factor"]
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({"canonical_run_id":config["run_id"],"status":"running","role":"width64-hard-mined-continuation","repository_commit":git_output("rev-parse","HEAD"),"repository_dirty":str(bool(git_output("status","--porcelain"))).lower()}); mlflow.log_params({"seed":config["seed"],"steps":config["training"]["steps"],"source_checkpoint_sha256":sha256(source),"config_sha256":sha256(config_path),"runner_sha256":sha256(Path(__file__))}); mlflow.log_artifact(str(config_path),artifact_path="preregistration"); mlflow.log_artifact(str(ROOT/"knowledge/autoresearch/ablations/exp-301-width64-hard-mining.md"),artifact_path="preregistration"); model.train()
        for step in range(1,config["training"]["steps"]+1):
            state,multiplicand,modulus,input_bit,target,quotient=mine(sampler.transitions(pool_size),config["training"]["batch_size"],config,selector); logits,quotient_logits=model(exp300.features(state,multiplicand,modulus,input_bit,model.width)); loss,stats=exp300.transition_loss(logits,quotient_logits,target,quotient,model.width,config["training"]["worst_bit_weight"],config["training"]["quotient_weight"]); optimizer.zero_grad(set_to_none=True); loss.backward(); grad=float(nn.utils.clip_grad_norm_(model.parameters(),config["training"]["gradient_clip"])); optimizer.step(); scheduler.step()
            if step==1 or step%100==0 or step==config["training"]["steps"]:
                record={"step":step,"elapsed_seconds":time.perf_counter()-started,"gradient_norm":grad,**stats}; history.append(record); mlflow.log_metrics({"train_loss":stats["loss"],"train_batch_exact":stats["batch_exact"],"gradient_norm":grad},step=step); print(json.dumps(record),flush=True)
        model.eval(); transition=exp300.evaluate_transitions(model,exp300.BigIntSampler(split["test"],config["seed"]+3150),config["evaluation"]["transition_cases"]); rollout=exp300.evaluate_rollouts(model,exp300.BigIntSampler(split["test"],config["seed"]+3151),config["evaluation"]["rollout_cases"]); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True); checkpoint=output/"weights.safetensors"; save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},checkpoint); gain=transition["exact_accuracy"]-source_transition["exact_accuracy"]; gates=config["gates"]; passed=gain>=gates["minimum_transition_gain"] and transition["exact_accuracy"]>=gates["minimum_transition"] and rollout["exact_accuracy"]>=gates["minimum_rollout"] and rollout["any_invalid_residue_rate"]<=gates["maximum_invalid_rollout_rate"]; decision="advance-width64-2400" if passed else "reject-width64-hard-mined-tcn"; summary={"run_id":config["run_id"],"status":"completed","mlflow":{"tracking_uri":tracking_uri(),"experiment_name":config["tracking"]["experiment_name"],"experiment_id":experiment_id,"run_id":run.info.run_id},"source":{"repository_commit":git_output("rev-parse","HEAD"),"dirty":bool(git_output("status","--porcelain")),"challenge_commit":"82510bba00a1126649bd76dd1a451f14d0b3eb60","config_sha256":sha256(config_path),"runner_sha256":sha256(Path(__file__)),"source_checkpoint_sha256":sha256(source)},"source_transition":source_transition,"trained":{"transition":transition,"rollout":rollout,"training":{"steps":config["training"]["steps"],"history":history},"checkpoint":{"path":str(checkpoint.relative_to(ROOT)),"sha256":sha256(checkpoint),"bytes":checkpoint.stat().st_size}},"comparison":{"transition_gain":gain},"gates":{**gates,"passed":passed},"decision":decision,"resource":{"wall_seconds":time.perf_counter()-started,"cpu_minute_cap":config["resource_cap"]["cpu_minutes"]},"limitations":["Python bigint arithmetic is confined to offline data, labels, mining, and evaluation.","No checked or consumed independent answer was read."]}; summary_path=output/"summary.json"; summary_path.write_text(json.dumps(summary,indent=2)+"\n"); mlflow.log_metrics({"source_transition":source_transition["exact_accuracy"],"trained_transition":transition["exact_accuracy"],"trained_rollout":rollout["exact_accuracy"],"transition_gain":gain,"wall_seconds":summary["resource"]["wall_seconds"]}); mlflow.log_artifact(str(summary_path),artifact_path="canonical-preview"); mlflow.set_tags({"status":"completed","decision":decision})
    print(json.dumps({"decision":decision,"source_transition":source_transition,"transition":transition,"rollout":rollout,"gain":gain,"checkpoint":summary["trained"]["checkpoint"],"mlflow":summary["mlflow"]},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
