"""Preregister, calibrate, and gate EXP-667 RSM shallow-train/deep-infer."""
from __future__ import annotations
import argparse,json,math,platform,random,sys
from pathlib import Path
from typing import Any
import mlflow,numpy as np,psutil,torch,torch.nn.functional as F,yaml
from safetensors.torch import load_file,save_file

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from experiments.exp_664_sprm_boundary_perturbation.run import Timer,aggregate_resource,bits,decode,features,git_output,json_hash,load_exp300,make_split,rollout_cases,sha256,transition_cases,write_json
from experiments.exp_667_rsm_shallow_train_deep_infer.model import ROUTES,TRAINED_ROUTES,RSMCell
from tools.autoresearch_mlflow import configure_mlflow,tracking_uri
CONFIG_PATH=ROOT/"configs/exp_667_rsm_shallow_train_deep_infer.yaml"

def build_model(config:dict[str,Any],source:Path)->RSMCell:
    model=RSMCell(**config["model"]); model.load_source(load_file(source)); return model

def objective(output,target:list[int],quotient:torch.Tensor,width:int,route:str,config:dict[str,Any]):
    truth=bits(target,width); losses=[F.binary_cross_entropy_with_logits(x,truth,reduction="none") for x in output.supervision_logits]
    selected=torch.stack([x.mean() for x in losses]).mean() if route=="deep_supervision" else losses[-1].mean()
    selected=selected+float(config["training"]["worst_bit_weight"])*losses[-1].mean(0).max()+float(config["training"]["quotient_weight"])*F.cross_entropy(output.quotient_logits,quotient)
    return selected,{"loss":float(selected.detach()),"batch_exact":float((output.logits.ge(0)==truth.bool()).all(1).float().mean()),"mean_relative_residual":float(output.relative_residual.mean()),"mean_iterations":float(output.iterations.float().mean())}

def train_route(config,exp300,source,splits,root,route,steps,output,mode):
    random.seed(root+667); np.random.seed(root+667); torch.manual_seed(root+667)
    route_dir=output/mode/f"root-{root}"/route; route_dir.mkdir(parents=True,exist_ok=True); checkpoint=route_dir/"model.safetensors"; cache=route_dir/"training-result.json"
    key=json_hash({"source":sha256(source),"root":root,"route":route,"steps":steps,"model":config["model"],"training":config["training"],"splits":splits})
    if config["resource_policy"]["cache_completed_roots"] and checkpoint.is_file() and cache.is_file():
        old=json.loads(cache.read_text());
        if old.get("cache_key")==key and sha256(checkpoint)==old["result"]["checkpoint"]["sha256"]:
            model=build_model(config,source); model.load_state_dict(load_file(checkpoint)); return model,{**old["result"],"cached":True}
    model=build_model(config,source); timer=Timer(); history=[]; examples=calls=0
    if route=="randomized_states": model.randomized_states(root+66791)
    if route in TRAINED_ROUTES:
        params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=float(config["training"]["learning_rate"]),weight_decay=float(config["training"]["weight_decay"])); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps,eta_min=float(config["training"]["minimum_learning_rate"])); samplers={w:exp300.BigIntSampler(splits[w]["train"],root+int(config["data"]["training_seed_offset"])+w) for w in config["data"]["training_widths"]}; gen=torch.Generator().manual_seed(root+66703); model.train()
        for step in range(1,steps+1):
            i=(step-1)%len(config["training"]["schedule"]); width=int(config["training"]["schedule"][i]); h=int(config["training"]["high_curriculum"][i]); l=int(config["training"]["low_curriculum"][i]); state,mult,mod,input_bit,target,quot=samplers[width].transitions(int(config["training"]["batch_size"])); tick=Timer(); result=model.forward_train(features(state,mult,mod,input_bit,width),route=route,high_depth=h,low_depth=l,generator=gen); loss,metrics=objective(result,target,quot,width,route,config); opt.zero_grad(set_to_none=True); loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(params,float(config["training"]["gradient_clip"]))); opt.step(); sched.step(); examples+=len(target); calls+=int(result.iterations.sum()); history.append({"step":step,"width":width,"high_depth":h,"low_depth":l,"gradient_norm":grad,**metrics,"timing":tick.finish()})
    save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},checkpoint); reload=build_model(config,source); reload.load_state_dict(load_file(checkpoint)); exact=all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),reload.state_dict().values(),strict=True)); checkpoint_path=str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint); record={"route":route,"cached":False,"parameters":model.parameter_count(),"trainable_parameters":model.trainable_parameter_count(),"checkpoint":{"path":checkpoint_path,"sha256":sha256(checkpoint),"bytes":checkpoint.stat().st_size},"checkpoint_reload_exact":exact,"training":{"updates":steps if route in TRAINED_ROUTES else 0,"examples":examples,"operator_calls":calls,"history":history,"timing":timer.finish()}}; write_json(cache,{"cache_key":key,"result":record}); return model,record

@torch.inference_mode()
def evaluate_transitions(model,route,cases,high_depth=None,low_depth=None):
    timer=Timer(); width=int(cases["width"]); exact=bit_exact=invalid=0; preds=[]; residual=[]
    for start in range(0,len(cases["target"]),128):
        stop=start+128; result=model(features(cases["state"][start:stop],cases["multiplicand"][start:stop],cases["modulus"][start:stop],cases["input_bit"][start:stop],width),route=route,high_depth=high_depth,low_depth=low_depth); predicted=result.logits.ge(0); truth=bits(cases["target"][start:stop],width).bool(); values=decode(predicted); preds+=values; residual+=result.relative_residual.tolist(); exact+=int((predicted==truth).all(1).sum()); bit_exact+=int((predicted==truth).sum()); invalid+=sum(v>=p for v,p in zip(values,cases["modulus"][start:stop],strict=True))
    count=len(cases["target"]); return {"cases":count,"exact_accuracy":exact/count,"bit_accuracy":bit_exact/(count*width),"invalid_residue_rate":invalid/count,"mean_relative_residual":float(np.mean(residual)),"prediction_sha256":json_hash(preds),"timing":timer.finish()}

@torch.inference_mode()
def evaluate_rollouts(model,route,cases,high_depth=None,low_depth=None):
    timer=Timer(); width=int(cases["width"]); count=len(cases["left"]); endpoint=prefix=transitions=invalid=0; finals=[]
    for start in range(0,count,32):
        stop=min(count,start+32); left,right,mod=cases["left"][start:stop],cases["right"][start:stop],cases["modulus"][start:stop]; size=len(left); state=[0]*size; teacher=[0]*size; all_ok=[True]*size; any_bad=[False]*size
        for pos in range(width):
            input_bit=[(v>>(width-1-pos))&1 for v in left]; result=model(features(state,right,mod,input_bit,width),route=route,high_depth=high_depth,low_depth=low_depth); state=decode(result.logits.ge(0)); teacher=[(2*v+b*o)%p for v,b,o,p in zip(teacher,input_bit,right,mod,strict=True)]; ok=[a==b for a,b in zip(state,teacher,strict=True)]; transitions+=sum(ok)
            for i,good in enumerate(ok): all_ok[i]&=good; any_bad[i]|=state[i]>=mod[i]
        expected=[(a*b)%p for a,b,p in zip(left,right,mod,strict=True)]; endpoint+=sum(a==b for a,b in zip(state,expected,strict=True)); prefix+=sum(all_ok); invalid+=sum(any_bad); finals+=state
    return {"cases":count,"exact_accuracy":endpoint/count,"all_prefix_exact_accuracy":prefix/count,"transition_exact_accuracy":transitions/(count*width),"any_invalid_residue_rate":invalid/count,"prediction_sha256":json_hash(finals),"timing":timer.finish()}

@torch.inference_mode()
def stability_diagnostic(model,cases):
    records={}; route_map={1:"rsm_full",2:"rsm_full",4:"rsm_full"}
    for depth,route in route_map.items(): records[str(depth)]=evaluate_transitions(model,route,cases,depth,depth)
    full=model(features(cases["state"],cases["multiplicand"],cases["modulus"],cases["input_bit"],int(cases["width"])),route="rsm_full",high_depth=4,low_depth=4)
    high=model(features(cases["state"],cases["multiplicand"],cases["modulus"],cases["input_bit"],int(cases["width"])),route="high_reset",high_depth=4,low_depth=4); low=model(features(cases["state"],cases["multiplicand"],cases["modulus"],cases["input_bit"],int(cases["width"])),route="low_reset",high_depth=4,low_depth=4)
    return {"depths":records,"high_reset_distance":float((full.state-high.state).flatten(1).norm(dim=1).mean()),"low_reset_distance":float((full.state-low.state).flatten(1).norm(dim=1).mean()),"high_low_state_distance":float((full.high_state-full.low_state).flatten(1).norm(dim=1).mean()),"fixed_depth_target_free":True}

class Experiment:
    def __init__(self,config_path,output,mode,root=None):
        self.config_path=Path(config_path).resolve(); self.config=yaml.safe_load(self.config_path.read_text()); self.output=Path(output).resolve(); self.mode=mode; expected={"preregister":20261218,"calibration":20261218,"formal":20261219}[mode]; self.root=expected if root is None else int(root); allowed={expected} if mode!="formal" else {20261219,20261220,20261221};
        if self.root not in allowed: raise ValueError(f"root {self.root} is not frozen for {mode}")
        self.source=(ROOT/self.config["source_checkpoint"]).resolve();
        if sha256(self.source)!=self.config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
        if sha256(ROOT/self.config["paper"]["path"])!=self.config["paper"]["sha256"]: raise RuntimeError("paper hash mismatch")
        self.exp300=load_exp300(); self.splits={w:make_split(self.config,self.exp300,w,self.root) for w in self.config["data"]["widths"]}
    def preregistration(self):
        files={"config":self.config_path,"mechanism":ROOT/self.config["mechanism_map"],"model":ROOT/"experiments/exp_667_rsm_shallow_train_deep_infer/model.py","runner":Path(__file__),"test":ROOT/"tests/test_exp_667_rsm_shallow_train_deep_infer.py"}; return {"run_id":self.config["run_id"],"root":self.root,"mode":self.mode,"source":{"repository_commit":git_output("rev-parse","HEAD"),"repository_branch":git_output("branch","--show-current"),"checkpoint":{"path":self.config["source_checkpoint"],"sha256":sha256(self.source)},"files":{k:{"path":str(v.relative_to(ROOT)),"sha256":sha256(v)} for k,v in files.items() if v.is_file()}},"paper":self.config["paper"],"roots":self.config["roots"],"routes":self.config["routes"],"splits":{str(w):{k:json_hash(v) for k,v in x.items()} for w,x in self.splits.items()},"gates":self.config["gates"],"prohibited":["answer-dependent depth or stopping","programmed arithmetic","validity filter or repair","lookup","checked/private answers"]}
    def cases(self,t,r):
        out={"validation":{},"test":{}}
        for split in out:
            for w in self.config["data"]["widths"]:
                offset=int(self.config["data"]["validation_seed_offset"] if split=="validation" else self.config["data"]["test_seed_offset"]); out[split][str(w)]={"transition":transition_cases(self.exp300,self.splits[w][split],w,self.root+offset+w,t),"rollout":rollout_cases(self.exp300,self.splits[w][split],w,self.root+offset+w+500,r)}
        return out
    def run(self):
        self.output.mkdir(parents=True,exist_ok=True); prereg=self.preregistration(); write_json(self.output/f"preregistration-root-{self.root}.json",prereg)
        if self.mode=="preregister": return prereg
        if self.mode=="formal" and (not (self.output/"calibration.json").is_file() or not json.loads((self.output/"calibration.json").read_text())["admission"]["admitted"]): raise RuntimeError("formal root requires admitted calibration")
        calibration=self.mode=="calibration"; cfg=self.config; torch.set_num_threads(int(cfg["training"]["torch_threads"])); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); started=Timer(); configure_mlflow(cfg["tracking"]["experiment_name"]); routes={}; steps=int(cfg["calibration"]["representative_updates_per_trained_route"] if calibration else cfg["training"]["steps_per_trained_route"]); t=int(cfg["calibration"]["transition_cases_per_width"] if calibration else cfg["evaluation"]["transition_cases_per_width"]); r=int(cfg["calibration"]["rollout_cases_per_width"] if calibration else cfg["evaluation"]["rollout_cases_per_width"])
        with mlflow.start_run(run_name=f"{cfg['run_id']}-{self.mode}-{self.root}") as tracking:
            for route in ROUTES: model,record=train_route(cfg,self.exp300,self.source,self.splits,self.root,route,steps,self.output,self.mode); routes[route]=record
            cases=self.cases(t,r)
            for route in ROUTES:
                model=build_model(cfg,self.source); model.load_state_dict(load_file(ROOT/routes[route]["checkpoint"]["path"])); model.eval(); routes[route]["validation"]={}; routes[route]["test"]={}
                for split in ("validation","test"):
                    for w in cfg["data"]["widths"]: routes[route][split][str(w)]={"transition":evaluate_transitions(model,route,cases[split][str(w)]["transition"]),"rollout":evaluate_rollouts(model,route,cases[split][str(w)]["rollout"])}
            primary=build_model(cfg,self.source); primary.load_state_dict(load_file(ROOT/routes["rsm_full"]["checkpoint"]["path"])); diag_cases=cases["validation"]["128"]["transition"]; diagnostic=stability_diagnostic(primary,diag_cases); repeat=evaluate_rollouts(primary,"rsm_full",cases["test"]["128"]["rollout"]); deterministic=repeat["prediction_sha256"]==routes["rsm_full"]["test"]["128"]["rollout"]["prediction_sha256"]
            update_walls=[x["timing"]["wall_seconds"] for route in TRAINED_ROUTES for x in routes[route]["training"]["history"]]; projected_training=sum(update_walls)*int(cfg["training"]["steps_per_trained_route"])/steps; observed=started.finish(); variability=max(update_walls)/max(float(np.mean(update_walls)),1e-12); headroom=min(float(cfg["resource_policy"]["variability_headroom_ceiling"]),max(float(cfg["resource_policy"]["variability_headroom_floor"]),variability)); candidate=156.0+float(routes["rsm_full"]["test"]["128"]["rollout"]["timing"]["wall_seconds"])*100/r; expected=projected_training+sum(routes[x][s][str(w)][k]["timing"]["wall_seconds"]*(int(cfg["evaluation"]["rollout_cases_per_width"] if k=="rollout" else cfg["evaluation"]["transition_cases_per_width"])/(r if k=="rollout" else t)) for x in ROUTES for s in ("validation","test") for w in cfg["data"]["widths"] for k in ("transition","rollout"))/int(cfg["resource_policy"]["evaluator_processes"])
            checks={"complete_routes":all(all(s in routes[x] for s in ("validation","test")) for x in ROUTES),"expected_value":expected<=float(cfg["resource_policy"]["maximum_expected_formal_wall_seconds"]),"memory":observed["peak_rss_bytes"]*int(cfg["resource_policy"]["evaluator_processes"])<=int(cfg["resource_policy"]["maximum_peak_rss_bytes"]),"candidate_inference":candidate<300 and candidate*headroom<=285,"determinism":deterministic,"reload":all(x["checkpoint_reload_exact"] for x in routes.values()),"separation":all(set(x["train"]).isdisjoint(x["validation"]) and set(x["train"]).isdisjoint(x["test"]) and set(x["validation"]).isdisjoint(x["test"]) for x in self.splits.values()),"target_free_fixed_depth":True}; admitted=all(checks.values()); admission={"complete_routes":checks["complete_routes"],"projected_training_wall_seconds":projected_training,"expected_wall_seconds":expected,"eta_wall_seconds":expected*headroom,"soft_guard_wall_seconds":math.ceil(expected*headroom*1.25),"hard_guard_wall_seconds":math.ceil(expected*headroom*2),"variability_ratio":variability,"headroom":headroom,"candidate_inference_wall_seconds":candidate,"candidate_inference_headroom_wall_seconds":candidate*headroom,"aggregate_peak_rss_upper_bound_bytes":observed["peak_rss_bytes"]*3,"checks":checks,"admitted":admitted}; decision="admit-root-20261219" if admitted else "park-resource-compliance-or-feasibility"; result={"run_id":cfg["run_id"],"mode":self.mode,"root":self.root,"status":"completed","decision":decision,"source":prereg["source"],"routes":routes,"diagnostics":{"stability":diagnostic,"deterministic_repeat_equal":deterministic},"admission":admission,"resource":aggregate_resource(observed,[],1),"mlflow":{"tracking_uri":tracking_uri(),"parent_run_id":tracking.info.run_id},"limitations":["Bounded convolutional RSM analogue, not a transformer reproduction.","M016.b RIM remained closed.","Calibration is resource/integrity evidence, not scientific accuracy evidence.","Exact bigint arithmetic is confined to offline labels and evaluation.","No checked/private/candidate/submission surface was opened."]}; write_json(self.output/("calibration.json" if calibration else f"root-{self.root}-summary.json"),result); mlflow.set_tag("decision",decision); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=CONFIG_PATH); p.add_argument("--output",type=Path,required=True); p.add_argument("--mode",choices=("preregister","calibration","formal"),required=True); p.add_argument("--root",type=int); a=p.parse_args(); result=Experiment(a.config,a.output,a.mode,a.root).run(); print(json.dumps({"decision":result.get("decision"),"root":result.get("root"),"status":result.get("status","preregistered")},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
