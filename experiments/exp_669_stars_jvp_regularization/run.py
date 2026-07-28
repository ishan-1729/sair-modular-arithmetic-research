"""Execute EXP-669 STARS JVP regularization calibration and gated science."""
from __future__ import annotations
import argparse,json,math,random,sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import mlflow,numpy as np,torch,torch.nn.functional as F,yaml
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.exp_664_sprm_boundary_perturbation.run import Timer,aggregate_resource,bits,features,git_output,json_hash,load_exp300,make_split,rollout_cases,sha256,transition_cases,write_json
from experiments.exp_667_rsm_shallow_train_deep_infer.run import evaluate_rollouts,evaluate_transitions
from experiments.exp_669_stars_jvp_regularization.model import ROUTES,TRAINED_ROUTES,STARSCell
from tools.autoresearch_mlflow import configure_mlflow,tracking_uri
CONFIG_PATH=ROOT/"configs/exp_669_stars_jvp_regularization.yaml"

def build_model(config:dict[str,Any],source:Path)->STARSCell:
    m=STARSCell(**config["model"]);m.load_source(load_file(source));return m
class InferenceView:
    """Compatibility view; every route invokes the exact unchanged source forward."""
    def __init__(self,model):self.model=model
    def __call__(self,batch,**unused):
        logits,q=self.model(batch);zero=torch.zeros(batch.shape[0],device=batch.device);return SimpleNamespace(logits=logits,quotient_logits=q,relative_residual=zero,iterations=torch.full_like(zero,len(self.model.dilation_schedule(batch.shape[-1])),dtype=torch.long))

def base_objective(logits,qlogits,target,quotient,width,cfg):
    truth=bits(target,width);per=F.binary_cross_entropy_with_logits(logits,truth,reduction="none");return per.mean()+float(cfg["training"]["worst_bit_weight"])*per.mean(0).max()+float(cfg["training"]["quotient_weight"])*F.cross_entropy(qlogits,quotient),truth
def train_route(config,exp300,source,splits,root,route,steps,output,mode):
    random.seed(root+669);np.random.seed(root+669);torch.manual_seed(root+669);d=output/mode/f"root-{root}"/route;d.mkdir(parents=True,exist_ok=True);ckpt=d/"model.safetensors";cache=d/"training-result.json";key=json_hash({"source":sha256(source),"root":root,"route":route,"steps":steps,"training":config["training"],"regularization":config["regularization"],"splits":splits})
    if ckpt.is_file() and cache.is_file():
        old=json.loads(cache.read_text());
        if old.get("cache_key")==key and sha256(ckpt)==old["result"]["checkpoint"]["sha256"]:
            m=build_model(config,source);m.load_state_dict(load_file(ckpt));return m,{**old["result"],"cached":True}
    m=build_model(config,source);timer=Timer();history=[];examples=calls=jvp_calls=0
    if route=="randomized":m.randomized_intervention(root+66991)
    if route in TRAINED_ROUTES:
        params=list(m.parameters());opt=torch.optim.AdamW(params,lr=float(config["training"]["learning_rate"]),weight_decay=float(config["training"]["weight_decay"]));sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps,eta_min=float(config["training"]["minimum_learning_rate"]));samplers={w:exp300.BigIntSampler(splits[w]["train"],root+int(config["data"]["training_seed_offset"])+w) for w in config["data"]["training_widths"]};lam=float(config["regularization"]["lambda"]);m.train()
        for step in range(1,steps+1):
            width=int(config["training"]["schedule"][(step-1)%4]);state,mult,mod,input_bit,target,quot=samplers[width].transitions(int(config["training"]["batch_size"]));batch=features(state,mult,mod,input_bit,width);tick=Timer();logits,qlogits,boundary,dilation,next_state=m.forward_with_boundary(batch);base,truth=base_objective(logits,qlogits,target,quot,width,config);record=m.jvp_proxy(boundary,batch,dilation,seed=root+step+66900,route=route);regularizer=record.proxy
            if route=="state_norm":regularizer=next_state.flatten(1).square().mean()
            weight=0.0 if route=="lambda_zero" else lam;loss=(1.0-weight)*base+weight*regularizer;opt.zero_grad(set_to_none=True);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(params,float(config["training"]["gradient_clip"])));reg_grads=torch.autograd.grad(record.proxy,boundary,allow_unused=True,retain_graph=False) if False else ();opt.step();sched.step();examples+=len(target);calls+=len(target)*len(m.dilation_schedule(width));jvp_calls+=record.jvp_calls*len(target);history.append({"step":step,"width":width,"loss":float(loss.detach()),"base_loss":float(base.detach()),"proxy":float(record.proxy),"regularizer":float(regularizer.detach()),"gradient_norm":grad,"batch_exact":float((logits.ge(0)==truth.bool()).all(1).float().mean()),"first_jvp_norm":float(record.first_jvp_norm),"vector_norm":float(record.vector_norm),"state_norm":float(record.state_norm),"jvp_calls":record.jvp_calls,"proxy_parameter_gradient_enabled":record.parameter_gradient_enabled,"timing":tick.finish()})
    save_file({k:v.detach().contiguous() for k,v in m.state_dict().items()},ckpt);re=build_model(config,source);re.load_state_dict(load_file(ckpt));exact=all(torch.equal(a,b) for a,b in zip(m.state_dict().values(),re.state_dict().values(),strict=True));path=str(ckpt.relative_to(ROOT)) if ckpt.is_relative_to(ROOT) else str(ckpt);result={"route":route,"cached":False,"parameters":m.parameter_count(),"checkpoint":{"path":path,"sha256":sha256(ckpt),"bytes":ckpt.stat().st_size},"checkpoint_reload_exact":exact,"inference_calls_per_transition":len(m.dilation_schedule(128)),"training":{"updates":steps if route in TRAINED_ROUTES else 0,"examples":examples,"source_recurrent_calls":calls,"jvp_calls":jvp_calls,"history":history,"timing":timer.finish()}};write_json(cache,{"cache_key":key,"result":result});return m,result

def stability_diagnostic(model,cases,seed,epsilon):
    width=int(cases["width"]);batch=features(cases["state"],cases["multiplicand"],cases["modulus"],cases["input_bit"],width);_,_,state,dilation,_=model.forward_with_boundary(batch);records={}
    for route in ("stars","lambda_zero","state_norm","random_stop_gradient"):
        record=model.jvp_proxy(state,batch,dilation,seed=seed,route=route);g=torch.Generator().manual_seed(seed+1);noise=torch.randn(state.shape,generator=g,dtype=state.dtype);noise=noise/noise.flatten(1).norm(dim=1).view(-1,1,1).clamp_min(1e-12)*epsilon;a=model.recurrent_update(state,batch,dilation);b=model.recurrent_update(state+noise,batch,dilation);records[route]={"one_step_jvp_squared_norm_proxy":float(record.proxy),"first_jvp_norm":float(record.first_jvp_norm),"vector_norm":float(record.vector_norm),"perturbation_amplification":float((b-a).flatten(1).norm(dim=1).mean()/epsilon),"state_norm":float(record.state_norm),"jvp_calls":record.jvp_calls,"exact_spectral_radius":False}
    return records
class Experiment:
    def __init__(self,config_path,output,mode,root=None):
        self.config_path=Path(config_path).resolve();self.config=yaml.safe_load(self.config_path.read_text());self.output=Path(output).resolve();self.mode=mode;expected={"preregister":20261226,"calibration":20261226,"formal":20261227}[mode];self.root=expected if root is None else int(root);allowed={expected} if mode!="formal" else {20261227,20261228,20261229}
        if self.root not in allowed:raise ValueError("unfrozen root")
        self.source=(ROOT/self.config["source_checkpoint"]).resolve();
        if sha256(self.source)!=self.config["source_checkpoint_sha256"] or sha256(ROOT/self.config["paper"]["path"])!=self.config["paper"]["sha256"]:raise RuntimeError("source identity mismatch")
        for name,digest in self.config["fixture_hashes"].items():
            if sha256(ROOT/".scratch/exp-658-joint-quotient-carry-transducer/pinned_sources"/name)!=digest:raise RuntimeError("fixture mismatch")
        self.exp300=load_exp300();self.splits={w:make_split(self.config,self.exp300,w,self.root) for w in self.config["data"]["widths"]}
    def preregistration(self):return {"run_id":self.config["run_id"],"root":self.root,"mode":self.mode,"source":{"repository_commit":git_output("rev-parse","HEAD"),"repository_branch":git_output("branch","--show-current"),"checkpoint_sha256":sha256(self.source),"challenge_commit":"82510bba00a1126649bd76dd1a451f14d0b3eb60"},"paper":self.config["paper"],"fixtures":self.config["fixture_hashes"],"routes":self.config["routes"],"regularization":self.config["regularization"],"roots":self.config["roots"],"splits":{str(w):{k:json_hash(v) for k,v in x.items()} for w,x in self.splits.items()},"gates":self.config["gates"],"closed":["M026.b random loop counts / M018.a","inference JVP","exact spectral radius","checked/private/candidate/submission"]}
    def cases(self,t,r):
        out={"validation":{},"test":{}}
        for split in out:
            for w in self.config["data"]["widths"]:
                off=int(self.config["data"]["validation_seed_offset"] if split=="validation" else self.config["data"]["test_seed_offset"]);out[split][str(w)]={"transition":transition_cases(self.exp300,self.splits[w][split],w,self.root+off+w,t),"rollout":rollout_cases(self.exp300,self.splits[w][split],w,self.root+off+w+500,r)}
        return out
    def run(self):
        self.output.mkdir(parents=True,exist_ok=True);pre=self.preregistration();write_json(self.output/f"preregistration-root-{self.root}.json",pre)
        if self.mode=="preregister":return pre
        if self.mode=="formal" and (not (self.output/"calibration.json").is_file() or not json.loads((self.output/"calibration.json").read_text())["admission"]["admitted"]):raise RuntimeError("formal blocked")
        cfg=self.config;cal=self.mode=="calibration";torch.set_num_threads(int(cfg["training"]["torch_threads"]));torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);started=Timer();configure_mlflow(cfg["tracking"]["experiment_name"]);routes={};steps=int(cfg["calibration"]["representative_updates_per_trained_route"] if cal else cfg["training"]["steps_per_trained_route"]);t=int(cfg["calibration"]["transition_cases_per_width"] if cal else cfg["evaluation"]["transition_cases_per_width"]);r=int(cfg["calibration"]["rollout_cases_per_width"] if cal else cfg["evaluation"]["rollout_cases_per_width"])
        with mlflow.start_run(run_name=f"{cfg['run_id']}-{self.mode}-{self.root}") as tracking:
            for route in ROUTES:_,routes[route]=train_route(cfg,self.exp300,self.source,self.splits,self.root,route,steps,self.output,self.mode)
            cases=self.cases(t,r)
            for route in ROUTES:
                m=build_model(cfg,self.source);m.load_state_dict(load_file(ROOT/routes[route]["checkpoint"]["path"]));m.eval();view=InferenceView(m);routes[route]["validation"]={};routes[route]["test"]={}
                for split in ("validation","test"):
                    for w in cfg["data"]["widths"]:routes[route][split][str(w)]={"transition":evaluate_transitions(view,route,cases[split][str(w)]["transition"]),"rollout":evaluate_rollouts(view,route,cases[split][str(w)]["rollout"])}
            primary=build_model(cfg,self.source);primary.load_state_dict(load_file(ROOT/routes["stars"]["checkpoint"]["path"]));diag={str(w):stability_diagnostic(primary,cases["validation"][str(w)]["transition"],self.root+66900+w,float(cfg["calibration"]["perturbation_epsilon"])) for w in cfg["data"]["widths"]};view=InferenceView(primary);repeat=evaluate_rollouts(view,"stars",cases["test"]["128"]["rollout"]);det=repeat["prediction_sha256"]==routes["stars"]["test"]["128"]["rollout"]["prediction_sha256"];source=build_model(cfg,self.source);identity=all(torch.equal(a,b) for a,b in zip(source.state_dict().values(),build_model(cfg,self.source).state_dict().values(),strict=True)) and source.parameter_count()==primary.parameter_count() and len(source.dilation_schedule(128))==routes["stars"]["inference_calls_per_transition"];walls=[x["timing"]["wall_seconds"] for route in TRAINED_ROUTES for x in routes[route]["training"]["history"]];project=sum(walls)*400/steps;variability=max(walls)/max(float(np.mean(walls)),1e-12);head=min(2.,max(1.25,variability));candidate=156.+max(0.,routes["stars"]["test"]["128"]["rollout"]["timing"]["wall_seconds"]-routes["frozen_source"]["test"]["128"]["rollout"]["timing"]["wall_seconds"])*100/r;expected=project+sum(routes[x][s][str(w)][k]["timing"]["wall_seconds"]*(int(cfg["evaluation"]["rollout_cases_per_width"] if k=="rollout" else cfg["evaluation"]["transition_cases_per_width"])/(r if k=="rollout" else t)) for x in ROUTES for s in ("validation","test") for w in cfg["data"]["widths"] for k in ("transition","rollout"))/3;resource=aggregate_resource(started.finish(),[],1);checks={"complete_routes":True,"expected_value":expected<=21600,"memory":resource["peak_rss_bytes"]*3<=6442450944,"candidate_inference":candidate<300 and candidate*head<=285,"determinism":det,"reload":all(x["checkpoint_reload_exact"] for x in routes.values()),"separation":all(set(x["train"]).isdisjoint(x["validation"]) and set(x["train"]).isdisjoint(x["test"]) and set(x["validation"]).isdisjoint(x["test"]) for x in self.splits.values()),"inference_identity":identity,"finite_jvp":all(math.isfinite(v[k]) for w in diag.values() for v in w.values() for k in ("one_step_jvp_squared_norm_proxy","perturbation_amplification"))};admitted=all(checks.values());admission={"admitted":admitted,"checks":checks,"projected_training_wall_seconds":project,"expected_wall_seconds":expected,"eta_wall_seconds":expected*head,"soft_guard_wall_seconds":math.ceil(expected*head*1.25),"hard_guard_wall_seconds":math.ceil(expected*head*2),"variability_ratio":variability,"headroom":head,"candidate_inference_wall_seconds":candidate,"candidate_inference_headroom_wall_seconds":candidate*head,"aggregate_peak_rss_upper_bound_bytes":resource["peak_rss_bytes"]*3};decision="admit-root-20261227" if admitted else "park-resource-compliance-or-feasibility";result={"run_id":cfg["run_id"],"status":"completed","mode":self.mode,"root":self.root,"decision":decision,"source":pre["source"],"routes":routes,"diagnostics":{"stability":diag,"deterministic_repeat_equal":det,"inference_identity":identity},"admission":admission,"resource":resource,"mlflow":{"tracking_uri":tracking_uri(),"parent_run_id":tracking.info.run_id},"limitations":["One-step normalized-vector JVP squared-norm proxy, not exact spectral radius.","M026.b/M018.a random loop counts closed.","Calibration is non-scientific.","No inference JVP or graph change.","No checked/private/candidate/submission surface opened."]};write_json(self.output/("calibration.json" if cal else f"root-{self.root}-summary.json"),result);mlflow.set_tag("decision",decision);return result
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=CONFIG_PATH);p.add_argument("--output",type=Path,required=True);p.add_argument("--mode",choices=("preregister","calibration","formal"),required=True);p.add_argument("--root",type=int);a=p.parse_args();x=Experiment(a.config,a.output,a.mode,a.root).run();print(json.dumps({"decision":x.get("decision"),"root":x.get("root")},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
