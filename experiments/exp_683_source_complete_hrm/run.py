"""Freeze, preflight, calibrate, and conditionally execute EXP683 HRM."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,platform,random,shutil,subprocess,sys,time
from pathlib import Path
from typing import Any
import mlflow,numpy as np,psutil,torch,torch.nn.functional as F,yaml
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU
from experiments.exp_664_sprm_boundary_perturbation.run import Timer,bits,decode,features,load_exp300,make_split,rollout_cases,transition_cases
from experiments.exp_683_source_complete_hrm.model import ROUTES,TRAINED_ROUTES,SourceCompleteHRM
from tools.autoresearch_mlflow import configure_mlflow,tracking_uri
CONFIG=ROOT/"configs/exp_683_source_complete_hrm.yaml"
PATHS={"config":"configs/exp_683_source_complete_hrm.yaml","model":"experiments/exp_683_source_complete_hrm/model.py","runner":"experiments/exp_683_source_complete_hrm/run.py","test":"tests/test_exp_683_source_complete_hrm.py","preregistration":"knowledge/autoresearch/ablations/exp-683-source-complete-hrm.md","mechanism":"knowledge/autoresearch/research/2026-07-30-exp683-source-complete-hrm-mechanism.md"}
CHECKPOINTS=(Path("/mnt/d/Projects/SAIR_mod_challenge/.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors"),Path("/mnt/d/Projects/SAIR_mod_challenge_exp655_t048_wt/.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors"))
def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1048576),b""):h.update(block)
 return h.hexdigest()
def write_json(path:Path,value:Any)->None:
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");os.replace(tmp,path)
def git(*args)->str:return subprocess.run(["git",*args],cwd=ROOT,check=True,capture_output=True,text=True,timeout=120).stdout.strip()
def json_hash(value)->str:return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def build(cfg):return SourceCompleteHRM(**cfg["model"])
def objective(out,target,quotient,width,route,cfg):
 truth=bits(target,width); selected=out.segment_logits if route!="no_deep_supervision" else (out.segment_logits[-1],); losses=[]
 for logits in selected:
  per=F.binary_cross_entropy_with_logits(logits,truth,reduction="none");losses.append(per.mean()+float(cfg["training"]["worst_bit_weight"])*per.mean(0).max())
 prediction=torch.stack(losses).mean()+float(cfg["training"]["quotient_weight"])*F.cross_entropy(out.segment_quotients[-1],quotient)
 correct=[(logit.ge(0)==truth.bool()).all(1).float() for logit in out.segment_logits]; qloss=prediction.new_zeros(())
 if route!="q_disabled":
  for index,q in enumerate(out.segment_q):
   halt=correct[index]; nxt=out.segment_q[min(index+1,len(out.segment_q)-1)].detach();cont=nxt[:,0] if index==len(out.segment_q)-1 else nxt.max(1).values;targets=torch.stack((halt,cont),1)
   if route=="q_shuffled":targets=targets.roll(1,0)
   qloss=qloss+F.binary_cross_entropy(q,targets)
 total=prediction+float(cfg["training"]["q_weight"])*qloss/len(out.segment_q)
 return total,{"loss":float(total.detach()),"prediction_loss":float(prediction.detach()),"q_loss":float(qloss.detach()),"batch_exact":float(correct[-1].mean()),"mean_segments":float(out.segments.float().mean())}
def source_model(checkpoint):
 m=SharedDilatedConvGRU(width=128,hidden=92,cycles=2);m.load_state_dict(load_file(checkpoint));return m.eval()
def train_route(cfg,checkpoint,splits,exp300,root,route,steps,output,mode):
 d=output/mode/f"root-{root}"/route;d.mkdir(parents=True,exist_ok=True);ck=d/"model.safetensors";cache=d/"training-result.json";key=json_hash({"root":root,"route":route,"steps":steps,"config":cfg,"checkpoint":sha256(checkpoint)})
 if ck.is_file() and cache.is_file():
  old=json.loads(cache.read_text());
  if old.get("key")==key and sha256(ck)==old["result"]["checkpoint"]["sha256"]:m=build(cfg);m.load_state_dict(load_file(ck));return m,{**old["result"],"cached":True}
 random.seed(root+683);np.random.seed(root+683);torch.manual_seed(root+683);m=build(cfg);timer=Timer();history=[];calls=examples=0
 if route=="randomized":m.randomized_intervention(root+68391)
 if route in TRAINED_ROUTES:
  params=list(m.parameters());opt=torch.optim.AdamW(params,lr=float(cfg["training"]["learning_rate"]),weight_decay=float(cfg["training"]["weight_decay"]));sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps,eta_min=float(cfg["training"]["minimum_learning_rate"]));samplers={w:exp300.BigIntSampler(splits[w]["train"],root+683000+w) for w in cfg["training"]["widths"]};m.train()
  for step in range(1,steps+1):
   width=int(cfg["training"]["schedule"][(step-1)%len(cfg["training"]["schedule"])]);state,mult,mod,bit,target,quot=samplers[width].transitions(int(cfg["training"]["batch_size"]));tick=Timer();out=m.forward_train(features(state,mult,mod,bit,width),route=route);loss,record=objective(out,target,quot,width,route,cfg);opt.zero_grad(set_to_none=True);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(params,float(cfg["training"]["gradient_clip"])));opt.step();sched.step();examples+=len(target);calls+=int(out.low_updates+out.high_updates)*len(target);history.append({"step":step,"width":width,"gradient_norm":grad,**record,"timing":tick.finish()})
 save_file({k:v.detach().contiguous() for k,v in m.state_dict().items()},ck);reload=build(cfg);reload.load_state_dict(load_file(ck));exact=all(torch.equal(a,b) for a,b in zip(m.state_dict().values(),reload.state_dict().values(),strict=True));result={"route":route,"cached":False,"parameters":m.parameter_count(),"checkpoint":{"path":str(ck.relative_to(ROOT)),"sha256":sha256(ck),"bytes":ck.stat().st_size},"reload_exact":exact,"training":{"updates":steps if route in TRAINED_ROUTES else 0,"examples":examples,"calls":calls,"history":history,"resource":timer.finish()}};write_json(cache,{"key":key,"result":result});return m,result
@torch.inference_mode()
def eval_transition(m,route,cases):
 width=int(cases["width"]);x=features(cases["state"],cases["multiplicand"],cases["modulus"],cases["input_bit"],width);timer=Timer();o=m(x,route=route);pred=o.logits.ge(0);truth=bits(cases["target"],width).bool();vals=decode(pred);n=len(vals);return {"cases":n,"exact_accuracy":float((pred==truth).all(1).float().mean()),"invalids":sum(v>=p for v,p in zip(vals,cases["modulus"],strict=True)),"mean_segments":float(o.segments.float().mean()),"prediction_sha256":json_hash(vals),"resource":timer.finish()}
@torch.inference_mode()
def eval_rollout(m,route,cases):
 width=int(cases["width"]);left,right,mod=cases["left"],cases["right"],cases["modulus"];state=[0]*len(left);teacher=[0]*len(left);first=[None]*len(left);invalid=[False]*len(left);segments=[];exact_transitions=0;timer=Timer()
 for pos in range(width):
  bit=[(v>>(width-1-pos))&1 for v in left];o=m(features(state,right,mod,bit,width),route=route);state=decode(o.logits.ge(0));teacher=[(2*s+b*r)%p for s,b,r,p in zip(teacher,bit,right,mod,strict=True)];segments+=o.segments.tolist()
  for i,(a,b,p) in enumerate(zip(state,teacher,mod,strict=True)):
   if a!=b and first[i] is None:first[i]=pos
   invalid[i]|=a>=p
   exact_transitions+=int(a==b)
 expected=[(a*b)%p for a,b,p in zip(left,right,mod,strict=True)];return {"cases":len(left),"exact_accuracy":sum(a==b for a,b in zip(state,expected,strict=True))/len(left),"transition_exact_accuracy":exact_transitions/(len(left)*width),"invalids":sum(invalid),"first_error_mean":None if all(v is None for v in first) else float(np.mean([v for v in first if v is not None])),"mean_segments":float(np.mean(segments)),"prediction_sha256":json_hash(state),"resource":timer.finish()}
class Experiment:
 def __init__(self,config,output,mode,root=None):
  self.path=Path(config).resolve();self.cfg=yaml.safe_load(self.path.read_text());self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=True);self.mode=mode;self.root=root;torch.set_num_threads(int(self.cfg["training"]["torch_threads"]));torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);expected=self.cfg["source_contract"]["checkpoint_sha256"];src=next((p for p in CHECKPOINTS if p.is_file() and sha256(p)==expected),None)
  if src is None:raise RuntimeError("EXP609 checkpoint missing")
  self.checkpoint=self.output/"fixtures/exp609.safetensors";self.checkpoint.parent.mkdir(parents=True,exist_ok=True)
  if not self.checkpoint.is_file():shutil.copyfile(src,self.checkpoint)
  if sha256(self.checkpoint)!=expected:raise RuntimeError("checkpoint hash mismatch")
  self.exp300=load_exp300()
 def identity(self):return {"repository_commit":git("rev-parse","HEAD"),"branch":git("branch","--show-current"),"challenge_commit":git("-C",str(ROOT/"sources/github/modular-arithmetic-challenge"),"rev-parse","HEAD"),"paper_sha256":sha256(Path(self.cfg["source_contract"]["paper_path"])),"checkpoint_sha256":sha256(self.checkpoint),**{f"{k}_sha256":sha256(ROOT/v) for k,v in PATHS.items()}}
 def protocol(self):return {"schema_version":"exp683-frozen-v1","source":self.identity(),"route_decision":self.cfg["route_decision"],"library":self.cfg["library_decision"],"model":self.cfg["model"],"routes":self.cfg["routes"],"training":self.cfg["training"],"data":self.cfg["data"],"evaluation":self.cfg["evaluation"],"gates":self.cfg["gates"],"resource":self.cfg["resource_policy"]}
 def freeze(self):
  p=self.output/"frozen-protocol.json";write_json(p,self.protocol());return {"status":"frozen","path":str(p.relative_to(ROOT)),"sha256":sha256(p),**self.protocol()}
 def assert_frozen(self):
  p=self.output/"frozen-protocol.json";current=json.loads(json.dumps(self.protocol()));
  if not p.is_file() or json.loads(p.read_text())!=current:raise RuntimeError("package differs from freeze")
 def preflight(self):
  self.assert_frozen();m=build(self.cfg);x=torch.randn(2,6,16);outputs={r:m(x,route=r) for r in ROUTES[:-1]};checks={"source_authenticated":self.identity()["paper_sha256"]==self.cfg["source_contract"]["paper_sha256"],"de_novo_4000":self.cfg["route_decision"]["selected"]=="source-meaningful-de-novo" and self.cfg["training"]["steps_per_trained_route"]==4000,"all_routes":set(self.cfg["routes"])==set(ROUTES),"target_free_deterministic":all(torch.equal(outputs[r].logits,m(x,route=r).logits) for r in ROUTES[:-1]),"roots":[self.cfg["data"]["calibration"]["root"],self.cfg["data"]["formal"]["root"],*[v["root"] for v in self.cfg["data"]["confirmations"]]]==[20261282,20261283,20261284,20261285],"denominators":all(k in self.cfg["evaluation"] for k in ("transition_cases_per_width","rollout_cases_per_width","calibration_transition_cases_per_width","calibration_rollout_cases_per_width")),"exact_transfer_impossible":self.cfg["route_decision"]["exact_transfer"] is False,"schema_fields":all(k in self.cfg for k in ("run_id","hypothesis","source_contract","routes","gates"))}
  if not all(checks.values()):raise RuntimeError(checks)
  out={"schema_version":"exp683-preflight-v1","status":"passed","checks":checks,"model_parameters":m.parameter_count(),"completion_marker":"preflight-complete"};write_json(self.output/"preflight.json",out);return out
 def calibration(self):
  self.assert_frozen();self.preflight();root=20261282 if self.root is None else int(self.root)
  if root!=20261282:raise ValueError("calibration root must be 20261282")
  splits={w:make_split(self.cfg,self.exp300,w,root) for w in sorted(set(self.cfg["data"]["widths"]+self.cfg["data"]["training_widths"]))};started=Timer();routes={};steps=int(self.cfg["resource_policy"]["calibration_updates_per_trained_route"])
  configure_mlflow(self.cfg["tracking"]["experiment_name"])
  with mlflow.start_run(run_name=f"{self.cfg['run_id']}-calibration-{root}") as tracking:
   for route in ROUTES[:-1]:_,routes[route]=train_route(self.cfg,self.checkpoint,splits,self.exp300,root,route,steps,self.output,"calibration")
   candidate=source_model(self.checkpoint);routes["candidate005"]={"route":"candidate005","parameters":candidate.parameter_count(),"training":{"updates":0},"reload_exact":True}
   cases={w:{"transition":transition_cases(self.exp300,splits[w]["test"],w,root+683100+w,int(self.cfg["evaluation"]["calibration_transition_cases_per_width"])),"rollout":rollout_cases(self.exp300,splits[w]["test"],w,root+683600+w,int(self.cfg["evaluation"]["calibration_rollout_cases_per_width"]))} for w in self.cfg["evaluation"]["widths"]}
   for route in ROUTES[:-1]:
    m=build(self.cfg);m.load_state_dict(load_file(ROOT/routes[route]["checkpoint"])) ;m.eval();routes[route]["evaluation"]={str(w):{"transition":eval_transition(m,route,cases[w]["transition"]),"rollout":eval_rollout(m,route,cases[w]["rollout"])} for w in self.cfg["evaluation"]["widths"]}
   walls=[h["timing"]["wall_seconds"] for r in TRAINED_ROUTES for h in routes[r]["training"]["history"]];project=sum(walls)*4000/max(steps,1);variability=max(walls)/max(float(np.mean(walls)),1e-12);head=min(float(self.cfg["resource_policy"]["headroom_ceiling"]),max(float(self.cfg["resource_policy"]["headroom_floor"]),variability));evalwall=sum(routes[r]["evaluation"][str(w)][k]["resource"]["wall_seconds"] for r in ROUTES[:-1] for w in self.cfg["evaluation"]["widths"] for k in ("transition","rollout"));expected=project+evalwall*300;candidate_seconds=156+routes["full_hrm"]["evaluation"]["128"]["rollout"]["resource"]["wall_seconds"]*100;resource=started.finish();checks={"complete_routes":set(routes)==set(ROUTES),"expected_value":expected<=float(self.cfg["resource_policy"]["maximum_expected_formal_wall_seconds"]),"memory":resource["peak_rss_bytes"]<=int(self.cfg["resource_policy"]["maximum_peak_rss_bytes"]),"candidate_inference":candidate_seconds*head<=float(self.cfg["gates"]["maximum_guarded_candidate_seconds"]),"reload":all(r.get("reload_exact",False) for r in routes.values()),"deterministic":True,"target_free_q_act":True,"fidelity_reduced":self.cfg["resource_policy"]["fidelity_reduced"] is False};admitted=all(checks.values());admission={"admitted":admitted,"checks":checks,"projected_training_wall_seconds":project,"expected_formal_wall_seconds":expected,"eta_wall_seconds":expected*head,"soft_guard_wall_seconds":math.ceil(expected*head*1.25),"hard_guard_wall_seconds":math.ceil(expected*head*2),"variability":variability,"headroom":head,"candidate_guarded_seconds":candidate_seconds*head,"resource":resource};decision="admit-root-20261283" if admitted else "park-root-20261282-resource-admission"
   result={"run_id":self.cfg["run_id"],"status":"completed","mode":"calibration","root":root,"decision":decision,"source":self.identity(),"routes":routes,"admission":admission,"mlflow":{"run_id":tracking.info.run_id,"tracking_uri":tracking_uri()},"limitations":["Calibration is resource/integrity evidence, not accuracy science.","Exact arithmetic appears only in offline labels/evaluation.","No protected evaluation surface opened."]};write_json(self.output/"calibration.json",result);write_json(self.output/"completion.json",{"completion_marker":"calibration-complete","decision":decision});return result
 def formal(self):
  admission=json.loads((self.output/"calibration.json").read_text());
  if not admission["admission"]["admitted"]:raise RuntimeError("formal root blocked by calibration")
  raise NotImplementedError("formal route is opened only by an admitted calibration")
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=CONFIG);p.add_argument("--output",type=Path,default=ROOT/".scratch/exp-683-source-complete-hrm");p.add_argument("--mode",choices=("freeze","preflight","calibration","formal"),required=True);p.add_argument("--root",type=int);a=p.parse_args();e=Experiment(a.config,a.output,a.mode,a.root);result=e.freeze() if a.mode=="freeze" else e.preflight() if a.mode=="preflight" else e.calibration() if a.mode=="calibration" else e.formal();print(json.dumps({"status":result.get("status"),"decision":result.get("decision")},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
