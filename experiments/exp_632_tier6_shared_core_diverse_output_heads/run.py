"""Train and evaluate complementary residual heads over the frozen Tier-6 core."""

from __future__ import annotations

import argparse, hashlib, importlib.util, json, random, subprocess, sys, time
from pathlib import Path

import mlflow, numpy as np, psutil, torch, torch.nn.functional as F, yaml
from safetensors.torch import load_file, save_file
from torch import nn

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sha256_text(value): return hashlib.sha256(value.encode()).hexdigest()
def git_output(*args): return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120).stdout.strip()
def decode(logits): return [sum(int(bit) << position for position, bit in enumerate(row)) for row in logits.ge(0).tolist()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_632_tier6_shared_core_diverse_output_heads.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-632-tier6-shared-core-diverse-output-heads")
    args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text())
    source_path = (ROOT / config["source_checkpoint"]).resolve(); dataset_path = (ROOT / config["training_dataset"]).resolve()
    if sha256(source_path) != config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
    if sha256(dataset_path) != config["training_dataset_sha256"]: raise RuntimeError("training dataset hash mismatch")
    exp300 = load_module("exp300_for_exp632", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
    cellular = load_module("cellular_for_exp632", ROOT / "experiments/exp_220_width_independent_cellular/model.py")
    head_module = load_module("heads_for_exp632", Path(__file__).with_name("model.py"))
    seed = int(config["seed"]); widths = (64, 128); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["training"]["torch_threads"]); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    process = psutil.Process(); started = time.perf_counter(); cpu_started = time.process_time(); runner_hash = sha256(Path(__file__)); model_hash = sha256(Path(__file__).with_name("model.py")); config_hash = sha256(config_path); commit = git_output("rev-parse", "HEAD")
    def split(width):
        local = {"data": {"prime_bits_min": width - config["data"]["prime_bit_span"] + 1, "prime_bits_max": width, "train_prime_identities": config["data"]["train_prime_identities"], "test_prime_identities": config["data"]["test_prime_identities"]}}
        return exp300.bigint_prime_split(local, seed + 63200 + width)
    splits = {width: split(width) for width in widths}; split_hash = sha256_text(json.dumps(splits, sort_keys=True)); dataset = json.loads(dataset_path.read_text())
    def base_model(width=128):
        model = cellular.SharedDilatedConvGRU(width=width, hidden=config["model"]["hidden"], cycles=config["model"]["cycles"]); model.load_state_dict(load_file(source_path)); model.eval(); return model
    def head_model(): return head_module.SharedCoreDiverseHeads(base_model(), heads=config["model"]["heads"], mask_seed=config["model"]["mask_seed"])
    model = head_model(); exact_features = torch.randn(256, 6, 128); reference = base_model()
    with torch.inference_mode():
        ref_logits, ref_q = reference(exact_features); initial_heads, initial_q = model.forward_heads(exact_features); initial_vote, vote_q = model(exact_features)
    exact_initialization = {"all_head_logits_equal": all(torch.equal(initial_heads[:, i], ref_logits) for i in range(3)), "vote_predictions_equal": torch.equal(initial_vote.ge(0), ref_logits.ge(0)), "quotient_equal": torch.equal(initial_q, ref_q) and torch.equal(vote_q, ref_q)}
    if not all(exact_initialization.values()): raise RuntimeError("exact head initialization failed")
    def records_batch(rows):
        return exp300.features([r["state"] for r in rows], [r["right"] for r in rows], [r["modulus"] for r in rows], [r["bit"] for r in rows], 128), exp300.bits([r["target"] for r in rows], 128)
    optimizer = torch.optim.AdamW(model.residual_heads.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["steps"], eta_min=config["training"]["minimum_learning_rate"])
    bootstrap = torch.Generator().manual_seed(seed + 17); row_rng = random.Random(seed + 18); cache_key = sha256_text(json.dumps({"config": config_hash, "runner": runner_hash, "model": model_hash, "source": sha256(source_path), "dataset": sha256(dataset_path)}, sort_keys=True)); partial_path = output / f"training-{cache_key}.partial.pt"
    completed = 0; history = []; cumulative_wall = cumulative_cpu = 0.0
    if partial_path.exists():
        partial = torch.load(partial_path, weights_only=False)
        if partial["cache_key"] != cache_key: raise RuntimeError("partial key mismatch")
        completed = partial["step"]; history = partial["history"]; cumulative_wall = partial["wall_seconds"]; cumulative_cpu = partial["aggregate_cpu_seconds"]; model.load_state_dict(partial["model"]); optimizer.load_state_dict(partial["optimizer"]); scheduler.load_state_dict(partial["scheduler"]); bootstrap.set_state(partial["bootstrap_state"]); row_rng.setstate(partial["row_rng_state"])
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags({"canonical_run_id": config["run_id"], "status": "running", "role": "tier6-shared-core-diverse-heads", "repository_commit": commit, "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower()})
        mlflow.log_params({"seed": seed, "steps": config["training"]["steps"], "source_checkpoint_sha256": sha256(source_path), "dataset_sha256": sha256(dataset_path), "config_sha256": config_hash, "runner_sha256": runner_hash, "split_sha256": split_hash}); mlflow.log_artifact(str(config_path), artifact_path="preregistration"); mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        segment_wall = time.perf_counter(); segment_cpu = time.process_time(); peak = process.memory_info().rss; model.train(); model.base.eval()
        hard_count = round(config["training"]["batch_size"] * config["training"]["hard_fraction"]); control_count = config["training"]["batch_size"] - hard_count
        for step in range(completed + 1, config["training"]["steps"] + 1):
            rows = [dataset["hard"][row_rng.randrange(len(dataset["hard"]))] for _ in range(hard_count)] + [dataset["control"][row_rng.randrange(len(dataset["control"]))] for _ in range(control_count)]; features, target = records_batch(rows); heads, _ = model.forward_heads(features); losses = []
            for index in range(3):
                selected = torch.randint(0, len(rows), (len(rows),), generator=bootstrap); losses.append(F.binary_cross_entropy_with_logits(heads[selected, index], target[selected]))
            loss = sum(losses) / len(losses); optimizer.zero_grad(set_to_none=True); loss.backward(); gradient = float(nn.utils.clip_grad_norm_(model.residual_heads.parameters(), config["training"]["gradient_clip"])); optimizer.step(); scheduler.step()
            if step == 1 or step % 25 == 0 or step == config["training"]["steps"]:
                item = {"step": step, "elapsed_wall_seconds": cumulative_wall + time.perf_counter() - segment_wall, "loss": float(loss.detach()), "gradient_norm": gradient}; history.append(item); mlflow.log_metrics({"train_loss": item["loss"], "gradient_norm": gradient}, step=step); print(json.dumps(item), flush=True)
            if step % config["training"]["save_every_steps"] == 0 or step == config["training"]["steps"]:
                wall_total = cumulative_wall + time.perf_counter() - segment_wall; cpu_total = cumulative_cpu + time.process_time() - segment_cpu; torch.save({"cache_key": cache_key, "step": step, "history": history, "wall_seconds": wall_total, "aggregate_cpu_seconds": cpu_total, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "bootstrap_state": bootstrap.get_state(), "row_rng_state": row_rng.getstate()}, partial_path)
                if time.perf_counter() - started > config["resource_budget"]["hard_runaway_ceiling_wall_seconds"]: raise TimeoutError("EXP-632 runaway ceiling")
        partial = torch.load(partial_path, weights_only=False); model.eval(); checkpoint = output / "weights.safetensors"; save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, checkpoint)
        def transition_eval(candidate, width, cases):
            sampler = exp300.BigIntSampler(splits[width]["test"], seed + 63300 + width); seen = correct = valid = disagree = 0; candidate.width = width; candidate.base.width = width
            with torch.inference_mode():
                while seen < cases:
                    size = min(512, cases-seen); st, ri, mo, bi, ta, _ = sampler.transitions(size); heads, _ = candidate.forward_heads(exp300.features(st,ri,mo,bi,width)); vote = heads.ge(0).sum(1) >= 2; expected = exp300.bits(ta,width).bool(); values = [sum(int(bit)<<i for i,bit in enumerate(row)) for row in vote.tolist()]; correct += int((vote==expected).all(1).sum()); valid += sum(v<p for v,p in zip(values,mo)); disagree += int((heads.ge(0)!=heads[:,0:1].ge(0)).any(dim=(1,2)).sum()); seen += size
            return {"cases": seen, "correct": correct, "exact_accuracy": correct/seen, "valid_residue_rate": valid/seen, "head_disagreement_cases": disagree, "head_disagreement_rate": disagree/seen}
        def rollout(candidate, left, right, moduli, width, use_heads):
            states=[0]*len(left); bad=[False]*len(left); disagree=[False]*len(left); wall=time.perf_counter(); cpu=time.process_time(); candidate.width=width
            if use_heads: candidate.base.width=width
            with torch.inference_mode():
                for position in range(width):
                    bits=[(v>>(width-1-position))&1 for v in left]; feat=exp300.features(states,right,moduli,bits,width)
                    if use_heads:
                        heads,_=candidate.forward_heads(feat); hb=heads.ge(0); vote=hb.sum(1)>=2; states=[sum(int(bit)<<i for i,bit in enumerate(row)) for row in vote.tolist()]; row_dis=(hb!=hb[:,0:1]).any(dim=(1,2)).tolist(); disagree=[a or b for a,b in zip(disagree,row_dis)]
                    else:
                        logits,_=candidate(feat); states=decode(logits)
                    bad=[prior or value>=p for prior,value,p in zip(bad,states,moduli)]
            expected=[(a*b)%p for a,b,p in zip(left,right,moduli)]; correct=sum(a==b for a,b in zip(states,expected)); return {"cases":len(left),"correct":correct,"exact_accuracy":correct/len(left),"any_invalid_residue_rate":sum(bad)/len(left),"head_disagreement_rate":sum(disagree)/len(left) if use_heads else 0.0,"wall_seconds":time.perf_counter()-wall,"aggregate_cpu_seconds":time.process_time()-cpu,"predictions":states}
        evaluation={}; control=base_model();
        for width in widths:
            pairs=exp300.BigIntSampler(splits[width]["test"],seed+63400+width).pairs(config["evaluation"]["rollout_cases"]); evaluation[str(width)]={"transition":transition_eval(model,width,config["evaluation"]["transition_cases"]),"control_rollout":rollout(control,*pairs,width,False),"challenger_rollout":rollout(model,*pairs,width,True)}
            evaluation[str(width)]["control_rollout"].pop("predictions"); evaluation[str(width)]["challenger_rollout"].pop("predictions")
        randomized=head_model(); trained_flat=torch.cat([p.detach().flatten() for p in model.residual_heads.parameters()]); mean=float(trained_flat.mean()); std=float(trained_flat.std().clamp_min(1e-6)); generator=torch.Generator().manual_seed(seed+91)
        with torch.no_grad():
            for parameter in randomized.residual_heads.parameters(): parameter.copy_(torch.randn(parameter.shape,generator=generator)*std+mean)
        random_pairs=exp300.BigIntSampler(splits[128]["test"],seed+63528).pairs(config["evaluation"]["randomized_rollout_cases"]); trained_random_matched=rollout(model,*random_pairs,128,True); randomized_result=rollout(randomized,*random_pairs,128,True); trained_random_matched.pop("predictions"); randomized_result.pop("predictions")
        det_pairs=exp300.BigIntSampler(splits[128]["test"],seed+63628).pairs(config["evaluation"]["determinism_cases"]); first=rollout(model,*det_pairs,128,True); second=rollout(model,*det_pairs,128,True); determinism={"predictions_equal":first["predictions"]==second["predictions"],"metrics_equal":first["correct"]==second["correct"] and first["any_invalid_residue_rate"]==second["any_invalid_residue_rate"]}; determinism["passed"]=all(determinism.values())
        e64,e128=evaluation["64"],evaluation["128"]; projected=36.9+e128["challenger_rollout"]["wall_seconds"]*110/config["evaluation"]["rollout_cases"]; gates=config["gates"]; gain=e128["challenger_rollout"]["exact_accuracy"]-e128["control_rollout"]["exact_accuracy"]; randomized_gain=trained_random_matched["exact_accuracy"]-randomized_result["exact_accuracy"]
        passed=e128["challenger_rollout"]["exact_accuracy"]>=gates["minimum_challenger_width128_rollout"] and gain>=gates["minimum_width128_gain_over_control"] and e64["challenger_rollout"]["exact_accuracy"]>=gates["minimum_challenger_width64_rollout"] and e128["transition"]["exact_accuracy"]>=gates["minimum_challenger_width128_transition"] and e128["challenger_rollout"]["any_invalid_residue_rate"]<=gates["maximum_challenger_width128_invalid_rollout_rate"] and e128["challenger_rollout"]["head_disagreement_rate"]>=gates["minimum_head_disagreement_rate"] and randomized_gain>=gates["minimum_trained_gain_over_randomized_heads"] and projected<=gates["maximum_projected_integrated_seconds"] and (determinism["passed"] or not gates["require_determinism"])
        decision="advance-tier6-diverse-heads" if passed else "reject-tier6-diverse-heads"; summary={"run_id":config["run_id"],"status":"completed","hypothesis":config["hypothesis"],"mlflow":{"tracking_uri":tracking_uri(),"experiment_name":config["tracking"]["experiment_name"],"experiment_id":experiment_id,"run_id":run.info.run_id},"source":{"repository_commit":commit,"dirty":bool(git_output("status","--porcelain")),"challenge_commit":"82510bba00a1126649bd76dd1a451f14d0b3eb60","config_sha256":config_hash,"runner_sha256":runner_hash,"model_sha256":model_hash,"split_sha256":split_hash,"dataset_sha256":sha256(dataset_path),"source_checkpoint_sha256":sha256(source_path)},"exact_initialization":exact_initialization,"training":{"steps":config["training"]["steps"],"history":history,"wall_seconds":partial["wall_seconds"],"aggregate_cpu_seconds":partial["aggregate_cpu_seconds"]},"model":{"parameters":model.parameter_count(),"trainable_parameters":model.trainable_parameter_count(),"checkpoint":{"path":str(checkpoint.relative_to(ROOT)),"sha256":sha256(checkpoint),"bytes":checkpoint.stat().st_size}},"evaluation":evaluation,"randomized_control":{"trained_matched":trained_random_matched,"randomized":randomized_result,"trained_gain":randomized_gain},"determinism":determinism,"comparison":{"width128_gain":gain,"projected_integrated_seconds":projected},"gates":{**gates,"passed":passed},"decision":decision,"resource_budget":config["resource_budget"],"runtime":{"wall_seconds":time.perf_counter()-started,"aggregate_cpu_seconds":time.process_time()-cpu_started,"torch_threads":config["training"]["torch_threads"],"peak_rss_bytes":peak},"limitations":["Fresh generated selection evidence only; checked, consumed independent, and private answers were not read.","Training reuses only content-hashed training records.","Randomized-head control matches global trained residual mean and standard deviation."]}; summary_path=output/"summary.json"; summary_path.write_text(json.dumps(summary,indent=2)+"\n"); mlflow.log_metrics({"width128_control":e128["control_rollout"]["exact_accuracy"],"width128_challenger":e128["challenger_rollout"]["exact_accuracy"],"width128_gain":gain,"randomized_gain":randomized_gain,"wall_seconds":summary["runtime"]["wall_seconds"]}); mlflow.log_artifact(str(summary_path),artifact_path="canonical-preview"); mlflow.set_tags({"status":"completed","decision":decision})
    print(json.dumps({"decision":decision,"evaluation":evaluation,"randomized_control":summary["randomized_control"],"determinism":determinism,"comparison":summary["comparison"],"checkpoint":summary["model"]["checkpoint"],"runtime":summary["runtime"],"mlflow":summary["mlflow"]},indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
