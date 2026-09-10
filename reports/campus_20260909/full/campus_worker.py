"""One bounded pipeline operation, launched only by run_all.py."""
import json
import os
from pathlib import Path
import sys
import time
from campus_support import ROOT, HERE, ALGORITHMS, install_policy, read, write, sha, best_complete


def open_study(run, algorithm, cfg, manifest):
    import pickle
    import optuna
    from src.experiments import study_signature
    strategies = ["none", "class_weight", "random_oversample"]
    path = run / f"{algorithm}_sampler.pkl"
    sampler = pickle.loads(path.read_bytes()) if path.exists() else optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    study = optuna.create_study(storage=f"sqlite:///{(run/'studies.sqlite3').as_posix()}", study_name=algorithm,
                               direction="maximize", load_if_exists=True, sampler=sampler,
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    signature = study_signature(manifest, cfg, algorithm, strategies)
    if study.user_attrs.get("signature", signature) != signature:
        raise ValueError("Study signature changed")
    study.set_user_attr("signature", signature)
    study.set_user_attr("strategies", strategies)
    return study, path, strategies


def tune_one(run, algorithm):
    import pickle
    import numpy as np
    import optuna
    from src import experiments as ex
    with ex.guarded_context(run, "tune", algorithm) as (_, cfg, X, Y, manifest, guard):
        ex.require_selection_open(run)
        study, sampler_path, strategies = open_study(run, algorithm, cfg, manifest)
        trial = study.ask()
        try:
            strategy = trial.suggest_categorical("imbalance", strategies)
            params = ex.suggest_parameters(trial, algorithm)
            trial.set_user_attr("model_params", params)
            trial.set_user_attr("imbalance", strategy)
            # Commit RNG after proposal and BEFORE fitting, so killed work is not replayed.
            tmp = sampler_path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(study.sampler)); tmp.replace(sampler_path)
            started = time.monotonic()
            scores, rounds = ex.score_recipe(algorithm, params, strategy, X, Y, cfg["splits"],
                                            ex.FeatureSpec.from_dict(manifest["feature_spec"]), cfg, guard, trial)
            trial.set_user_attr("fold_scores", scores)
            trial.set_user_attr("best_iterations", rounds)
            trial.set_user_attr("seconds", time.monotonic()-started)
            study.tell(trial, float(np.mean(scores)))
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except BaseException:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise
        finally:
            study.trials_dataframe().to_csv(run/f"{algorithm}_trials.csv", index=False)
    return {"attempts": len(study.trials)}


def sensitivity(run, algorithm, variant):
    import numpy as np
    from src import experiments as ex
    from src.artifacts import json_hash
    from src.baseline_checkpoint import BaselineCheckpoint
    from src.metrics import average_precision
    meta = read(run/algorithm/"model_manifest.json")
    target = run/"sensitivity"/algorithm/variant
    target.mkdir(parents=True, exist_ok=True)
    with ex.guarded_context(run, "explain", algorithm) as (_, cfg, X, Y, manifest, guard):
        definition = read(ROOT/"reports/step3_20260908/sensitivity_execution_partitions.json")[variant]
        folds, seed = definition["folds"], definition["seed"]
        rounds = [m["rounds"] for m in meta["models"]]
        signature = json_hash({"manifest":sha(run/algorithm/"model_manifest.json"),"variant":definition})
        cp = BaselineCheckpoint(target, algorithm, signature, [meta["imbalance"]], folds, Y.shape[1])
        pooled = np.full(Y.shape, np.nan)
        scores, support = [], []
        for number, fold in enumerate(folds):
            train, score = fold["train"], fold["score"]
            P = np.empty((len(score), Y.shape[1])); xr = yr = None
            for j in range(Y.shape[1]):
                guard.check(force=True)
                key = f"{meta['imbalance']}/{number}/{j}"
                cached = cp.load(key)
                if cached is None:
                    if xr is None:
                        xr, yr = ex.resample(X[train], Y[train], meta["imbalance"], seed=seed+number,
                                             n_bits=meta["feature_spec"]["n_bits"])
                    model, fitted = ex.fit_binary(algorithm, meta["model_params"], xr, yr[:,j],
                                                 threads=2, seed=seed, rounds=rounds[j], guard=guard,
                                                 weighting=meta["imbalance"]=="class_weight")
                    values = model.predict_proba(X[score])[:,1]
                    del model
                    cp.save(key, values, fitted)
                else:
                    values, _ = cached
                P[:,j] = values
            pooled[score] = P
            scores.append(average_precision(Y[score], P))
            support.append(Y[score].sum(axis=0).tolist())
        fit = cfg["splits"]["fit"]
        if not np.isfinite(pooled[fit]).all():
            raise ValueError("Incomplete sensitivity OOF coverage")
        np.savez_compressed(target/"oof_predictions.npz", indices=fit, probabilities=pooled[fit], truth=Y[fit])
        result={"variant":variant,"seed":seed,"grouped":definition["grouped"],"fold_macro_ap":scores,
                "mean_fold_macro_ap":float(np.mean(scores)),"pooled_oof_macro_ap":average_precision(Y[fit],pooled[fit]),
                "fold_positive_support":support,"manifest_sha256":sha(run/algorithm/"model_manifest.json"),
                "interpretation":"Conditional robustness; fixed recipe and rounds; no retuning"}
        guard.check(force=True)
        write(target/"metrics.json", result)
        return result


def evaluate(run, algorithm):
    import numpy as np
    import joblib
    from src import experiments as ex
    from src.metrics import evaluate as metrics
    candidates = {a:run/a/"model_manifest.json" for a in ALGORITHMS}
    if not all(p.exists() for p in candidates.values()):
        raise ValueError("Both candidates must be finalized before test access")
    hashes = {a:sha(p) for a,p in candidates.items()}
    freeze=run/"selection_frozen.json"
    if freeze.exists():
        if read(freeze)["model_manifests"] != hashes:
            raise ValueError("Candidates changed after freeze")
    else:
        values={a:read(p)["cv_ap"] for a,p in candidates.items()}
        write(freeze,{"model_manifests":hashes,"cv_ap":values,"selected_by_cv":max(values,key=values.get)})
    output=run/algorithm
    if (output/"test_metrics.json").exists():
        return read(output/"test_metrics.json")
    cfg=read(run/"run.json"); meta=read(output/"model_manifest.json")
    X,Y,manifest=ex.load_dataset(cfg["dataset"],"test")
    if meta["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Dataset identity differs")
    P=np.zeros(Y.shape)
    for j,item in enumerate(meta["models"]):
        if sha(output/item["filename"]) != item["sha256"]:
            raise ValueError("Model hash differs")
        model=joblib.load(output/item["filename"])
        P[:,j]=model.predict_proba(X)[:,1]
        del model
    result=metrics(Y,P,np.asarray([m["threshold"] for m in meta["models"]]),meta["labels"])
    result["evaluation_status"]=manifest["evaluation_status"]
    np.savez_compressed(output/"test_predictions.npz",probabilities=P,truth=Y)
    write(output/"test_metrics.json",result)
    return result


def operation(request):
    cfg = install_policy(request["seconds"])
    from src import experiments as ex
    run=Path(request["run"]); algorithm=request["algorithm"]; action=request["operation"]
    if action=="init":
        if not (run/"run.json").exists():
            ex.initialize(ROOT/"data/builds/audited-20260904-v2", run)
        current=read(run/"run.json")
        if current["splits"] != read(ROOT/"reports/step3_20260908/development_partitions.json"):
            raise ValueError("Regenerated partitions differ")
        current["campus_amendment_sha256"]=sha(HERE/"amendment.json")
        write(run/"run.json",current)
        return {"initialized":True}
    if action=="baseline": return ex.baseline(run,algorithm)
    if action=="tune": return tune_one(run,algorithm)
    if action=="fit":
        # Explicitly verify the frozen tie-break before using the core finalizer.
        import optuna
        study=optuna.load_study(storage=f"sqlite:///{(run/'studies.sqlite3').as_posix()}",study_name=algorithm)
        best=best_complete(study.trials)
        if best is None or study.best_trial.number != best.number:
            raise ValueError("No eligible candidate or CV tie-break mismatch")
        return str(ex.finalize(run,algorithm))
    if action=="sensitivity": return sensitivity(run,algorithm,request["variant"])
    if action=="evaluate": return evaluate(run,algorithm)
    if action in {"export","benchmark"}:
        X,_,_=ex.load_dataset(ROOT/"data/builds/audited-20260904-v2")
        index=run/algorithm/"onnx_location.json"
        if action=="export":
            from export_for_mobile import export_bundle
            target=run/algorithm/("onnx-"+request["id"])
            result=export_bundle(run/algorithm,target,X[:128])
            write(index,{"directory":target.name,"bundle_id":result["bundle_id"]})
            return result["bundle_id"]
        from src.benchmark_onnx import benchmark
        result=benchmark(run/algorithm/read(index)["directory"],X[:128])
        write(run/algorithm/"desktop_benchmark.json",result)
        return result
    raise ValueError("Unknown operation")


def main():
    request=read(sys.argv[1])
    gate=Path(request["gate"])
    started=time.monotonic()
    while not gate.exists():
        if time.monotonic()-started>30: raise RuntimeError("Supervisor release timed out")
        time.sleep(.05)
    try:
        value=operation(request)
        result={"status":"OK","value":value}
    except BaseException as exc:
        import traceback
        from src.runtime import ResourceLimit
        result={"status":"RESOURCE" if isinstance(exc,ResourceLimit) else "FAILED",
                "reason":str(exc),"traceback":traceback.format_exc()}
        traceback.print_exc()
    write(request["result"],result)
    return 0 if result["status"]=="OK" else 2


if __name__=="__main__": raise SystemExit(main())
