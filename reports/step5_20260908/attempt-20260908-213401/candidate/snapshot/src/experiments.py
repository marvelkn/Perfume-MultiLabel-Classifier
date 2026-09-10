"""Training-only selection with separate Optuna studies and all-label fold scores."""
import copy
from contextlib import contextmanager
from uuid import uuid4
from threadpoolctl import threadpool_limits
import json
import pickle
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import optuna
from optuna.trial import TrialState
from .artifacts import digest, environment, json_hash, write_json
from .config import CONFIG
from .featurize import FeatureSpec
from .splits import development_partitions, scaffold_groups
from .resampling import resample
from .metrics import average_precision, evaluate, thresholds_for
from .runtime import ResourceGuard, validate_training_resources, exclusive_run
from .baseline_checkpoint import BaselineCheckpoint

def load_dataset(directory, partition="train"):
    directory = Path(directory)
    manifest = json.loads((directory / "dataset_manifest.json").read_text())
    if manifest.get("protocol_version") != 2 or not manifest.get("identity_join_validated"):
        raise ValueError("Rebuild and audit the dataset; legacy artifacts cannot enter new experiments")
    for filename, expected in manifest["files"].items():
        if Path(filename).name != filename or digest(directory / filename) != expected:
            raise ValueError(f"Dataset integrity check failed: {filename}")
    spec = FeatureSpec.from_dict(manifest["feature_spec"])
    if spec.schema_id != manifest["feature_schema_id"]:
        raise ValueError("Invalid dataset feature schema")
    X = sp.load_npz(directory / f"X_{partition}.npz").toarray().astype(np.float32)
    Y = sp.load_npz(directory / f"Y_{partition}.npz").toarray().astype(np.uint8)
    if X.shape != (len(Y), spec.n_features) or Y.shape[1] != len(manifest["labels"]):
        raise ValueError("Dataset dimensions do not match manifest")
    if not np.isfinite(X).all() or not np.isin(Y, [0,1]).all():
        raise ValueError("Invalid data values")
    return X, Y, manifest

def initialize(dataset, run):
    validate_training_resources(CONFIG["resources"])
    X, Y, manifest = load_dataset(dataset)
    run = Path(run).resolve()
    if run.exists():
        raise FileExistsError("Use a new run directory")
    split_cfg = dict(CONFIG["split"])
    split_cfg.pop("test_size")
    split_cfg.pop("strategy",None)
    seed=manifest["split"]["seed"]
    groups=None
    if manifest["split"].get("strategy") == "scaffold":
        groups=scaffold_groups((Path(dataset)/"smiles_train.txt").read_text().splitlines())
    splits = development_partitions(Y, seed=seed, groups=groups, **split_cfg)
    for fold in splits["folds"]:
        for role in ("train", "stop", "score"):
            if np.any(Y[fold[role]].sum(axis=0) == 0) or np.any(Y[fold[role]].sum(axis=0) == len(fold[role])):
                raise ValueError(f"Missing class in {role}; revise the partition protocol before tuning")
    threshold_y = Y[splits["threshold"]]
    if np.any(threshold_y.sum(axis=0) == 0) or np.any(threshold_y.sum(axis=0) == len(threshold_y)):
        raise ValueError("Threshold partition lacks class support")
    run.mkdir(parents=True)
    write_json(run / "run.json", {"dataset": str(Path(dataset).resolve()), "dataset_id": manifest["dataset_id"],
                                 "splits": splits, "resources": CONFIG["resources"], "seed": seed,
                                 "environment": environment(), "metric": "macro_average_precision",
                                 "selection_scope": "all_labels", "test_access": "evaluation command only"})
    return run

def context(run):
    run = Path(run)
    cfg = json.loads((run / "run.json").read_text())
    X,Y,manifest = load_dataset(cfg["dataset"])
    if cfg["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Run and dataset differ")
    return run, cfg, X, Y, manifest

@contextmanager
def guarded_context(run, operation, algorithm, temperature_file=None):
    run = Path(run)
    if algorithm not in {"xgb", "lgbm"} or operation not in {"baseline", "tune", "fit", "explain"}:
        raise ValueError("Unknown resource session operation or algorithm")
    if "_reference" in run.resolve().parts:
        raise ValueError("Archived reference runs must never be executed")
    cfg = json.loads((run / "run.json").read_text(encoding="utf-8"))
    validate_training_resources(cfg["resources"])
    log = run / "sessions" / f"{operation}-{algorithm}-{uuid4().hex}.jsonl"
    with exclusive_run(run):
        with ResourceGuard(cfg["resources"], temperature_file, log_path=log,
                           metadata={"operation": operation, "algorithm": algorithm}) as guard:
            with threadpool_limits(limits=cfg["resources"]["threads"]):
                loaded = context(run)
                guard.check(force=True)
                yield (*loaded, guard)
                guard.check(force=True)


def suggest_parameters(trial, algorithm):
    # Separate spaces follow each learner's complexity controls. Ranges are initial hypotheses.
    if algorithm == "xgb":
        return {"max_depth": trial.suggest_int("max_depth",3,8),
                "min_child_weight": trial.suggest_float("min_child_weight",1,30,log=True),
                "learning_rate": trial.suggest_float("learning_rate",.01,.2,log=True),
                "gamma": trial.suggest_float("gamma",0,5),
                "subsample": trial.suggest_float("subsample",.6,1),
                "colsample_bytree": trial.suggest_float("colsample_bytree",.4,1),
                "reg_alpha": trial.suggest_float("reg_alpha",1e-5,10,log=True) if trial.suggest_categorical("use_l1",[False,True]) else 0.,
                "reg_lambda": trial.suggest_float("reg_lambda",1e-3,100,log=True)}
    if algorithm == "lgbm":
        depth = trial.suggest_int("max_depth",4,12)
        return {"max_depth": depth, "num_leaves": trial.suggest_int("num_leaves",8,min(128,2**depth)),
                "min_child_samples": trial.suggest_int("min_child_samples",10,150),
                "learning_rate": trial.suggest_float("learning_rate",.005,.15,log=True),
                "min_split_gain": trial.suggest_float("min_split_gain",0,1),
                "subsample": trial.suggest_float("subsample",.6,1), "subsample_freq": trial.suggest_int("subsample_freq",1,7),
                "colsample_bytree": trial.suggest_float("colsample_bytree",.5,1),
                "reg_alpha": trial.suggest_float("reg_alpha",1e-5,10,log=True) if trial.suggest_categorical("use_l1",[False,True]) else 0.,
                "reg_lambda": trial.suggest_float("reg_lambda",1e-3,50,log=True),
                "max_bin": trial.suggest_categorical("max_bin",[63,127,255])}
    raise ValueError("Unknown algorithm")

def fit_binary(algorithm, params, X, y, *, stop=None, threads=2, seed=42, weighting=False, guard=None, rounds=None):
    if np.unique(y).size != 2:
        raise ValueError("Both classes must be present in a training label")
    if guard:
        guard.check(force=True)
    params = dict(params)
    weight = float((len(y)-y.sum())/y.sum()) if weighting else 1.
    if algorithm == "xgb":
        import xgboost as xgb
        callbacks = []
        if guard:
            class Budget(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, history):
                    guard.check()
                    return False
            callbacks.append(Budget())
        # logloss controls early stopping; study ranking uses sklearn AP separately.
        model = xgb.XGBClassifier(**params, n_estimators=rounds or 2000, tree_method="hist", device="cpu",
                                  n_jobs=threads, random_state=seed, scale_pos_weight=weight,
                                  eval_metric="logloss", early_stopping_rounds=50 if stop else None, callbacks=callbacks)
        model.fit(X,y,eval_set=[stop] if stop else None,verbose=False)
        # Runtime callbacks are training-only and contain live process/sensor state.
        model.set_params(callbacks=None)
        if stop:
            best = model.best_iteration + 1
            # Slice the booster so exported native and ONNX inference use the same trees.
            model._Booster = model.get_booster()[:best]
        else:
            best = rounds or 2000
    elif algorithm == "lgbm":
        import lightgbm as lgb
        callbacks = [lgb.log_evaluation(0)]
        if stop:
            callbacks.append(lgb.early_stopping(50, verbose=False))
        if guard:
            def budget(env):
                guard.check()
            budget.order = 5
            callbacks.append(budget)
        model = lgb.LGBMClassifier(**params, n_estimators=rounds or 3000, device_type="cpu", n_jobs=threads,
                                   random_state=seed, scale_pos_weight=weight, verbosity=-1,
                                   deterministic=True, force_col_wise=True)
        model.fit(X,y,eval_set=[stop] if stop else None,eval_metric="binary_logloss",callbacks=callbacks)
        best = model.best_iteration_ if stop else (rounds or 3000)
    else:
        raise ValueError("Unknown algorithm")
    return model, int(best)

def score_recipe(algorithm, params, strategy, X, Y, splits, spec, cfg, guard, trial=None, checkpoint=None):
    scores, iterations = [], []
    for number, fold in enumerate(splits["folds"]):
        guard.check(force=True)
        train, stop, score = (fold[k] for k in ("train", "stop", "score"))
        xr = yr = None
        predictions = np.zeros((len(score), Y.shape[1]))
        fold_iterations = []
        for j in range(Y.shape[1]):
            guard.check(force=True)
            key = f"{strategy}/{number}/{j}"
            cached = checkpoint.load(key) if checkpoint is not None else None
            if cached is None:
                if xr is None:
                    xr, yr = resample(X[train], Y[train], strategy,
                                      seed=cfg["seed"] + number, n_bits=spec.n_bits)
                model, rounds = fit_binary(algorithm, params, xr, yr[:, j], stop=(X[stop], Y[stop, j]),
                                           threads=cfg["resources"]["threads"], seed=cfg["seed"],
                                           weighting=strategy == "class_weight", guard=guard)
                probabilities = model.predict_proba(X[score])[:, 1]
                del model
                if checkpoint is not None:
                    checkpoint.save(key, probabilities, rounds)
            else:
                probabilities, rounds = cached
            predictions[:, j] = probabilities
            fold_iterations.append(rounds)
        guard.check(force=True)
        iterations.append(fold_iterations)
        scores.append(average_precision(Y[score], predictions))
        if trial:
            trial.report(float(np.mean(scores)), step=number)
            if trial.should_prune():
                trial.set_user_attr("fold_scores", scores)
                raise optuna.TrialPruned()
    return scores, iterations


def require_selection_open(run):
    if (Path(run)/"selection_frozen.json").exists() or any(Path(run).glob("*/test_metrics.json")):
        raise ValueError("Test evaluation has begun; model selection is frozen for this run")

def study_signature(manifest, cfg, algorithm, strategies):
    current = environment()
    return json_hash({"dataset":manifest["dataset_id"],"splits":cfg["splits"],"resources":cfg["resources"],
                      "algorithm":algorithm,"strategies":list(strategies),"source":current["source_sha256"],
                      "packages":current["packages"]})

def tune(run, algorithm, n_trials, temperature_file=None, strategies=("none","class_weight","random_oversample")):
    if n_trials < 1:
        raise ValueError("Trial count must be positive")
    with guarded_context(run, "tune", algorithm, temperature_file) as (run, cfg, X, Y, manifest, guard):
        spec = FeatureSpec.from_dict(manifest["feature_spec"])
        require_selection_open(run)
        signature = study_signature(manifest,cfg,algorithm,strategies)
        storage = f"sqlite:///{(run/'studies.sqlite3').resolve().as_posix()}"
        sampler_file = run / f"{algorithm}_sampler.pkl"
        sampler = pickle.loads(sampler_file.read_bytes()) if sampler_file.exists() else optuna.samplers.TPESampler(seed=cfg["seed"],n_startup_trials=10)
        study = optuna.create_study(storage=storage,study_name=algorithm,direction="maximize",load_if_exists=True,
                                    sampler=sampler,pruner=optuna.pruners.MedianPruner(n_startup_trials=10,n_warmup_steps=1))
        if study.user_attrs.get("signature",signature) != signature:
            raise ValueError("Study protocol/code differs; create a new run")
        if (run / algorithm / "model_manifest.json").exists():
            raise ValueError("This algorithm is finalized; create a new run to tune further")
        study.set_user_attr("signature",signature)
        study.set_user_attr("strategies",list(strategies))
        def objective(trial):
            strategy = trial.suggest_categorical("imbalance",list(strategies))
            params = suggest_parameters(trial,algorithm)
            trial.set_user_attr("model_params",params)
            trial.set_user_attr("imbalance",strategy)
            start = time.monotonic()
            scores, rounds = score_recipe(algorithm,params,strategy,X,Y,cfg["splits"],spec,cfg,guard,trial)
            trial.set_user_attr("fold_scores",scores)
            trial.set_user_attr("best_iterations",rounds)
            trial.set_user_attr("seconds",time.monotonic()-start)
            return float(np.mean(scores))
        def checkpoint(study, trial):
            temporary = sampler_file.with_suffix(".tmp")
            temporary.write_bytes(pickle.dumps(study.sampler))
            temporary.replace(sampler_file)
            study.trials_dataframe().to_csv(run/f"{algorithm}_trials.csv",index=False)
        try:
            study.optimize(objective,n_trials=n_trials,n_jobs=1,gc_after_trial=True,callbacks=[checkpoint])
        finally:
            checkpoint(study,None)
        return study

def baseline(run, algorithm, temperature_file=None):
    with guarded_context(run, "baseline", algorithm, temperature_file) as (run, cfg, X, Y, manifest, guard):
        require_selection_open(run)
        target = run / f"{algorithm}_baselines.json"
        if target.exists():
            raise FileExistsError(target)
        spec = FeatureSpec.from_dict(manifest["feature_spec"])
        strategies = ("none", "class_weight", "random_oversample")
        signature = json_hash({"study": study_signature(manifest, cfg, algorithm, strategies),
                               "seed": cfg["seed"], "labels": manifest["labels"],
                               "feature_spec": manifest["feature_spec"], "model_params": {}})
        checkpoint = BaselineCheckpoint(run, algorithm, signature, strategies, cfg["splits"]["folds"], Y.shape[1])
        results = {}
        for strategy in strategies:
            scores, _ = score_recipe(algorithm, {}, strategy, X, Y, cfg["splits"], spec, cfg, guard,
                                     checkpoint=checkpoint)
            results[strategy] = {"fold_ap": scores, "mean_ap": float(np.mean(scores))}
            checkpoint.record_results(results)
        guard.check(force=True)
        write_json(target, results)
        return results


def finalize(run, algorithm, temperature_file=None):
    with guarded_context(run, "fit", algorithm, temperature_file) as (run, cfg, X, Y, manifest, guard):
        import joblib
        require_selection_open(run)
        output = run/algorithm
        if (output/"model_manifest.json").exists():
            raise FileExistsError("This algorithm is already finalized")
        study = optuna.load_study(storage=f"sqlite:///{(run/'studies.sqlite3').resolve().as_posix()}",study_name=algorithm)
        signature=study_signature(manifest,cfg,algorithm,study.user_attrs.get("strategies",[]))
        if study.user_attrs.get("signature") != signature:
            raise ValueError("Study code/protocol differs; final fitting cannot proceed")
        trial = study.best_trial
        params, strategy = trial.user_attrs["model_params"], trial.user_attrs["imbalance"]
        spec = FeatureSpec.from_dict(manifest["feature_spec"])
        train, threshold = cfg["splits"]["fit"], cfg["splits"]["threshold"]
        xr,yr = resample(X[train],Y[train],strategy,seed=cfg["seed"],n_bits=spec.n_bits)
        rounds = np.median(np.asarray(trial.user_attrs["best_iterations"]),axis=0).astype(int)
        probabilities = np.zeros((len(threshold),Y.shape[1]))
        recipe=json_hash({"signature":signature,"trial":trial.number,"rounds":rounds.tolist()})
        progress_path=output/"fit_progress.json"
        models=[]
        if output.exists():
            if not progress_path.exists(): raise ValueError("Unrecognized partial fit directory")
            progress=json.loads(progress_path.read_text())
            if progress["recipe"] != recipe: raise ValueError("Cannot resume a different final-fitting recipe")
            models=progress["models"]
            if [m["label"] for m in models] != manifest["labels"][:len(models)]:
                raise ValueError("Partial model order mismatch")
        else:
            output.mkdir()
            write_json(progress_path,{"recipe":recipe,"models":[]})
        for j,label in enumerate(manifest["labels"]):
            guard.check(force=True)
            if j < len(models):
                item=models[j]
                if Path(item["filename"]).name != item["filename"] or digest(output/item["filename"]) != item["sha256"]:
                    raise ValueError("Partial model integrity failure")
                model=joblib.load(output/item["filename"])
            else:
                model,_ = fit_binary(algorithm,params,xr,yr[:,j],threads=cfg["resources"]["threads"],seed=cfg["seed"],
                                    weighting=strategy=="class_weight",guard=guard,rounds=int(rounds[j]))
                filename = f"{algorithm}_{j:03d}.pkl"
                temporary=output/(filename+".tmp")
                joblib.dump(model,temporary);temporary.replace(output/filename)
                models.append({"label":label,"filename":filename,"sha256":digest(output/filename),"rounds":int(rounds[j])})
                write_json(progress_path,{"recipe":recipe,"models":models})
            probabilities[:,j] = model.predict_proba(X[threshold])[:,1]
            del model
        thresholds = thresholds_for(Y[threshold],probabilities)
        for item,t in zip(models,thresholds):
            item["threshold"] = float(t)
        np.savez_compressed(output/"threshold_predictions.npz",probabilities=probabilities,truth=Y[threshold])
        guard.check(force=True)
        write_json(output/"model_manifest.json",{"algorithm":algorithm,"dataset_id":manifest["dataset_id"],"feature_spec":spec.to_dict(),
                   "feature_schema_id":spec.schema_id,"labels":manifest["labels"],"models":models,
                   "best_trial":trial.number,"cv_ap":trial.value,"model_params":params,"imbalance":strategy,
                   "threshold_source":"held-out development partition","environment":environment()})
        return output

def evaluate_test(run, algorithm):
    import joblib
    run = Path(run)
    output = run/algorithm
    if (output/"test_metrics.json").exists():
        raise FileExistsError("Test evaluation already recorded; it must not drive retuning")
    # Freeze both candidates and the validation-selected winner before reading test data.
    candidates = {name: run/name/"model_manifest.json" for name in ("xgb", "lgbm")}
    if not all(p.exists() for p in candidates.values()):
        raise ValueError("Finalize both XGBoost and LightGBM before either test evaluation")
    hashes = {name:digest(p) for name,p in candidates.items()}
    freeze = run/"selection_frozen.json"
    if freeze.exists():
        if json.loads(freeze.read_text())["model_manifests"] != hashes:
            raise ValueError("Finalized candidates changed after test access")
    else:
        scores = {name:json.loads(p.read_text())["cv_ap"] for name,p in candidates.items()}
        write_json(freeze,{"model_manifests":hashes,"cv_ap":scores,"selected_by_cv":max(scores,key=scores.get)})
    cfg = json.loads((run/"run.json").read_text())
    meta = json.loads((output/"model_manifest.json").read_text())
    X,Y,manifest = load_dataset(cfg["dataset"],"test")
    if meta["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Final models and test dataset differ")
    probabilities = np.zeros_like(Y,dtype=float)
    for j,item in enumerate(meta["models"]):
        if digest(output/item["filename"]) != item["sha256"]:
            raise ValueError("Model integrity failure")
        model = joblib.load(output/item["filename"])
        probabilities[:,j] = model.predict_proba(X)[:,1]
        del model
    metrics = evaluate(Y,probabilities,np.asarray([m["threshold"] for m in meta["models"]]),meta["labels"])
    metrics["evaluation_status"] = manifest["evaluation_status"]
    write_json(output/"test_metrics.json",metrics)
    np.savez_compressed(output/"test_predictions.npz",probabilities=probabilities,truth=Y)
    return metrics
