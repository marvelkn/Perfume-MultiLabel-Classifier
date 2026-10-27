"""Dynamic-label A-D experiments for the active five-source perfume dataset."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import optuna
import psutil
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import xgboost

from .data import ROOT, digest
from .perfume_data import OUTPUT, extract_features
from .grouped_splits import POLICY, inner_folds, structure_feature_groups, validate_grouped_split
from .metrics import METRICS, binary_metrics, summarize

ALGORITHMS = ("xgb", "lgbm")
CONDITIONS = ("A", "B", "C", "D")
PROTOCOL = ROOT / "alignment/protocol_v3.json"
COMPUTE_PROFILE = ROOT / "alignment/campus_compute_profile.json"
RUN_ID = "perfume-five-grouped-v3"
WEIGHTING_POLICY = "balanced-sample-weight-from-training-partition-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def environment():
    return {name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "rdkit", "xgboost", "lightgbm", "optuna", "joblib")}


def compute_profile():
    profile = read(COMPUTE_PROFILE)
    threads = profile.get("threads_per_model")
    if not isinstance(threads, int) or not 1 <= threads <= 64:
        raise ValueError("Invalid threads_per_model in campus compute profile.")
    if profile.get("execution_device") != "cpu":
        raise ValueError("This reproducible pipeline currently supports the CPU profile only.")
    return profile


def check_host():
    amendment = read(ROOT / "reports/campus_20260909/full/amendment.json")
    if platform.system() not in {"Windows", "Linux"}:
        raise RuntimeError("Campus execution supports Windows or Linux only.")
    if platform.node().casefold() == amendment["excluded_laptop_host"].casefold():
        raise RuntimeError("Campus execution is blocked on the original laptop.")


def protocol_path(dataset):
    return PROTOCOL


def preflight(dataset):
    dataset = Path(dataset)
    manifest = read(dataset / "dataset_manifest.json")
    if manifest.get("group_policy") != POLICY:
        raise ValueError("Legacy ungrouped dataset rejected. Use perfume-five-grouped-v2.")
    for name, expected in manifest["files"].items():
        if digest(dataset / name) != expected:
            raise ValueError(f"Dataset changed: {name}")
    protocol = read(protocol_path(dataset))
    if protocol["dataset_manifest_sha256"] != digest(dataset / "dataset_manifest.json"):
        raise ValueError("Protocol belongs to another dataset.")
    if protocol["environment"] != environment():
        raise ValueError("Dependency versions differ from the locked alignment protocol.")
    labels = manifest["labels"]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Invalid label registry.")
    if not set(manifest["summary_labels"]).issubset(labels):
        raise ValueError("Unknown summary label.")
    split = read(dataset / "splits.json")
    records = pd.read_csv(dataset / "records.csv")
    config = read(dataset / "config.json")
    if config["split"].get("group_policy") != POLICY or protocol.get("group_policy") != POLICY:
        raise ValueError("Grouped protocol/config mismatch.")
    if protocol.get("id") != RUN_ID:
        raise ValueError("Use the grouped-v3 experiment protocol.")
    if protocol.get("class_imbalance", {}).get("policy") != WEIGHTING_POLICY:
        raise ValueError("Class-imbalance policy is not the locked grouped-v3 policy.")
    tuning_policy = protocol.get("tuning", {})
    if tuning_policy.get("seconds_per_algorithm") != 2 * tuning_policy.get("seconds_per_study", -1):
        raise ValueError("Tuning budget must be identical for C and D within each algorithm.")
    if manifest["labels"] != split["eligible_labels"]:
        raise ValueError("Split and model label registries differ.")
    if records.smiles.duplicated().any():
        raise ValueError("Duplicate canonical molecules.")
    # Recompute from public structures, not test arrays or outcomes; do not trust supplied group IDs.
    all_morgan, all_desc = extract_features(records, config)
    groups = structure_feature_groups(records.smiles.tolist(), all_morgan)
    validate_grouped_split(split, groups)
    stored_groups = pd.read_csv(dataset / "groups.csv")
    if (stored_groups.group_id.tolist() != groups.tolist()
            or stored_groups.smiles.tolist() != records.smiles.tolist()
            or stored_groups.source_row.tolist() != records.source_row.tolist()):
        raise ValueError("Stored group registry differs from computed structures/features.")
    with np.load(dataset / "train.npz", allow_pickle=False) as values:
        y, x, d = values["truth"], values["morgan"], values["descriptors"]
        if y.ndim != 2 or y.shape[1] != len(labels) or y.shape[0] != len(split["train"]):
            raise ValueError("Training target shape differs from label metadata.")
        spec = manifest["feature_spec"]
        if x.shape != (len(y), spec["n_bits"]) or d.shape != (len(y), len(spec["descriptors"])):
            raise ValueError("Feature matrix shape differs from feature metadata.")
        if not np.isfinite(x).all() or not np.isfinite(d).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Invalid training features/targets.")
        if not np.array_equal(x, all_morgan[split["train"]]) or not np.array_equal(d, all_desc[split["train"]]):
            raise ValueError("Training features differ from recomputed structures.")
        if values["row_index"].tolist() != split["train"]:
            raise ValueError("Stored matrix row order differs from the split.")
    if set(split["train"]) & set(split["test"]):
        raise ValueError("Training/test row overlap.")
    positions = {i: k for k, i in enumerate(split["train"])}
    for label in split["eligible_labels"]:
        j = labels.index(label)
        seen = []
        for fold in split["folds"][label]:
            a, b = set(fold["train"]), set(fold["validation"])
            if a & b or a | b != set(split["train"]):
                raise ValueError("Invalid training/validation partition.")
            for indices in (a, b):
                if np.unique(y[[positions[i] for i in indices], j]).size != 2:
                    raise ValueError(f"Both classes required for fold label {label}.")
            seen.extend(fold["validation"])
        if sorted(seen) != sorted(split["train"]):
            raise ValueError("Validation folds must cover each training row exactly once.")
    return manifest, protocol


@contextmanager
def run_lock(run):
    path = run / ".alignment.lock"
    if path.exists():
        previous = read(path)
        same_host = previous.get("host", "").casefold() == platform.node().casefold()
        active = same_host and psutil.pid_exists(int(previous.get("pid", -1)))
        if active:
            raise RuntimeError("Another training process is still active for this run.")
        path.unlink()
    with path.open("x", encoding="utf-8") as f:
        json.dump({"host": platform.node(), "pid": os.getpid(), "created": time.time()}, f)
    try:
        yield
    finally:
        path.unlink()


class Limits:
    def __init__(self, deadline=None):
        self.deadline = deadline
        self.checked = 0

    def check(self, force=False):
        now = time.perf_counter()
        if self.deadline is not None and now >= self.deadline:
            raise TimeoutError("Tuning budget exhausted.")
        if not force and now - self.checked < 1:
            return
        self.checked = now
        if psutil.virtual_memory().available < 4 * 1024**3:
            raise RuntimeError("Less than 4 GiB system memory available.")
        if psutil.Process().memory_info().rss > 4 * 1024**3:
            raise RuntimeError("Training process exceeds 4 GiB RSS.")


def make_model(algorithm, parameters):
    params = dict(parameters)
    profile = compute_profile()
    threads = profile["threads_per_model"]
    if algorithm == "xgb":
        return XGBClassifier(n_jobs=threads, random_state=42, eval_metric="logloss",
                             tree_method=profile["xgboost_tree_method"], **params)
    if algorithm == "lgbm":
        return LGBMClassifier(n_jobs=threads, random_state=42, verbosity=-1, **params)
    raise ValueError("Unknown algorithm.")


def balanced_sample_weights(y):
    """Use the same fold-local weighting rule for XGBoost and LightGBM."""
    values = np.asarray(y, dtype=np.int8)
    if values.ndim != 1 or len(values) == 0 or not np.isin(values, [0, 1]).all():
        raise ValueError("Expected a non-empty binary target vector.")
    counts = np.bincount(values, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Balanced weighting requires both classes.")
    per_class = len(values) / (2.0 * counts)
    return per_class[values]


def fit(algorithm, parameters, x, y, limits):
    limits.check(force=True)
    if np.unique(y).size != 2:
        raise ValueError("Training requires both classes.")
    model = make_model(algorithm, parameters)
    weights = balanced_sample_weights(y)
    if algorithm == "xgb":
        class Callback(xgboost.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                limits.check()
                return False
        model.set_params(callbacks=[Callback()])
        model.fit(x, y, sample_weight=weights, verbose=False)
        model.set_params(callbacks=None)
    else:
        def callback(env):
            limits.check()
        callback.order = 5
        model.fit(x, y, sample_weight=weights, callbacks=[callback])
    limits.check(force=True)
    return model


def suggest(trial, algorithm):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
    }
    if algorithm == "xgb":
        params["min_child_weight"] = trial.suggest_float("min_child_weight", 1, 30, log=True)
        params["gamma"] = trial.suggest_float("gamma", 0, 5)
    else:
        params["num_leaves"] = trial.suggest_int("num_leaves", 8, min(128, 2**params["max_depth"]))
        params["min_child_samples"] = trial.suggest_int("min_child_samples", 10, 150)
        params["subsample_freq"] = 1
    return params


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def cross_validate(algorithm, params, x, y, labels, folds, positions, limits, wanted,
                   checkpoint=None, detail_path=None, oof_path=None):
    """Evaluate per label; optional artifacts make the final CV fully auditable."""
    result = read(checkpoint) if checkpoint and checkpoint.exists() else {}
    details = read(detail_path) if detail_path and detail_path.exists() else {}
    oof_probabilities = np.full(y.shape, np.nan, dtype=np.float32)
    oof_decisions = np.full(y.shape, -1, dtype=np.int8)
    if oof_path and oof_path.exists():
        with np.load(oof_path, allow_pickle=False) as saved:
            oof_probabilities = saved["probabilities"]
            oof_decisions = saved["decisions"]
        if oof_probabilities.shape != y.shape or oof_decisions.shape != y.shape:
            raise ValueError("OOF checkpoint shape changed.")

    for name in wanted:
        j = labels.index(name)
        artifacts_ready = (
            (detail_path is None or name in details)
            and (oof_path is None or np.isfinite(oof_probabilities[:, j]).all())
        )
        if name in result and artifacts_ready:
            continue
        fold_values = []
        fold_details = []
        oof_probabilities[:, j] = np.nan
        oof_decisions[:, j] = -1
        for fold_index, fold in enumerate(folds[name]):
            tr = np.array([positions[i] for i in fold["train"]], dtype=int)
            va = np.array([positions[i] for i in fold["validation"]], dtype=int)
            model = fit(algorithm, params, x[tr], y[tr, j], limits)
            probabilities = model.predict_proba(x[va])[:, 1]
            decisions = (probabilities >= 0.5).astype(np.int8)
            metrics = binary_metrics(y[va, j], probabilities, decisions)
            fold_values.append(metrics)
            fold_details.append({
                "fold": fold_index,
                "training_rows": int(len(tr)),
                "validation_rows": int(len(va)),
                "training_positives": int(y[tr, j].sum()),
                "validation_positives": int(y[va, j].sum()),
                "metrics": metrics,
            })
            oof_probabilities[va, j] = probabilities
            oof_decisions[va, j] = decisions
        if not np.isfinite(oof_probabilities[:, j]).all() and oof_path:
            raise ValueError(f"OOF coverage incomplete for label {name}.")
        result[name] = {
            metric: float(np.mean([value[metric] for value in fold_values]))
            for metric in METRICS
        }
        details[name] = fold_details
        if checkpoint:
            write(checkpoint, result)
        if detail_path:
            write(detail_path, details)
        if oof_path:
            save_npz(oof_path, probabilities=oof_probabilities, decisions=oof_decisions,
                     truth=y, labels=np.asarray(labels))
    return result


def f1_from_metrics(value):
    precision, recall = value["precision"], value["recall"]
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def prediction_diagnostics(truth, decisions, labels):
    per_label = [
        binary_metrics(truth[:, j], decisions[:, j].astype(float), decisions[:, j])
        for j in range(len(labels))
    ]
    return {
        "mean_true_labels_per_molecule": float(np.mean(np.sum(truth, axis=1))),
        "mean_predicted_labels_per_molecule": float(np.mean(np.sum(decisions, axis=1))),
        "true_positive_rate": float(np.mean(truth)),
        "predicted_positive_rate": float(np.mean(decisions)),
        "macro_f1": float(np.mean([f1_from_metrics(value) for value in per_label])),
        "label_count": len(labels),
    }


def select_validation_thresholds(truth, probabilities, labels, grid):
    """Choose each label threshold from OOF validation predictions only."""
    thresholds = {}
    fixed_grid = sorted({float(value) for value in grid} | {0.5})
    for j, name in enumerate(labels):
        candidates = []
        for threshold in fixed_grid:
            decisions = (probabilities[:, j] >= threshold).astype(np.int8)
            value = binary_metrics(truth[:, j], probabilities[:, j], decisions)
            candidates.append((f1_from_metrics(value), -abs(threshold - 0.5), threshold, value))
        best_f1, _, threshold, metrics = max(candidates, key=lambda item: item[:3])
        at_half = binary_metrics(
            truth[:, j], probabilities[:, j], (probabilities[:, j] >= 0.5).astype(np.int8)
        )
        thresholds[name] = {
            "threshold": threshold,
            "oof_f1": best_f1,
            "oof_f1_at_0_5": f1_from_metrics(at_half),
            "oof_precision": metrics["precision"],
            "oof_recall": metrics["recall"],
            "training_positives": int(truth[:, j].sum()),
        }
    return {
        "method": "per-label maximum F1 on repeated grouped OOF probabilities",
        "selection_data": "training-validation only; held-out test is never used",
        "grid": fixed_grid,
        "labels": thresholds,
    }


def repeated_grouped_cv(run, key, algorithm, params, x, y, labels, groups, protocol):
    """Compare the two algorithm finalists across fixed grouped-CV repetitions."""
    folder = run / "robustness" / key
    folder.mkdir(parents=True, exist_ok=True)
    seeds = protocol["robustness"]["seeds"]
    max_folds = protocol["robustness"]["folds"]
    local_indices = np.arange(len(y), dtype=int)
    positions = {int(i): int(i) for i in local_indices}
    summaries, probability_runs = {}, []
    for seed in seeds:
        seed_folder = folder / f"seed_{seed}"
        seed_folder.mkdir(exist_ok=True)
        generated = {
            name: inner_folds(local_indices, y[:, j], groups, max_folds, seed)
            for j, name in enumerate(labels)
        }
        write(seed_folder / "folds.json", {
            "seed": seed,
            "group_policy": POLICY,
            "folds": generated,
        })
        values = cross_validate(
            algorithm, params, x, y, labels, generated, positions, Limits(), labels,
            seed_folder / "cv_per_label.json",
            seed_folder / "cv_fold_metrics.json",
            seed_folder / "cv_oof_predictions.npz",
        )
        summary = summarize(values, labels)
        summaries[str(seed)] = summary
        write(seed_folder / "cv_summary.json", summary)
        with np.load(seed_folder / "cv_oof_predictions.npz", allow_pickle=False) as saved:
            probability_runs.append(saved["probabilities"])

    pooled = np.mean(np.stack(probability_runs), axis=0)
    decisions = (pooled >= 0.5).astype(np.int8)
    save_npz(folder / "pooled_oof_predictions.npz", probabilities=pooled,
             decisions=decisions, truth=y, labels=np.asarray(labels),
             seeds=np.asarray(seeds, dtype=int))
    thresholds = select_validation_thresholds(
        y, pooled, labels, protocol["threshold_selection"]["grid"]
    )
    write(folder / "thresholds.json", thresholds)
    mean_metrics = {
        metric: float(np.mean([summary["metrics"][metric] for summary in summaries.values()]))
        for metric in METRICS
    }
    std_metrics = {
        metric: float(np.std([summary["metrics"][metric] for summary in summaries.values()], ddof=1))
        for metric in METRICS
    }
    result = {
        "candidate": key,
        "seeds": seeds,
        "folds_per_label_maximum": max_folds,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "per_seed": summaries,
        "oof_diagnostics_at_0_5": prediction_diagnostics(y, decisions, labels),
        "weighting_policy": WEIGHTING_POLICY,
    }
    write(folder / "robustness_summary.json", result)
    return result


def tuning(run, key, algorithm, x, y, meta, split, positions, protocol):
    folder = run / key
    state_path = folder / "tuning_budget.json"
    state = read(state_path) if state_path.exists() else {"charged_seconds": 0, "active_reservation": 0}
    # A killed process cannot silently reclaim time it may have consumed.
    state["charged_seconds"] += state.get("active_reservation", 0)
    state["active_reservation"] = 0
    write(state_path, state)
    study = optuna.create_study(
        study_name=key, storage=f"sqlite:///{(folder / 'optuna.sqlite3').as_posix()}",
        direction="maximize", load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=42))
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    sampler_path = folder / "sampler.joblib"
    if sampler_path.exists():
        study.sampler = joblib.load(sampler_path)
    budget = protocol["tuning"]["seconds_per_study"]
    safety_max = protocol["tuning"].get("safety_max_trials_per_study")
    remaining = max(0.0, budget - state["charged_seconds"])
    started = time.perf_counter()
    under_safety_limit = lambda: safety_max is None or len(study.trials) < safety_max
    if remaining > 1 and under_safety_limit():
        state["active_reservation"] = remaining
        state["stop_rule"] = "elapsed active compute time"
        write(state_path, state)
        limits = Limits(started + remaining)
        try:
            while under_safety_limit():
                if time.perf_counter() >= limits.deadline:
                    break
                trial = study.ask()
                try:
                    params = suggest(trial, algorithm)
                    values = cross_validate(algorithm, params, x, y, meta["labels"], split["folds"],
                                            positions, limits, meta["summary_labels"])
                    score = summarize(values, meta["summary_labels"])["metrics"]["auprc"]
                    trial.set_user_attr("resolved_parameters", params)
                    study.tell(trial, score)
                except TimeoutError:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    break
                except BaseException:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    raise
                finally:
                    joblib.dump(study.sampler, sampler_path)
                    study.trials_dataframe().to_csv(folder / "trials.csv", index=False)
        finally:
            state["charged_seconds"] = min(budget, state["charged_seconds"] + time.perf_counter() - started)
            state["active_reservation"] = 0
            state["attempted_trials"] = len(study.trials)
            state["completed_trials"] = sum(
                trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
            )
            state["stop_reason"] = (
                "time_budget_exhausted"
                if state["charged_seconds"] >= budget - 1
                else "safety_trial_limit_reached"
            )
            write(state_path, state)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError(f"{key}: no completed trial; do not label an untuned fallback as tuned.")
    best = max(complete, key=lambda t: (t.value, -t.number))
    return best.user_attrs["resolved_parameters"]


def run_all(dataset, run):
    check_host()
    dataset, run = Path(dataset).resolve(), Path(run).resolve()
    meta, protocol = preflight(dataset)
    run.mkdir(parents=True, exist_ok=True)
    source = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "alignment").glob("*.py")}
    identity = {"dataset_manifest": digest(dataset / "dataset_manifest.json"),
                "protocol": digest(protocol_path(dataset)), "source": source, "environment": environment(),
                "compute_profile_sha256": digest(COMPUTE_PROFILE), "compute_profile": compute_profile(),
                "hardware": {"host": platform.node(), "system": platform.system(),
                             "processor": platform.processor(), "physical_cores": psutil.cpu_count(logical=False),
                             "logical_cpus": psutil.cpu_count(logical=True),
                             "ram_bytes": psutil.virtual_memory().total}}
    identity_path = run / "identity.json"
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError("Run source/configuration/environment changed; do not resume.")
    if not identity_path.exists():
        write(identity_path, identity)
    with run_lock(run):
        split = read(dataset / "splits.json")
        with np.load(dataset / "train.npz", allow_pickle=False) as values:
            y, morgan, desc = values["truth"], values["morgan"], values["descriptors"]
            row_index = values["row_index"].astype(int)
            positions = {int(i): k for k, i in enumerate(row_index)}
        all_groups = pd.read_csv(dataset / "groups.csv").group_id.to_numpy()
        training_groups = all_groups[row_index]
        summaries = {}
        frozen = run / "selection_frozen.json"
        if not frozen.exists():
            for algorithm in ALGORITHMS:
                for condition in CONDITIONS:
                    key = f"{algorithm}_{condition}"
                    folder = run / key
                    folder.mkdir(exist_ok=True)
                    x = morgan if condition in ("A", "C") else np.column_stack([morgan, desc])
                    if (folder / "recipe.json").exists():
                        params = read(folder / "recipe.json")
                    else:
                        params = {"n_estimators": 100} if condition in ("A", "B") else tuning(
                            run, key, algorithm, x, y, meta, split, positions, protocol)
                        write(folder / "recipe.json", params)
                    policy = {
                        "class_imbalance": WEIGHTING_POLICY,
                        "weight_source": "each training fold only during CV; full training data for final fit",
                        "decision_threshold_primary": 0.5,
                    }
                    policy_path = folder / "training_policy.json"
                    if policy_path.exists() and read(policy_path) != policy:
                        raise ValueError("Training policy checkpoint changed.")
                    write(policy_path, policy)
                    limits = Limits()
                    values = cross_validate(
                        algorithm, params, x, y, meta["labels"], split["folds"], positions,
                        limits, split["eligible_labels"], folder / "cv_per_label.json",
                        folder / "cv_fold_metrics.json", folder / "cv_oof_predictions.npz",
                    )
                    summary = summarize(values, meta["summary_labels"])
                    summaries[key] = summary
                    write(folder / "cv_summary.json", summary)
                    saved = read(folder / "models.json") if (folder / "models.json").exists() else {}
                    for name in split["eligible_labels"]:
                        if name in saved:
                            if digest(folder / saved[name]["file"]) != saved[name]["sha256"]:
                                raise ValueError("Model checkpoint changed.")
                            continue
                        j = meta["labels"].index(name)
                        model = fit(algorithm, params, x, y[:, j], limits)
                        filename = f"label_{j:03d}.joblib"
                        joblib.dump(model, folder / filename)
                        saved[name] = {"file": filename, "sha256": digest(folder / filename)}
                        write(folder / "models.json", saved)
                    print(key, "CV + final fit complete", flush=True)

            finalists = {
                algorithm: max(
                    (f"{algorithm}_{condition}" for condition in CONDITIONS),
                    key=lambda key: (summaries[key]["metrics"]["auprc"], key),
                )
                for algorithm in ALGORITHMS
            }
            robustness = {}
            for algorithm, key in finalists.items():
                condition = key.split("_", 1)[1]
                x = morgan if condition in ("A", "C") else np.column_stack([morgan, desc])
                robustness[key] = repeated_grouped_cv(
                    run, key, algorithm, read(run / key / "recipe.json"), x, y,
                    meta["labels"], training_groups, protocol,
                )
                print(key, "repeated grouped CV complete", flush=True)
            selected = max(
                finalists.values(),
                key=lambda key: (robustness[key]["mean_metrics"]["auprc"], key),
            )
            candidate_names = (
                "recipe.json", "training_policy.json", "cv_per_label.json",
                "cv_fold_metrics.json", "cv_oof_predictions.npz", "cv_summary.json", "models.json",
            )
            files = [
                run / f"{algorithm}_{condition}" / name
                for algorithm in ALGORITHMS for condition in CONDITIONS for name in candidate_names
            ]
            files.extend(
                path for path in (run / "robustness").rglob("*")
                if path.is_file() and path.name != ".alignment.lock"
            )
            write(frozen, {
                "protocol": RUN_ID,
                "finalists_from_primary_validation": finalists,
                "selected_from_repeated_grouped_cv": selected,
                "selection_metric": "macro Average Precision",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "files": {str(path.relative_to(run)): digest(path) for path in files},
            })
        freeze = read(frozen)
        for name, expected in freeze["files"].items():
            if digest(run / name) != expected:
                raise ValueError("Frozen selection changed.")
        # First access to held-out arrays is after all eight candidates are frozen.
        with np.load(dataset / "test.npz", allow_pickle=False) as values:
            yt, mt, dt = values["truth"], values["morgan"], values["descriptors"]
        test_summaries = {}
        for algorithm in ALGORITHMS:
            for condition in CONDITIONS:
                key = f"{algorithm}_{condition}"
                folder = run / key
                result_path = folder / "test_summary.json"
                if result_path.exists():
                    result = read(result_path)
                    if result["selection_sha256"] != digest(frozen):
                        raise ValueError("Test result belongs to another selection.")
                    for name, expected in result["artifact_sha256"].items():
                        if digest(folder / name) != expected:
                            raise ValueError("Saved test artifact changed.")
                    test_summaries[key] = result["summary"]
                    continue
                x = mt if condition in ("A", "C") else np.column_stack([mt, dt])
                models = read(folder / "models.json")
                per_label = {name: dict.fromkeys(METRICS) for name in meta["labels"]}
                p = np.full(yt.shape, np.nan)
                decisions = np.full(yt.shape, -1, dtype=np.int8)
                for name, entry in models.items():
                    if digest(folder / entry["file"]) != entry["sha256"]:
                        raise ValueError("Model changed since final fitting.")
                    model = joblib.load(folder / entry["file"])
                    j = meta["labels"].index(name)
                    p[:, j] = model.predict_proba(x)[:, 1]
                    decisions[:, j] = model.predict(x)
                    per_label[name] = binary_metrics(yt[:, j], p[:, j], decisions[:, j])
                save_npz(folder / "test_predictions.npz", probabilities=p, decisions=decisions,
                         truth=yt, labels=np.asarray(meta["labels"]))
                write(folder / "test_per_label.json", per_label)
                summary = summarize(per_label, meta["summary_labels"])
                write(result_path, {
                    "role": "primary Suh-aligned comparison at fixed threshold 0.5",
                    "summary": summary,
                    "diagnostics": prediction_diagnostics(yt, decisions, meta["labels"]),
                    "selection_sha256": digest(frozen),
                    "artifact_sha256": {
                        name: digest(folder / name)
                        for name in ("test_predictions.npz", "test_per_label.json")
                    },
                })
                test_summaries[key] = summary

        threshold_summaries = {}
        for algorithm, key in freeze["finalists_from_primary_validation"].items():
            folder = run / key
            threshold_source = run / "robustness" / key / "thresholds.json"
            threshold_data = read(threshold_source)
            thresholds = np.asarray([
                threshold_data["labels"][name]["threshold"] for name in meta["labels"]
            ])
            with np.load(folder / "test_predictions.npz", allow_pickle=False) as saved:
                probabilities, truth = saved["probabilities"], saved["truth"]
            decisions = (probabilities >= thresholds[None, :]).astype(np.int8)
            per_label = {
                name: binary_metrics(truth[:, j], probabilities[:, j], decisions[:, j])
                for j, name in enumerate(meta["labels"])
            }
            save_npz(
                folder / "test_predictions_validation_thresholds.npz",
                probabilities=probabilities, decisions=decisions, truth=truth,
                labels=np.asarray(meta["labels"]), thresholds=thresholds,
            )
            write(folder / "test_per_label_validation_thresholds.json", per_label)
            result = {
                "role": "secondary sensitivity analysis; thresholds selected without test data",
                "summary": summarize(per_label, meta["summary_labels"]),
                "diagnostics": prediction_diagnostics(truth, decisions, meta["labels"]),
                "threshold_source_sha256": digest(threshold_source),
                "selection_sha256": digest(frozen),
            }
            write(folder / "test_summary_validation_thresholds.json", result)
            threshold_summaries[key] = result

        write(run / "complete.json", {
            "protocol": RUN_ID,
            "finalists_from_primary_validation": freeze["finalists_from_primary_validation"],
            "selected_from_repeated_grouped_cv": freeze["selected_from_repeated_grouped_cv"],
            "primary_test_threshold_0_5": test_summaries,
            "secondary_test_validation_thresholds": threshold_summaries,
            "confirmatory_status": protocol["confirmatory_status"],
            "onnx_exported": False,
            "mobile_integration_validated": False,
            "compute_profile": compute_profile(),
        })
    return run / "complete.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUTPUT)
    parser.add_argument("--run", type=Path, default=ROOT / f"runs/{RUN_ID}")
    parser.add_argument("--check", action="store_true", help="Validate inputs only, without loading test or fitting.")
    args = parser.parse_args()
    if args.check:
        meta, protocol = preflight(args.dataset)
        print(json.dumps({"status": "INPUTS_VERIFIED", "labels": len(meta["labels"]),
                          "summary_labels": len(meta["summary_labels"]), "training_started": False}))
    else:
        print(run_all(args.dataset, args.run))
