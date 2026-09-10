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
from .grouped_splits import POLICY, structure_feature_groups, validate_grouped_split
from .metrics import METRICS, binary_metrics, summarize

ALGORITHMS = ("xgb", "lgbm")
CONDITIONS = ("A", "B", "C", "D")
PROTOCOL = ROOT / "alignment/protocol.json"


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


def check_host():
    amendment = read(ROOT / "reports/campus_20260909/full/amendment.json")
    if platform.system() not in {"Windows", "Linux"}:
        raise RuntimeError("Campus execution supports Windows or Linux only.")
    if platform.node().casefold() == amendment["excluded_laptop_host"].casefold():
        raise RuntimeError("Campus execution is blocked on the original laptop.")


def protocol_path(dataset):
    local = Path(dataset) / "experiment_protocol.json"
    return local if local.exists() else PROTOCOL


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
    if algorithm == "xgb":
        return XGBClassifier(n_jobs=2, random_state=42, eval_metric="logloss", **params)
    if algorithm == "lgbm":
        return LGBMClassifier(n_jobs=2, random_state=42, class_weight="balanced", verbosity=-1, **params)
    raise ValueError("Unknown algorithm.")


def fit(algorithm, parameters, x, y, limits):
    limits.check(force=True)
    if np.unique(y).size != 2:
        raise ValueError("Training requires both classes.")
    model = make_model(algorithm, parameters)
    if algorithm == "xgb":
        class Callback(xgboost.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                limits.check()
                return False
        model.set_params(callbacks=[Callback()])
        model.fit(x, y, verbose=False)
        model.set_params(callbacks=None)
    else:
        def callback(env):
            limits.check()
        callback.order = 5
        model.fit(x, y, callbacks=[callback])
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


def cross_validate(algorithm, params, x, y, labels, folds, positions, limits, wanted, checkpoint=None):
    result = read(checkpoint) if checkpoint and checkpoint.exists() else {}
    for name in wanted:
        if name in result:
            continue
        j = labels.index(name)
        values = []
        for fold in folds[name]:
            tr = np.array([positions[i] for i in fold["train"]])
            va = np.array([positions[i] for i in fold["validation"]])
            model = fit(algorithm, params, x[tr], y[tr, j], limits)
            values.append(binary_metrics(y[va, j], model.predict_proba(x[va])[:, 1], model.predict(x[va])))
        result[name] = {metric: float(np.mean([v[metric] for v in values])) for metric in METRICS}
        if checkpoint:
            write(checkpoint, result)
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
    remaining = protocol["tuning"]["seconds_per_study"] - state["charged_seconds"]
    started = time.perf_counter()
    if remaining > 1 and len(study.trials) < protocol["tuning"]["attempts_per_study"]:
        state["active_reservation"] = remaining
        write(state_path, state)
        limits = Limits(started + remaining)
        try:
            while len(study.trials) < protocol["tuning"]["attempts_per_study"]:
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
            state["charged_seconds"] += time.perf_counter() - started
            state["active_reservation"] = 0
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
                "protocol": digest(protocol_path(dataset)), "source": source, "environment": environment()}
    identity_path = run / "identity.json"
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError("Run source/configuration/environment changed; do not resume.")
    if not identity_path.exists():
        write(identity_path, identity)
    with run_lock(run):
        split = read(dataset / "splits.json")
        with np.load(dataset / "train.npz", allow_pickle=False) as values:
            y, morgan, desc = values["truth"], values["morgan"], values["descriptors"]
            positions = {int(i): k for k, i in enumerate(values["row_index"])}
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
                    limits = Limits()
                    values = cross_validate(algorithm, params, x, y, meta["labels"], split["folds"], positions,
                                            limits, split["eligible_labels"], folder / "cv_per_label.json")
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
            selected = max(summaries, key=lambda key: (summaries[key]["metrics"]["auprc"], key))
            files = [run / f"{a}_{c}" / name for a in ALGORITHMS for c in CONDITIONS
                     for name in ("recipe.json", "cv_per_label.json", "cv_summary.json", "models.json")]
            write(frozen, {"selected_from_validation": selected, "timestamp": datetime.now(timezone.utc).isoformat(),
                           "files": {str(p.relative_to(run)): digest(p) for p in files}})
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
                np.savez_compressed(folder / "test_predictions.npz", probabilities=p, decisions=decisions,
                                    truth=yt, labels=np.asarray(meta["labels"]))
                write(folder / "test_per_label.json", per_label)
                summary = summarize(per_label, meta["summary_labels"])
                write(result_path, {"summary": summary, "selection_sha256": digest(frozen),
                                    "artifact_sha256": {n: digest(folder / n) for n in
                                                        ("test_predictions.npz", "test_per_label.json")}})
                test_summaries[key] = summary
        write(run / "complete.json", {"selected_from_validation": freeze["selected_from_validation"],
                                      "test": test_summaries, "onnx_exported": False,
                                      "mobile_integration_validated": False})
    return run / "complete.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUTPUT)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/perfume-five-grouped-v2")
    parser.add_argument("--check", action="store_true", help="Validate inputs only, without loading test or fitting.")
    args = parser.parse_args()
    if args.check:
        meta, protocol = preflight(args.dataset)
        print(json.dumps({"status": "INPUTS_VERIFIED", "labels": len(meta["labels"]),
                          "summary_labels": len(meta["summary_labels"]), "training_started": False}))
    else:
        print(run_all(args.dataset, args.run))
