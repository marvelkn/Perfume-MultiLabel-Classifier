"""Final D-only Optuna continuation with interruption-safe budget checkpoints."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil
import time
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
import psutil

from . import experiments as core
from .data import ROOT, digest
from .experiments_v4 import paired_label_bootstrap, paired_seed_comparison
from .grouped_splits import POLICY
from .metrics import binary_metrics, summarize
from .perfume_data import OUTPUT

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

RUN_ID = "perfume-five-grouped-v5"
PROTOCOL = ROOT / "alignment/protocol_v5.json"
SEED_ROOT = ROOT / "v5_seed"
SEED_MANIFEST = SEED_ROOT / "seed_manifest.json"
V4_COMPLETE = SEED_ROOT / "v4_complete.json"
V4_SELECTION = SEED_ROOT / "v4_selection_frozen.json"
ALGORITHMS = ("xgb", "lgbm")
CANDIDATES = ("xgb_D", "lgbm_D")
INCUMBENTS = {"xgb": "xgb_D_v3", "lgbm": "lgbm_D_v3"}


def algorithm_for(key):
    return key.split("_", 1)[0]


def features(morgan, descriptors):
    return np.column_stack([morgan, descriptors])


def trial_state_counts(study):
    return dict(Counter(trial.state.name for trial in study.trials))


def preflight(dataset):
    meta, _ = core.preflight(dataset)
    protocol = core.read(PROTOCOL)
    if protocol.get("id") != RUN_ID or tuple(protocol["algorithms"]) != ALGORITHMS:
        raise ValueError("Use the locked grouped-v5 protocol.")
    if protocol.get("condition") != "D":
        raise ValueError("V5 is restricted to condition D.")
    if protocol["dataset_manifest_sha256"] != digest(Path(dataset) / "dataset_manifest.json"):
        raise ValueError("V5 protocol belongs to another dataset.")
    if protocol["environment"] != core.environment():
        raise ValueError("Dependency versions differ from the v5 protocol.")
    if protocol.get("group_policy") != POLICY:
        raise ValueError("Grouped split policy changed.")
    if protocol["class_imbalance"]["policy"] != core.WEIGHTING_POLICY:
        raise ValueError("Class weighting differs from grouped-v4.")
    tuning = protocol["tuning"]
    if tuning["additional_seconds_total"] != len(CANDIDATES) * tuning[
            "additional_seconds_per_study"]:
        raise ValueError("Both algorithms must receive the same fresh budget.")
    if not 5 <= tuning["heartbeat_seconds"] <= 60:
        raise ValueError("Budget heartbeat must be between 5 and 60 seconds.")

    seed = core.read(SEED_MANIFEST)
    if seed["source_run"] != "perfume-five-grouped-v4":
        raise ValueError("V5 seed is not grouped-v4.")
    for name, expected in seed["files"].items():
        if digest(SEED_ROOT / name) != expected:
            raise ValueError(f"V5 seed artifact changed: {name}")
    for key in CANDIDATES:
        study = optuna.load_study(
            study_name=key,
            storage=f"sqlite:///{(SEED_ROOT / key / 'optuna.sqlite3').as_posix()}",
        )
        expected = seed["studies"][key]
        states = trial_state_counts(study)
        if len(study.trials) != expected["database_trials"]:
            raise ValueError(f"{key}: seed database trial count changed.")
        if states.get("COMPLETE", 0) != expected["completed_trials"]:
            raise ValueError(f"{key}: seed completed-trial count changed.")
        if states.get("FAIL", 0) != expected["failed_trials"]:
            raise ValueError(f"{key}: seed failed-trial count changed.")
    v4_complete = core.read(V4_COMPLETE)
    v4_selection = core.read(V4_SELECTION)
    if v4_complete["selected_by_algorithm"] != INCUMBENTS:
        raise ValueError("V4 did not freeze the expected incumbents.")
    if v4_selection["test_used_for_selection"]:
        raise ValueError("V4 selection unexpectedly used test data.")
    for algorithm, incumbent in INCUMBENTS.items():
        folder = SEED_ROOT / incumbent
        if core.read(SEED_ROOT / f"{algorithm}_D/recipe.json") != core.read(
                folder / "recipe.json"):
            raise ValueError(f"{algorithm}: v4 D recipe differs from the incumbent.")
    return meta, protocol


def export_trials(study, path):
    study.trials_dataframe().to_csv(path, index=False)


def initialize_studies(run, protocol):
    marker = run / "seed_initialized.json"
    if marker.exists():
        value = core.read(marker)
        if value["seed_manifest_sha256"] != digest(SEED_MANIFEST):
            raise ValueError("Run was initialized from another v5 seed.")
        for key in CANDIDATES:
            required = ("optuna.sqlite3", "sampler.joblib", "trials.csv", "v5_budget.json")
            if any(not (run / key / name).is_file() for name in required):
                raise ValueError("Initialized v5 study is incomplete.")
        return

    seed = core.read(SEED_MANIFEST)
    for key in CANDIDATES:
        folder = run / key
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("optuna.sqlite3", "sampler.joblib"):
            destination = folder / name
            if destination.exists():
                if digest(destination) != seed["files"][f"{key}/{name}"]:
                    raise ValueError("Partial v5 seed initialization changed.")
            else:
                shutil.copy2(SEED_ROOT / key / name, destination)
        study = optuna.load_study(
            study_name=key,
            storage=f"sqlite:///{(folder / 'optuna.sqlite3').as_posix()}",
        )
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.RUNNING:
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        export_trials(study, folder / "trials.csv")
        core.write(folder / "v5_budget.json", {
            "budget_seconds": protocol["tuning"]["additional_seconds_per_study"],
            "charged_seconds": 0.0,
            "heartbeat_seconds": protocol["tuning"]["heartbeat_seconds"],
            "status": "initialized",
            "active_trial": None,
            "source_database_trials": seed["studies"][key]["database_trials"],
            "source_completed_trials": seed["studies"][key]["completed_trials"],
        })
    core.write(marker, {
        "source_run": "perfume-five-grouped-v4",
        "seed_manifest_sha256": digest(SEED_MANIFEST),
        "initialized_at": datetime.now(timezone.utc).isoformat(),
    })


class HeartbeatLimits(core.Limits):
    def __init__(self, deadline, heartbeat, interval):
        super().__init__(deadline)
        self.heartbeat = heartbeat
        self.interval = interval
        self.last_heartbeat = 0.0

    def check(self, force=False):
        super().check(force)
        now = time.perf_counter()
        if now - self.last_heartbeat >= self.interval:
            self.heartbeat(now)
            self.last_heartbeat = now


def tuning(run, key, algorithm, x, y, meta, split, positions, protocol):
    folder = run / key
    state_path = folder / "v5_budget.json"
    state = core.read(state_path)
    budget = protocol["tuning"]["additional_seconds_per_study"]
    if state["budget_seconds"] != budget:
        raise ValueError(f"{key}: v5 budget changed.")
    study = optuna.load_study(
        study_name=key,
        storage=f"sqlite:///{(folder / 'optuna.sqlite3').as_posix()}",
    )
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    sampler_path = folder / "sampler.joblib"
    study.sampler = joblib.load(sampler_path)
    export_trials(study, folder / "trials.csv")

    remaining = max(0.0, budget - float(state["charged_seconds"]))
    safety_max = protocol["tuning"]["safety_max_trials_per_study"]
    if remaining > 1 and len(study.trials) < safety_max:
        session_start = time.perf_counter()
        initial_charge = float(state["charged_seconds"])

        def heartbeat(now):
            state["charged_seconds"] = min(
                budget, initial_charge + max(0.0, now - session_start)
            )
            state["last_heartbeat_at"] = datetime.now(timezone.utc).isoformat()
            state["status"] = "running"
            core.write(state_path, state)

        limits = HeartbeatLimits(
            session_start + remaining,
            heartbeat,
            protocol["tuning"]["heartbeat_seconds"],
        )
        heartbeat(session_start)
        try:
            while len(study.trials) < safety_max and time.perf_counter() < limits.deadline:
                trial = study.ask()
                state["active_trial"] = trial.number
                heartbeat(time.perf_counter())
                try:
                    params = core.suggest(trial, algorithm)
                    values = core.cross_validate(
                        algorithm, params, x, y, meta["labels"], split["folds"],
                        positions, limits, meta["summary_labels"],
                    )
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
                    state["active_trial"] = None
                    joblib.dump(study.sampler, sampler_path)
                    export_trials(study, folder / "trials.csv")
                    heartbeat(time.perf_counter())
        finally:
            state["active_trial"] = None
            state["charged_seconds"] = min(
                budget, initial_charge + max(0.0, time.perf_counter() - session_start)
            )
            state["status"] = "complete" if state["charged_seconds"] >= budget - 1 else "paused"
            state["database_trials"] = len(study.trials)
            state["completed_trials"] = sum(
                trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
            )
            state["failed_trials"] = sum(
                trial.state == optuna.trial.TrialState.FAIL for trial in study.trials
            )
            state["stop_reason"] = (
                "time_budget_exhausted"
                if state["charged_seconds"] >= budget - 1
                else "interrupted_or_safety_limit"
            )
            core.write(state_path, state)
            joblib.dump(study.sampler, sampler_path)
            export_trials(study, folder / "trials.csv")

    complete = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete:
        raise RuntimeError(f"{key}: no completed trial.")
    best = max(complete, key=lambda trial: (trial.value, -trial.number))
    return best.user_attrs["resolved_parameters"]


def select_candidates(robustness):
    scores = {key: robustness[key]["mean_metrics"]["auprc"] for key in CANDIDATES}
    selected = {}
    for algorithm, incumbent in INCUMBENTS.items():
        incumbent_score = core.read(
            SEED_ROOT / incumbent / "robustness_summary.json"
        )["mean_metrics"]["auprc"]
        scores[incumbent] = incumbent_score
        candidate = f"{algorithm}_D"
        selected[algorithm] = max(
            (candidate, incumbent),
            key=lambda key: (scores[key], key == incumbent),
        )
    overall = max(
        selected.values(),
        key=lambda key: (scores[key], key in INCUMBENTS.values(), key),
    )
    return overall, selected, scores


def fit_final_models(folder, algorithm, params, x, y, labels):
    saved = core.read(folder / "models.json") if (folder / "models.json").exists() else {}
    for j, name in enumerate(labels):
        if name in saved:
            if digest(folder / saved[name]["file"]) != saved[name]["sha256"]:
                raise ValueError("Final-model checkpoint changed.")
            continue
        model = core.fit(algorithm, params, x, y[:, j], core.Limits())
        filename = f"label_{j:03d}.joblib"
        joblib.dump(model, folder / filename)
        saved[name] = {"file": filename, "sha256": digest(folder / filename)}
        core.write(folder / "models.json", saved)


def evaluate_test(run, frozen, key, algorithm, x, truth, labels):
    folder = run / key
    result_path = folder / "test_summary.json"
    if result_path.exists():
        result = core.read(result_path)
        if result["selection_sha256"] != digest(frozen):
            raise ValueError("Test result belongs to another selection.")
        for name, expected in result["artifact_sha256"].items():
            if digest(folder / name) != expected:
                raise ValueError("Saved v5 test artifact changed.")
        with np.load(folder / "test_predictions.npz", allow_pickle=False) as saved:
            probabilities = saved["probabilities"]
    else:
        models = core.read(folder / "models.json")
        probabilities = np.full(truth.shape, np.nan, dtype=np.float32)
        decisions = np.full(truth.shape, -1, dtype=np.int8)
        per_label = {}
        for j, name in enumerate(labels):
            entry = models[name]
            if digest(folder / entry["file"]) != entry["sha256"]:
                raise ValueError("Model changed after selection.")
            model = joblib.load(folder / entry["file"])
            probabilities[:, j] = model.predict_proba(x)[:, 1]
            decisions[:, j] = (probabilities[:, j] >= 0.5).astype(np.int8)
            per_label[name] = binary_metrics(truth[:, j], probabilities[:, j], decisions[:, j])
        core.save_npz(
            folder / "test_predictions.npz", probabilities=probabilities,
            decisions=decisions, truth=truth, labels=np.asarray(labels),
        )
        core.write(folder / "test_per_label.json", per_label)
        result = {
            "role": "final exploratory grouped-v5 comparison at threshold 0.5",
            "algorithm": algorithm,
            "summary": summarize(per_label, labels),
            "diagnostics": core.prediction_diagnostics(truth, decisions, labels),
            "selection_sha256": digest(frozen),
            "artifact_sha256": {
                name: digest(folder / name)
                for name in ("test_predictions.npz", "test_per_label.json")
            },
        }
        core.write(result_path, result)

    threshold_path = folder / "test_summary_validation_thresholds.json"
    if threshold_path.exists():
        threshold_result = core.read(threshold_path)
        if threshold_result["selection_sha256"] != digest(frozen):
            raise ValueError("Threshold result belongs to another selection.")
        for name, expected in threshold_result["artifact_sha256"].items():
            if digest(folder / name) != expected:
                raise ValueError("Saved v5 threshold artifact changed.")
    else:
        threshold_data = core.read(run / "robustness" / key / "thresholds.json")
        thresholds = np.asarray([
            threshold_data["labels"][name]["threshold"] for name in labels
        ])
        decisions = (probabilities >= thresholds[None, :]).astype(np.int8)
        per_label = {
            name: binary_metrics(truth[:, j], probabilities[:, j], decisions[:, j])
            for j, name in enumerate(labels)
        }
        core.save_npz(
            folder / "test_predictions_validation_thresholds.npz",
            probabilities=probabilities, decisions=decisions, truth=truth,
            labels=np.asarray(labels), thresholds=thresholds,
        )
        core.write(folder / "test_per_label_validation_thresholds.json", per_label)
        threshold_result = {
            "role": "secondary exploratory analysis; thresholds use validation only",
            "summary": summarize(per_label, labels),
            "diagnostics": core.prediction_diagnostics(truth, decisions, labels),
            "selection_sha256": digest(frozen),
            "threshold_source_sha256": digest(run / "robustness" / key / "thresholds.json"),
            "artifact_sha256": {
                name: digest(folder / name)
                for name in (
                    "test_predictions_validation_thresholds.npz",
                    "test_per_label_validation_thresholds.json",
                )
            },
        }
        core.write(threshold_path, threshold_result)
    return result, threshold_result


def run_all(dataset, run):
    core.check_host()
    dataset, run = Path(dataset).resolve(), Path(run).resolve()
    meta, protocol = preflight(dataset)
    run.mkdir(parents=True, exist_ok=True)
    identity = {
        "protocol": digest(PROTOCOL),
        "dataset_manifest": digest(dataset / "dataset_manifest.json"),
        "seed_manifest": digest(SEED_MANIFEST),
        "source": {str(path.relative_to(ROOT)): digest(path)
                   for path in (ROOT / "alignment").glob("*.py")},
        "environment": core.environment(),
        "compute_profile_sha256": digest(core.COMPUTE_PROFILE),
        "hardware": {
            "host": platform.node(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(logical=True),
            "ram_bytes": psutil.virtual_memory().total,
        },
    }
    identity_path = run / "identity.json"
    if identity_path.exists() and core.read(identity_path) != identity:
        raise ValueError("V5 source, seed, configuration, or environment changed.")
    if not identity_path.exists():
        core.write(identity_path, identity)

    with core.run_lock(run):
        initialize_studies(run, protocol)
        split = core.read(dataset / "splits.json")
        with np.load(dataset / "train.npz", allow_pickle=False) as values:
            y = values["truth"]
            x = features(values["morgan"], values["descriptors"])
            row_index = values["row_index"].astype(int)
            positions = {int(value): index for index, value in enumerate(row_index)}
        training_groups = pd.read_csv(dataset / "groups.csv").group_id.to_numpy()[row_index]

        recipes = {}
        for key in CANDIDATES:
            algorithm = algorithm_for(key)
            recipes[key] = tuning(
                run, key, algorithm, x, y, meta, split, positions, protocol
            )
            folder = run / key
            core.write(folder / "recipe.json", recipes[key])
            core.write(folder / "training_policy.json", {
                "class_imbalance": core.WEIGHTING_POLICY,
                "condition": "D",
                "selection_data": "training-validation only",
            })
            values = core.cross_validate(
                algorithm, recipes[key], x, y, meta["labels"], split["folds"],
                positions, core.Limits(), meta["labels"],
                folder / "cv_per_label.json", folder / "cv_fold_metrics.json",
                folder / "cv_oof_predictions.npz",
            )
            core.write(folder / "cv_summary.json", summarize(values, meta["labels"]))
            print(key, "tuning + primary CV complete", flush=True)

        robustness = {}
        for key in CANDIDATES:
            robustness[key] = core.repeated_grouped_cv(
                run, key, algorithm_for(key), recipes[key], x, y, meta["labels"],
                training_groups, protocol,
            )
            print(key, "repeated grouped CV complete", flush=True)

        frozen = run / "selection_frozen.json"
        if not frozen.exists():
            overall, selected, scores = select_candidates(robustness)
            names = (
                "recipe.json", "training_policy.json", "cv_per_label.json",
                "cv_fold_metrics.json", "cv_oof_predictions.npz", "cv_summary.json",
                "v5_budget.json", "trials.csv", "sampler.joblib", "optuna.sqlite3",
            )
            files = [run / key / name for key in CANDIDATES for name in names]
            files.extend(path for path in (run / "robustness").rglob("*") if path.is_file())
            core.write(frozen, {
                "protocol": RUN_ID,
                "selected_by_algorithm": selected,
                "overall_selected_from_repeated_grouped_cv": overall,
                "incumbents": INCUMBENTS,
                "validation_scores": scores,
                "selection_metric": "mean macro Average Precision across five grouped-CV seeds",
                "test_used_for_selection": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "files": {str(path.relative_to(run)): digest(path) for path in files},
            })
        freeze = core.read(frozen)
        for name, expected in freeze["files"].items():
            if digest(run / name) != expected:
                raise ValueError("Frozen grouped-v5 selection changed.")

        selected = freeze["selected_by_algorithm"]
        needs_test = any(key in CANDIDATES for key in selected.values())
        if needs_test:
            with np.load(dataset / "test.npz", allow_pickle=False) as values:
                test_truth = values["truth"]
                test_x = features(values["morgan"], values["descriptors"])

        test_summaries = {}
        threshold_summaries = {}
        per_label = {}
        selected_robustness = {}
        for algorithm in ALGORITHMS:
            key = selected[algorithm]
            if key in CANDIDATES:
                folder = run / key
                fit_final_models(folder, algorithm, recipes[key], x, y, meta["labels"])
                result, threshold_result = evaluate_test(
                    run, frozen, key, algorithm, test_x, test_truth, meta["labels"]
                )
                test_summaries[key] = result["summary"]
                threshold_summaries[key] = threshold_result["summary"]
                per_label[algorithm] = core.read(folder / "test_per_label.json")
                selected_robustness[algorithm] = robustness[key]
            else:
                folder = SEED_ROOT / key
                test_summaries[key] = core.read(folder / "test_summary.json")["summary"]
                threshold_summaries[key] = core.read(
                    folder / "test_summary_validation_thresholds.json"
                )["summary"]
                per_label[algorithm] = core.read(folder / "test_per_label.json")
                selected_robustness[algorithm] = core.read(
                    folder / "robustness_summary.json"
                )

        seed = core.read(SEED_MANIFEST)
        tuning_outcome = {}
        for key in CANDIDATES:
            study = optuna.load_study(
                study_name=key,
                storage=f"sqlite:///{(run / key / 'optuna.sqlite3').as_posix()}",
            )
            states = trial_state_counts(study)
            initial = seed["studies"][key]
            tuning_outcome[key] = {
                "source_completed_trials": initial["completed_trials"],
                "new_completed_trials": states.get("COMPLETE", 0) - initial["completed_trials"],
                "database_trials": len(study.trials),
                "completed_trials": states.get("COMPLETE", 0),
                "failed_trials": states.get("FAIL", 0),
                "v5_budget": core.read(run / key / "v5_budget.json"),
            }

        seed_comparison = paired_seed_comparison(
            selected_robustness["xgb"], selected_robustness["lgbm"]
        )
        label_bootstrap = paired_label_bootstrap(
            per_label["xgb"], per_label["lgbm"], meta["labels"]
        )
        core.write(run / "paired_seed_comparison.json", seed_comparison)
        core.write(run / "paired_label_bootstrap_test.json", label_bootstrap)
        core.write(run / "complete.json", {
            "protocol": RUN_ID,
            "selected_by_algorithm": selected,
            "overall_selected_from_repeated_grouped_cv": freeze[
                "overall_selected_from_repeated_grouped_cv"
            ],
            "tuning_outcome": tuning_outcome,
            "validation_scores": freeze["validation_scores"],
            "primary_test_threshold_0_5": test_summaries,
            "secondary_test_validation_thresholds": threshold_summaries,
            "paired_seed_comparison": seed_comparison,
            "paired_label_bootstrap_test": label_bootstrap,
            "test_used_for_selection": False,
            "confirmatory_status": protocol["confirmatory_status"],
            "stop_after_v5": protocol["stop_after_v5"],
            "compute_profile": core.compute_profile(),
        })
    return run / "complete.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUTPUT)
    parser.add_argument("--run", type=Path, default=ROOT / f"runs/{RUN_ID}")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        meta, protocol = preflight(args.dataset)
        print({
            "status": "V5_INPUTS_VERIFIED",
            "algorithms": list(ALGORITHMS),
            "condition": "D",
            "labels": len(meta["labels"]),
            "additional_seconds_per_study": protocol["tuning"][
                "additional_seconds_per_study"
            ],
            "additional_seconds_total": protocol["tuning"]["additional_seconds_total"],
            "heartbeat_seconds": protocol["tuning"]["heartbeat_seconds"],
            "training_started": False,
        })
    else:
        print(run_all(args.dataset, args.run))
