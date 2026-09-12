"""Fair Optuna continuation for XGBoost and LightGBM after grouped-v3."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil

import joblib
import numpy as np
import pandas as pd
import psutil

from . import experiments as core
from .data import ROOT, digest
from .grouped_splits import POLICY
from .metrics import binary_metrics, summarize
from .perfume_data import OUTPUT

RUN_ID = "perfume-five-grouped-v4"
PROTOCOL = ROOT / "alignment/protocol_v4.json"
SEED_ROOT = ROOT / "v4_seed"
SEED_MANIFEST = SEED_ROOT / "seed_manifest.json"
V3_REFERENCE = SEED_ROOT / "v3_reference.json"
ALGORITHMS = ("xgb", "lgbm")
CANDIDATES = ("xgb_C", "xgb_D", "lgbm_C", "lgbm_D")
INCUMBENTS = {"xgb": "xgb_D_v3", "lgbm": "lgbm_D_v3"}


def algorithm_for(key):
    return key.split("_", 1)[0]


def condition_for(key):
    return key.split("_", 1)[1]


def features_for(condition, morgan, descriptors):
    return morgan if condition == "C" else np.column_stack([morgan, descriptors])


def preflight(dataset):
    meta, _ = core.preflight(dataset)
    protocol = core.read(PROTOCOL)
    if protocol.get("id") != RUN_ID or tuple(protocol["algorithms"]) != ALGORITHMS:
        raise ValueError("Use the symmetric grouped-v4 continuation protocol.")
    if protocol["dataset_manifest_sha256"] != digest(Path(dataset) / "dataset_manifest.json"):
        raise ValueError("V4 protocol belongs to another dataset.")
    if protocol["environment"] != core.environment():
        raise ValueError("Dependency versions differ from the v4 protocol.")
    if protocol.get("group_policy") != POLICY:
        raise ValueError("Grouped split policy changed.")
    if protocol["class_imbalance"]["policy"] != core.WEIGHTING_POLICY:
        raise ValueError("V4 weighting differs from grouped-v3.")
    tuning = protocol["tuning"]
    if tuning["seconds_per_study"] != (
            tuning["initial_charged_seconds"] + tuning["additional_seconds_per_study"]):
        raise ValueError("V4 continuation budget is inconsistent.")
    if tuning["seconds_per_algorithm"] != 2 * tuning["seconds_per_study"]:
        raise ValueError("C and D must receive the same budget within each algorithm.")
    if tuning["additional_seconds_total"] != len(CANDIDATES) * tuning[
            "additional_seconds_per_study"]:
        raise ValueError("XGBoost and LightGBM must receive the same added budget.")

    seed = core.read(SEED_MANIFEST)
    if seed["source_run"] != "perfume-five-grouped-v3":
        raise ValueError("Warm-start seed is not grouped-v3.")
    for name, expected in seed["files"].items():
        if digest(SEED_ROOT / name) != expected:
            raise ValueError(f"Warm-start artifact changed: {name}")
    for key in CANDIDATES:
        state = core.read(SEED_ROOT / key / "tuning_budget.json")
        trials = pd.read_csv(SEED_ROOT / key / "trials.csv")
        completed = int((trials.state == "COMPLETE").sum())
        if state["charged_seconds"] != tuning["initial_charged_seconds"]:
            raise ValueError(f"{key}: unexpected initial charged time.")
        if completed != seed["studies"][key]["completed_trials"]:
            raise ValueError(f"{key}: trial registry differs from the seed manifest.")

    reference = core.read(V3_REFERENCE)
    for algorithm, incumbent_key in INCUMBENTS.items():
        expected = reference["incumbents"][incumbent_key]
        folder = SEED_ROOT / incumbent_key
        if core.read(folder / "recipe.json") != expected["recipe"]:
            raise ValueError(f"{incumbent_key}: recipe differs from the frozen reference.")
        robustness = core.read(folder / "robustness_summary.json")
        test = core.read(folder / "test_summary.json")
        if robustness["mean_metrics"]["auprc"] != expected["repeated_cv_mean_auprc"]:
            raise ValueError(f"{incumbent_key}: validation score changed.")
        if test["summary"]["metrics"]["auprc"] != expected["test_auprc"]:
            raise ValueError(f"{incumbent_key}: test score changed.")
        if algorithm_for(incumbent_key) != algorithm:
            raise ValueError("Incumbent registry is inconsistent.")
    return meta, protocol


def initialize_studies(run):
    marker = run / "seed_initialized.json"
    required = ("optuna.sqlite3", "sampler.joblib", "trials.csv", "tuning_budget.json")
    if marker.exists():
        value = core.read(marker)
        if value["seed_manifest_sha256"] != digest(SEED_MANIFEST):
            raise ValueError("Run was initialized from another warm-start seed.")
        for key in CANDIDATES:
            if any(not (run / key / name).is_file() for name in required):
                raise ValueError("Initialized Optuna study is incomplete.")
        return
    if any((run / key).exists() for key in CANDIDATES):
        raise ValueError("Partial seed initialization; use a new empty grouped-v4 run folder.")
    for key in CANDIDATES:
        folder = run / key
        folder.mkdir(parents=True)
        for name in required:
            shutil.copy2(SEED_ROOT / key / name, folder / name)
    core.write(marker, {
        "source_run": "perfume-five-grouped-v3",
        "seed_manifest_sha256": digest(SEED_MANIFEST),
        "initialized_at": datetime.now(timezone.utc).isoformat(),
    })


def choose_candidates(robustness, reference):
    scores = {key: robustness[key]["mean_metrics"]["auprc"] for key in CANDIDATES}
    for incumbent in INCUMBENTS.values():
        scores[incumbent] = reference["incumbents"][incumbent]["repeated_cv_mean_auprc"]
    best_extended = {}
    selected_by_algorithm = {}
    for algorithm in ALGORITHMS:
        extended = tuple(key for key in CANDIDATES if algorithm_for(key) == algorithm)
        best_extended[algorithm] = max(extended, key=lambda key: (scores[key], key))
        incumbent = INCUMBENTS[algorithm]
        selected_by_algorithm[algorithm] = max(
            (best_extended[algorithm], incumbent),
            key=lambda key: (scores[key], key == incumbent),
        )
    overall = max(
        selected_by_algorithm.values(),
        key=lambda key: (scores[key], key in INCUMBENTS.values(), key),
    )
    return overall, selected_by_algorithm, best_extended, scores


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
            raise ValueError("Test result belongs to another frozen selection.")
        for name, expected in result["artifact_sha256"].items():
            if digest(folder / name) != expected:
                raise ValueError("Saved v4 test artifact changed.")
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
        core.save_npz(folder / "test_predictions.npz", probabilities=probabilities,
                      decisions=decisions, truth=truth, labels=np.asarray(labels))
        core.write(folder / "test_per_label.json", per_label)
        result = {
            "role": "exploratory grouped-v4 comparison at fixed threshold 0.5",
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
            raise ValueError("Threshold test result belongs to another selection.")
        for name, expected in threshold_result["artifact_sha256"].items():
            if digest(folder / name) != expected:
                raise ValueError("Saved threshold test artifact changed.")
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


def paired_seed_comparison(xgb_summary, lgbm_summary):
    seeds = sorted(set(xgb_summary["per_seed"]) & set(lgbm_summary["per_seed"]), key=int)
    deltas = np.asarray([
        xgb_summary["per_seed"][seed]["metrics"]["auprc"]
        - lgbm_summary["per_seed"][seed]["metrics"]["auprc"]
        for seed in seeds
    ])
    mean = float(deltas.mean())
    std = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    half_width = 2.7764451051977987 * std / np.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    return {
        "difference": "XGBoost minus LightGBM",
        "metric": "macro Average Precision",
        "seeds": [int(seed) for seed in seeds],
        "per_seed_difference": deltas.tolist(),
        "mean_difference": mean,
        "sample_std": std,
        "t_95_ci": [mean - half_width, mean + half_width],
    }


def paired_label_bootstrap(xgb_per_label, lgbm_per_label, labels, iterations=10000, seed=42):
    deltas = np.asarray([
        xgb_per_label[name]["auprc"] - lgbm_per_label[name]["auprc"]
        for name in labels
    ], dtype=float)
    rng = np.random.default_rng(seed)
    samples = deltas[rng.integers(0, len(deltas), size=(iterations, len(deltas)))].mean(axis=1)
    return {
        "difference": "XGBoost minus LightGBM",
        "metric": "per-label Average Precision",
        "resampling_unit": "label",
        "labels": len(labels),
        "iterations": iterations,
        "seed": seed,
        "observed_mean_difference": float(deltas.mean()),
        "percentile_95_ci": np.percentile(samples, [2.5, 97.5]).tolist(),
        "bootstrap_probability_xgb_higher": float(np.mean(samples > 0)),
    }


def run_all(dataset, run):
    core.check_host()
    dataset, run = Path(dataset).resolve(), Path(run).resolve()
    meta, protocol = preflight(dataset)
    run.mkdir(parents=True, exist_ok=True)
    identity = {
        "protocol": digest(PROTOCOL),
        "dataset_manifest": digest(dataset / "dataset_manifest.json"),
        "seed_manifest": digest(SEED_MANIFEST),
        "v3_reference": digest(V3_REFERENCE),
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
        raise ValueError("V4 source, seed, configuration, or environment changed.")
    if not identity_path.exists():
        core.write(identity_path, identity)

    with core.run_lock(run):
        initialize_studies(run)
        split = core.read(dataset / "splits.json")
        with np.load(dataset / "train.npz", allow_pickle=False) as values:
            y = values["truth"]
            morgan = values["morgan"]
            descriptors = values["descriptors"]
            row_index = values["row_index"].astype(int)
            positions = {int(value): index for index, value in enumerate(row_index)}
        training_groups = pd.read_csv(dataset / "groups.csv").group_id.to_numpy()[row_index]
        reference = core.read(V3_REFERENCE)

        recipes = {}
        for key in CANDIDATES:
            algorithm, condition = algorithm_for(key), condition_for(key)
            recipes[key] = core.tuning(
                run, key, algorithm, features_for(condition, morgan, descriptors),
                y, meta, split, positions, protocol,
            )
            print(key, "tuning continuation complete", flush=True)

        for key in CANDIDATES:
            algorithm, condition = algorithm_for(key), condition_for(key)
            folder = run / key
            params = recipes[key]
            recipe_path = folder / "recipe.json"
            if recipe_path.exists() and core.read(recipe_path) != params:
                raise ValueError(f"{key}: recipe checkpoint changed.")
            core.write(recipe_path, params)
            core.write(folder / "training_policy.json", {
                "class_imbalance": core.WEIGHTING_POLICY,
                "condition": condition,
                "selection_data": "training-validation only",
            })
            values = core.cross_validate(
                algorithm, params, features_for(condition, morgan, descriptors), y,
                meta["labels"], split["folds"], positions, core.Limits(), meta["labels"],
                folder / "cv_per_label.json", folder / "cv_fold_metrics.json",
                folder / "cv_oof_predictions.npz",
            )
            core.write(folder / "cv_summary.json", summarize(values, meta["labels"]))
            print(key, "primary CV complete", flush=True)

        robustness = {}
        for key in CANDIDATES:
            algorithm, condition = algorithm_for(key), condition_for(key)
            robustness[key] = core.repeated_grouped_cv(
                run, key, algorithm, recipes[key],
                features_for(condition, morgan, descriptors), y, meta["labels"],
                training_groups, protocol,
            )
            print(key, "repeated grouped CV complete", flush=True)

        frozen = run / "selection_frozen.json"
        if not frozen.exists():
            overall, selected_by_algorithm, best_extended, scores = choose_candidates(
                robustness, reference
            )
            names = (
                "recipe.json", "training_policy.json", "cv_per_label.json",
                "cv_fold_metrics.json", "cv_oof_predictions.npz", "cv_summary.json",
            )
            files = [run / key / name for key in CANDIDATES for name in names]
            files.extend(path for path in (run / "robustness").rglob("*") if path.is_file())
            core.write(frozen, {
                "protocol": RUN_ID,
                "selected_by_algorithm": selected_by_algorithm,
                "best_extended_by_algorithm": best_extended,
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
                raise ValueError("Frozen grouped-v4 selection changed.")

        selected_by_algorithm = freeze["selected_by_algorithm"]
        needs_new_test = any(key in CANDIDATES for key in selected_by_algorithm.values())
        if needs_new_test:
            with np.load(dataset / "test.npz", allow_pickle=False) as values:
                test_truth = values["truth"]
                test_morgan = values["morgan"]
                test_descriptors = values["descriptors"]

        test_summaries = {}
        threshold_summaries = {}
        test_per_label = {}
        selected_robustness = {}
        for algorithm in ALGORITHMS:
            key = selected_by_algorithm[algorithm]
            if key in CANDIDATES:
                condition = condition_for(key)
                folder = run / key
                x_train = features_for(condition, morgan, descriptors)
                fit_final_models(folder, algorithm, recipes[key], x_train, y, meta["labels"])
                print(key, "final fit complete", flush=True)
                result, threshold_result = evaluate_test(
                    run, frozen, key, algorithm,
                    features_for(condition, test_morgan, test_descriptors),
                    test_truth, meta["labels"],
                )
                test_summaries[key] = result["summary"]
                threshold_summaries[key] = threshold_result["summary"]
                test_per_label[algorithm] = core.read(folder / "test_per_label.json")
                selected_robustness[algorithm] = robustness[key]
            else:
                folder = SEED_ROOT / key
                test_summaries[key] = core.read(folder / "test_summary.json")["summary"]
                threshold_summaries[key] = core.read(
                    folder / "test_summary_validation_thresholds.json"
                )["summary"]
                test_per_label[algorithm] = core.read(folder / "test_per_label.json")
                selected_robustness[algorithm] = core.read(
                    folder / "robustness_summary.json"
                )

        tuning_outcome = {}
        seed_registry = core.read(SEED_MANIFEST)["studies"]
        for key in CANDIDATES:
            state = core.read(run / key / "tuning_budget.json")
            trials = pd.read_csv(run / key / "trials.csv")
            completed = int((trials.state == "COMPLETE").sum())
            initial = seed_registry[key]["completed_trials"]
            tuning_outcome[key] = {
                "attempted_trials": int(len(trials)),
                "completed_trials": completed,
                "initial_completed_trials": initial,
                "new_completed_trials": completed - initial,
                "charged_seconds": state["charged_seconds"],
            }

        comparisons = {}
        for algorithm in ALGORITHMS:
            incumbent = INCUMBENTS[algorithm]
            extended = freeze["best_extended_by_algorithm"][algorithm]
            selected = selected_by_algorithm[algorithm]
            comparisons[algorithm] = {
                "incumbent": incumbent,
                "best_extended": extended,
                "selected": selected,
                "incumbent_validation_auprc": freeze["validation_scores"][incumbent],
                "best_extended_validation_auprc": freeze["validation_scores"][extended],
                "validation_difference_extended_minus_incumbent": (
                    freeze["validation_scores"][extended]
                    - freeze["validation_scores"][incumbent]
                ),
                "selected_test_auprc": test_summaries[selected]["metrics"]["auprc"],
                "incumbent_test_auprc": reference["incumbents"][incumbent]["test_auprc"],
                "test_difference_selected_minus_incumbent": (
                    test_summaries[selected]["metrics"]["auprc"]
                    - reference["incumbents"][incumbent]["test_auprc"]
                ),
            }

        core.write(run / "paired_seed_comparison.json", paired_seed_comparison(
            selected_robustness["xgb"], selected_robustness["lgbm"]
        ))
        core.write(run / "paired_label_bootstrap_test.json", paired_label_bootstrap(
            test_per_label["xgb"], test_per_label["lgbm"], meta["labels"]
        ))
        complete = {
            "protocol": RUN_ID,
            "selected_by_algorithm": selected_by_algorithm,
            "overall_selected_from_repeated_grouped_cv": freeze[
                "overall_selected_from_repeated_grouped_cv"
            ],
            "tuning_outcome": tuning_outcome,
            "per_algorithm_comparison": comparisons,
            "primary_test_threshold_0_5": test_summaries,
            "secondary_test_validation_thresholds": threshold_summaries,
            "paired_seed_comparison": core.read(run / "paired_seed_comparison.json"),
            "paired_label_bootstrap_test": core.read(run / "paired_label_bootstrap_test.json"),
            "test_used_for_selection": False,
            "confirmatory_status": protocol["confirmatory_status"],
            "compute_profile": core.compute_profile(),
        }
        core.write(run / "complete.json", complete)
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
            "status": "V4_INPUTS_VERIFIED",
            "algorithms": list(ALGORITHMS),
            "labels": len(meta["labels"]),
            "additional_seconds_per_study": protocol["tuning"][
                "additional_seconds_per_study"
            ],
            "additional_seconds_total": protocol["tuning"]["additional_seconds_total"],
            "training_started": False,
        })
    else:
        print(run_all(args.dataset, args.run))
