"""Read-only checks of frozen ML method artifacts. No fitting or mutation of project."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(r"C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from research.results import main_run
from research.splits import inner_folds
from research.validation import select_validation_thresholds
from research.experiment import select_candidates

def read(p):
    return json.loads(p.read_text(encoding="utf-8"))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

run = main_run()
data = ROOT / "data/builds/perfume-five-grouped-v2"
config = read(ROOT / "KODE_ML/research/config/experiment.json")
meta = read(data / "dataset_manifest.json")
split = read(data / "splits.json")
labels = meta["labels"]
with np.load(data / "train.npz", allow_pickle=False) as z:
    truth, indices = z["truth"], z["row_index"]
with np.load(data / "test.npz", allow_pickle=False) as z:
    test_truth = z["truth"]
groups = pd.read_csv(data / "groups.csv").group_id.to_numpy()[indices]
positions = {int(row): pos for pos, row in enumerate(indices)}
universe = set(range(len(truth)))
candidates = [f"{alg}_{cond}" for alg in ("xgb", "lgbm") for cond in "ABCD"]
result = {"run": str(run), "no_fit_or_project_write": True, "folds": {}, "candidates": {}, "tuning": {}}
fold_counts = set()
fold_total = 0
for seed in config["robustness"]["seeds"]:
    paths = [run / "robustness" / key / f"seed_{seed}" / "folds.json" for key in candidates]
    assert len({sha(path) for path in paths}) == 1
    saved = read(paths[0])["folds"]
    assert list(saved) == labels
    for j, name in enumerate(labels):
        folds = saved[name]
        generated = inner_folds(np.arange(len(truth)), truth[:, j], groups, 5, seed)
        assert folds == generated
        seen = []
        fold_counts.add(len(folds))
        for f in folds:
            a, b = f["train"], f["validation"]
            assert set(a) | set(b) == universe and not set(a) & set(b)
            assert len(a) == len(set(a)) and len(b) == len(set(b))
            assert not set(groups[a]) & set(groups[b])
            assert len(np.unique(truth[a, j])) == len(np.unique(truth[b, j])) == 2
            seen.extend(b)
            fold_total += 1
        assert sorted(seen) == list(range(len(truth)))
        if seed == 42:
            tuning_local = [{role: [positions[i] for i in f[role]] for role in ("train", "validation")}
                            for f in split["folds"][name]]
            assert tuning_local == folds
    result["folds"][str(seed)] = {"identical_all_eight_candidates": True, "regenerated_exact": True}
result["folds_checked"] = fold_total
result["fold_count_values"] = sorted(fold_counts)
robustness = {}
for key in candidates:
    rob_path = run / "robustness" / key
    summary = read(rob_path / "robustness_summary.json")
    robustness[key] = summary
    arrays = []
    for seed in config["robustness"]["seeds"]:
        folder = rob_path / f"seed_{seed}"
        with np.load(folder / "cv_oof_predictions.npz", allow_pickle=False) as z:
            assert np.array_equal(z["truth"], truth)
            assert z["labels"].tolist() == labels
            p = z["probabilities"]
            assert p.shape == truth.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            assert np.array_equal(z["decisions"], p >= .5)
            arrays.append(p)
    pooled = np.mean(np.stack(arrays), axis=0)
    with np.load(rob_path / "pooled_oof_predictions.npz", allow_pickle=False) as z:
        assert np.array_equal(pooled, z["probabilities"])
        assert np.array_equal(truth, z["truth"])
    threshold = select_validation_thresholds(truth, pooled, labels, config["threshold_selection"]["grid"])
    assert threshold == read(rob_path / "thresholds.json")
    result["candidates"][key] = {"oof_cohort_and_average_correct": True, "all_109_thresholds_recomputed_exactly": True,
                                  "cv_macro_ap": summary["mean_metrics"]["auprc"]}
    if key[-1] in "AB":
        assert read(run / key / "recipe.json") == {"n_estimators": 100}
        continue
    trials = pd.read_csv(run / key / "trials.csv")
    complete = trials[trials.state == "COMPLETE"]
    failed = trials[trials.state == "FAIL"]
    best = complete.sort_values(["value", "number"], ascending=[False, True]).iloc[0]
    tune = read(run / key / "tuning_summary.json")
    recipe = read(run / key / "recipe.json")
    assert recipe == ast.literal_eval(best.user_attrs_resolved_parameters)
    assert int(best.number) == tune["best_trial"] and np.isclose(best.value, tune["best_macro_ap"], rtol=0, atol=1e-15)
    assert len(complete) == tune["complete_trials"] and len(failed) == tune["failed_trials"]
    assert set(failed.user_attrs_stop_reason) == {"budget_exhausted"}
    assert tune["budget"]["charged_seconds"] == tune["budget"]["budget_seconds"] == 28800
    assert tune["budget"]["status"] == "complete"
    for _, trial in trials.iterrows():
        resolved = ast.literal_eval(trial.user_attrs_resolved_parameters)
        for name, value in resolved.items():
            if name == "subsample_freq":
                assert value == 1
            else:
                assert np.isclose(trial[f"params_{name}"], value, rtol=0, atol=1e-14)
    seed42 = summary["per_seed"]["42"]["metrics"]["auprc"]
    assert np.isclose(seed42, best.value, rtol=0, atol=1e-12)
    duration = (pd.to_datetime(trials.datetime_complete).max() - pd.to_datetime(trials.datetime_start).min()).total_seconds()
    result["tuning"][key] = {"complete_trials": len(complete), "failed_at_budget": len(failed), "best_trial": int(best.number),
                              "best_objective": float(best.value), "seed42_cv_objective": seed42,
                              "recipes_match_winner": True, "actual_params_match_optuna": True,
                              "walltime_trial_span_seconds": duration}
selection = read(run / "selection_frozen.json")
overall, by_algorithm, scores = select_candidates(robustness)
assert overall == selection["overall_selected_from_repeated_grouped_cv"]
assert by_algorithm == selection["selected_by_algorithm"]
assert scores == selection["validation_scores"]
result["selection_recomputed"] = {"overall": overall, "by_algorithm": by_algorithm}
result["test_positive_counts"] = {"minimum": int(test_truth.sum(axis=0).min()), "maximum": int(test_truth.sum(axis=0).max()),
                                  "one_class_labels": int(sum(len(np.unique(test_truth[:, j])) < 2 for j in range(len(labels))))}
events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
first_test = min(e["time"] for e in events if e["stage"] == "final_fit_and_test")
assert selection["timestamp"] < first_test
result["freeze_time"] = selection["timestamp"]
result["first_final_fit_and_test_event"] = first_test
out = Path(__file__).with_name("method_evidence.json")
out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
