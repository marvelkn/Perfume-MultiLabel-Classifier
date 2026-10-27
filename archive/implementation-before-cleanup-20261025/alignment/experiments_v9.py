"""V9: controlled Morgan 1024/2048 + eight descriptors, CPU, 16 h per study."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
import psutil

from . import experiments as core
from .data import ROOT, digest
from .descriptors_v7 import DESCRIPTORS8, EXTRA, verify as verify_features
from .experiments_v5 import fit_final_models
from .metrics import binary_metrics, summarize
from .perfume_data import OUTPUT
from .tuning_v6 import event, suggest
from .tuning_v9 import tune

warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names", category=UserWarning)
RUN_ID = "perfume-five-grouped-v9-bits-d8"
PROTOCOL = ROOT / "alignment/protocol_v9.json"
ALGORITHMS = ("xgb", "lgbm")
CANDIDATES = tuple(f"{a}_m{b}_{kind}" for a in ALGORITHMS for b in (1024, 2048) for kind in ("base", "tuned"))
TUNED = tuple(k for k in CANDIDATES if k.endswith("_tuned"))
BASELINES = tuple(k for k in CANDIDATES if k.endswith("_base"))


def bits(key):
    if key not in CANDIDATES:
        raise ValueError(f"Unknown V9 candidate: {key}")
    return int(key.split("_")[1][1:])


def features(key, morgan, descriptors):
    if (morgan.ndim != 2 or morgan.shape[1] != 2048 or descriptors.shape != (len(morgan), 8)
            or not np.isfinite(descriptors).all() or not np.isin(morgan, [0, 1]).all()):
        raise ValueError("Expected frozen binary Morgan 2048 and eight ordered descriptors.")
    fp = morgan if bits(key) == 2048 else np.maximum(morgan[:, :1024], morgan[:, 1024:])
    return np.column_stack((fp, descriptors)).astype(np.float32)


def verify_files(root, files):
    for name, expected in files.items():
        if digest(Path(root) / name) != expected:
            raise ValueError(f"Artifact changed: {name}")


def preflight(dataset=OUTPUT, extra=EXTRA):
    meta, _ = core.preflight(dataset)
    p = core.read(PROTOCOL)
    if p["id"] != RUN_ID or p["candidates"] != list(CANDIDATES):
        raise ValueError("Wrong V9 protocol/candidates.")
    if p["environment"] != core.environment() or p["dataset_manifest_sha256"] != digest(Path(dataset) / "dataset_manifest.json"):
        raise ValueError("V9 environment/dataset mismatch.")
    if p["group_policy"] != core.POLICY or p["class_imbalance"]["policy"] != core.WEIGHTING_POLICY:
        raise ValueError("Frozen grouping/weighting changed.")
    if (p["tuning"]["seconds_per_study"] != 57600 or p["tuning"]["seconds_total"] != 230400
            or p["tuning"]["snapshot_seconds"] != [28800, 57600]):
        raise ValueError("V9 requires four equal 16-hour budgets and 8/16-hour snapshots.")
    if p["robustness"]["seeds"] != [7, 21, 42, 84, 100] or p["robustness"]["folds"] != 5:
        raise ValueError("Frozen V7 validation seeds/folds changed.")
    if p["compute_profile_sha256"] != digest(core.COMPUTE_PROFILE):
        raise ValueError("Compute profile changed.")
    profile = core.compute_profile()
    if profile["threads_per_model"] != 6 or profile["execution_device"] != "cpu":
        raise ValueError("V9 uses the same six-thread CPU profile for every candidate.")
    feature_meta, _, _ = verify_features(dataset, extra)
    if p["feature_manifest_sha256"] != digest(Path(extra) / "manifest.json"):
        raise ValueError("Feature manifest changed.")
    if p["features"]["descriptors"] != list(DESCRIPTORS8) or p["features"]["morgan_bits"] != [1024, 2048]:
        raise ValueError("Unexpected V9 feature definitions.")
    for algorithm in ALGORITHMS:
        recipe = p["initial_recipes"][algorithm]
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            if suggest(optuna.trial.FixedTrial(recipe), algorithm, p) != recipe:
                raise ValueError("Initial recipe outside search space.")
    return dict(meta, feature_spec=feature_meta["feature_spec"]), p


def select_candidates(robustness):
    scores = {k: robustness[k]["mean_metrics"]["auprc"] for k in CANDIDATES}
    # Fixed tie-breaking: declared order, never test performance.
    chosen = {a: max((k for k in CANDIDATES if k.startswith(a + "_")),
                    key=lambda k: (scores[k], -CANDIDATES.index(k))) for a in ALGORITHMS}
    overall = max(chosen.values(), key=lambda k: (scores[k], -CANDIDATES.index(k)))
    return overall, chosen, scores


def evaluate_test(run, frozen, key, x, truth, labels):
    folder = run / key
    models = core.read(folder / "models.json")
    verify_files(folder, {v["file"]: v["sha256"] for v in models.values()})
    primary = folder / "test_summary.json"
    if primary.exists():
        result = core.read(primary)
        if result["selection_sha256"] != digest(frozen):
            raise ValueError("Selection changed.")
        verify_files(folder, result["artifact_sha256"])
        with np.load(folder / "test_predictions.npz", allow_pickle=False) as z:
            prob = z["probabilities"]
            if not np.array_equal(z["truth"], truth) or z["labels"].tolist() != labels:
                raise ValueError("Test cohort changed.")
    else:
        prob = np.column_stack([joblib.load(folder / models[n]["file"]).predict_proba(x)[:, 1] for n in labels]).astype(np.float32)
        decisions = (prob >= .5).astype(np.int8)
        values = {n: binary_metrics(truth[:, j], prob[:, j], decisions[:, j]) for j, n in enumerate(labels)}
        core.save_npz(folder / "test_predictions.npz", probabilities=prob, decisions=decisions, truth=truth, labels=np.asarray(labels))
        core.write(folder / "test_per_label.json", values)
        result = {"role": "exploratory V9; primary fixed threshold 0.5", "summary": summarize(values, labels),
                  "diagnostics": core.prediction_diagnostics(truth, decisions, labels),
                  "selection_sha256": digest(frozen),
                  "artifact_sha256": {n: digest(folder / n) for n in ("test_predictions.npz", "test_per_label.json")}}
        core.write(primary, result)
    thresholds_path = run / "robustness" / key / "thresholds.json"
    secondary = folder / "test_summary_validation_thresholds.json"
    if secondary.exists():
        extra = core.read(secondary)
        if extra["selection_sha256"] != digest(frozen) or extra["threshold_source_sha256"] != digest(thresholds_path):
            raise ValueError("Threshold source changed.")
        verify_files(folder, extra["artifact_sha256"])
    else:
        info = core.read(thresholds_path)
        thresholds = np.asarray([info["labels"][n]["threshold"] for n in labels])
        decisions = (prob >= thresholds[None, :]).astype(np.int8)
        values = {n: binary_metrics(truth[:, j], prob[:, j], decisions[:, j]) for j, n in enumerate(labels)}
        core.save_npz(folder / "test_predictions_validation_thresholds.npz", probabilities=prob, decisions=decisions,
                      truth=truth, labels=np.asarray(labels), thresholds=thresholds)
        core.write(folder / "test_per_label_validation_thresholds.json", values)
        extra = {"role": "secondary; thresholds from development OOF only", "summary": summarize(values, labels),
                 "diagnostics": core.prediction_diagnostics(truth, decisions, labels),
                 "selection_sha256": digest(frozen), "threshold_source_sha256": digest(thresholds_path),
                 "artifact_sha256": {n: digest(folder / n) for n in ("test_predictions_validation_thresholds.npz", "test_per_label_validation_thresholds.json")}}
        core.write(secondary, extra)
    return result, extra


def budget_validation(run, protocol, recipes, robustness, morgan, desc, y, labels, groups):
    rows = []
    for key in TUNED:
        saved = core.read(run / key / "budget_snapshots.json")
        short = saved["28800"]
        row = {"candidate": key, "trial_8h": short["best_trial"], "trial_16h": saved["57600"]["best_trial"],
               "search_ap_8h": short["best_cv_ap"], "search_ap_16h": saved["57600"]["best_cv_ap"],
               "completed_trials_8h": short["completed_trials"], "completed_trials_16h": saved["57600"]["completed_trials"],
               "cv_ap_16h": robustness[key]["mean_metrics"]["auprc"], "status": short["status"]}
        if short["status"] == "available":
            if short["parameters"] == recipes[key]:
                short_cv = robustness[key]
                row["validation_reused_identical_recipe"] = True
            else:
                event(run, "validate_8h_snapshot", candidate=key)
                short_cv = core.repeated_grouped_cv(run, key + "_8h", key.split("_")[0], short["parameters"],
                    features(key, morgan, desc), y, labels, groups, protocol)
                for seed in protocol["robustness"]["seeds"]:
                    if digest(run / "robustness" / (key + "_8h") / f"seed_{seed}" / "folds.json") != digest(run / "robustness" / key / f"seed_{seed}" / "folds.json"):
                        raise ValueError("8 h / 16 h validation folds differ.")
                row["validation_reused_identical_recipe"] = False
            row["cv_ap_8h"] = short_cv["mean_metrics"]["auprc"]
            row["delta_cv_ap_16h_minus_8h"] = row["cv_ap_16h"] - row["cv_ap_8h"]
            row["delta_per_seed"] = [robustness[key]["per_seed"][str(s)]["metrics"]["auprc"] - short_cv["per_seed"][str(s)]["metrics"]["auprc"] for s in protocol["robustness"]["seeds"]]
        rows.append(row)
    core.write(run / "budget_comparison.json", {"rows": rows, "test_used": False,
        "caveat": "Within-study snapshots; additional repeated CV is development evidence, not independent/nested evaluation."})
    pd.DataFrame(rows).to_csv(run / "budget_comparison.csv", index=False)


def contrasts(run, protocol, robustness):
    for seed in protocol["robustness"]["seeds"]:
        if len({digest(run / "robustness" / k / f"seed_{seed}" / "folds.json") for k in CANDIDATES}) != 1:
            raise ValueError("Candidate validation folds differ.")
    rows = []
    for a in ALGORITHMS:
        pairs = [(f"{a}_m1024_base", f"{a}_m2048_base", "bits_at_fixed_baseline"),
                 (f"{a}_m1024_tuned", f"{a}_m2048_tuned", "bits_after_equal_budget_tuning")]
        pairs += [(f"{a}_m{b}_base", f"{a}_m{b}_tuned", f"tuning_m{b}") for b in (1024, 2048)]
        for left, right, meaning in pairs:
            delta = [robustness[right]["per_seed"][str(s)]["metrics"]["auprc"] - robustness[left]["per_seed"][str(s)]["metrics"]["auprc"] for s in protocol["robustness"]["seeds"]]
            rows.append({"left": left, "right": right, "meaning": meaning, "delta_right_minus_left": delta,
                         "mean_delta": float(np.mean(delta)), "sd_delta": float(np.std(delta, ddof=1)),
                         "positive_seeds": sum(d > 0 for d in delta), "negative_seeds": sum(d < 0 for d in delta)})
    result = {"identical_paired_folds": True, "contrasts": rows,
              "caveat": "Descriptive paired differences; seeds are correlated; no p-value or automatic superiority claim."}
    core.write(run / "paired_feature_cv.json", result)
    pd.DataFrame(rows).to_csv(run / "feature_deltas.csv", index=False)
    return result


def summaries(run, meta, protocol, freeze):
    rows = []
    for key in CANDIDATES:
        rob = core.read(run / "robustness" / key / "robustness_summary.json")
        test = core.read(run / key / "test_summary.json")
        extra = core.read(run / key / "test_summary_validation_thresholds.json")
        rows.append({"candidate": key, "morgan_bits": bits(key), "descriptors": 8, "input_features": bits(key) + 8,
                     "tuned": key in TUNED, "cv_ap_mean": rob["mean_metrics"]["auprc"], "cv_ap_sd": rob["std_metrics"]["auprc"],
                     **test["summary"]["metrics"], "macro_f1_0_5": test["diagnostics"]["macro_f1"],
                     "macro_f1_oof_threshold": extra["diagnostics"]["macro_f1"],
                     "precision_oof_threshold": extra["summary"]["metrics"]["precision"],
                     "recall_oof_threshold": extra["summary"]["metrics"]["recall"],
                     "predicted_labels_per_molecule": test["diagnostics"]["mean_predicted_labels_per_molecule"],
                     "selected_overall": key == freeze["overall_selected_from_repeated_grouped_cv"]})
    pd.DataFrame(rows).to_csv(run / "comparison_all.csv", index=False)
    overall = freeze["overall_selected_from_repeated_grouped_cv"]
    spec = dict(meta["feature_spec"], n_bits=bits(overall), descriptors=list(DESCRIPTORS8),
                use_descriptors=True, dimensions={"input": bits(overall) + 8}, dtype="float32", fragment_policy="single")
    core.write(run / "inference_bundle.json", {
        "protocol": RUN_ID, "selected_candidate": overall, "algorithm": overall.split("_")[0],
        "models_directory": overall, "models": core.read(run / overall / "models.json"), "feature_spec": spec,
        "feature_names": [f"morgan_{i:04d}" for i in range(bits(overall))] + list(DESCRIPTORS8),
        "labels": meta["labels"], "primary_threshold": 0.5,
        "secondary_thresholds_path": f"robustness/{overall}/thresholds.json", "environment": core.environment(),
        "selection_sha256": digest(run / "selection_frozen.json"),
        "score_interpretation": "Uncalibrated score; not intensity, sensory certainty or mixture composition.",
        "release_policy": "Exploratory artifact; V7 remains the locked thesis result. No automatic deployment.",
        "prediction_command": f"python -m alignment.predict_v9 --run runs/{RUN_ID} --smiles SMILES"})
    lines = ["# Hasil eksplorasi V9: Morgan 1024/2048 + delapan deskriptor", "",
             f"Kandidat pilihan berdasarkan AP validasi: **{overall}**.", "", protocol["confirmatory_status"], "",
             "| Kandidat | Fitur | CV AP | SD | Test AP | AUROC | F1 0,5 | F1 ambang OOF |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['candidate']} | {r['input_features']} | {r['cv_ap_mean']:.6f} | {r['cv_ap_sd']:.6f} | {r['auprc']:.6f} | {r['auroc']:.6f} | {r['macro_f1_0_5']:.6f} | {r['macro_f1_oof_threshold']:.6f} |")
    lines += ["", "## Apakah 16 jam membantu dibanding 8 jam?", "",
              "| Studi | Trial 8 h / 16 h | AP repeated CV 8 h | AP repeated CV 16 h | Selisih |",
              "|---|---|---:|---:|---:|"]
    for r in core.read(run / "budget_comparison.json")["rows"]:
        if r["status"] == "available":
            lines.append(f"| {r['candidate']} | {r['trial_8h']} / {r['trial_16h']} | {r['cv_ap_8h']:.6f} | {r['cv_ap_16h']:.6f} | {r['delta_cv_ap_16h_minus_8h']:+.6f} |")
        else:
            lines.append(f"| {r['candidate']} | Tidak ada trial selesai pada 8 h | - | {r['cv_ap_16h']:.6f} | - |")
    lines += ["", "Snapshot 8 h hanya analisis validasi; tidak ikut memilih kandidat atau dievaluasi pada test.",
              "Semua kandidat memakai delapan deskriptor, bobot kelas, kelompok dan fold yang sama.",
              "Perbandingan baseline mengisolasi panjang bit; konfigurasi setelah Optuna dapat berbeda.",
              "Panjang bit atau waktu tuning lebih besar tidak otomatis lebih baik; laporkan semua penurunan.",
              "AP adalah Average Precision; bukan accuracy atau persentase kekuatan aroma.",
              "Test lama pernah dilihat; repeated CV bukan nested CV. Tidak mengklaim evaluasi independen atau signifikansi dari lima seed.",
              "V7 tetap hasil skripsi yang dikunci. V9 adalah eksplorasi tambahan, tanpa penggantian model aplikasi otomatis."]
    (run / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"protocol": RUN_ID, "completed_at": datetime.now(timezone.utc).isoformat(),
            "overall_selected_from_repeated_grouped_cv": overall, "selected_by_algorithm": freeze["selected_by_algorithm"],
            "comparison": rows, "test_used_for_selection": False,
            "tuning_outcome": {k: core.read(run / k / "tuning_summary.json") for k in TUNED},
            "confirmatory_status": protocol["confirmatory_status"]}


def run_all(dataset=OUTPUT, run=None, extra=EXTRA):
    core.check_host()
    dataset, run = Path(dataset).resolve(), Path(run or ROOT / "runs" / RUN_ID).resolve()
    meta, protocol = preflight(dataset, extra)
    _, all_morgan, all_desc = verify_features(dataset, extra)
    run.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(run).free < 10 * 1024**3:
        raise RuntimeError("Sediakan minimal 10 GiB ruang disk; 20 GiB dianjurkan.")
    identity = {"protocol": digest(PROTOCOL), "dataset": digest(dataset / "dataset_manifest.json"),
                "features": digest(Path(extra) / "manifest.json"), "environment": core.environment(),
                "source": {p.relative_to(ROOT).as_posix(): digest(p) for folder in ("alignment", "src") for p in (ROOT / folder).rglob("*.py")},
                "compute_profile": digest(core.COMPUTE_PROFILE), "host": platform.node(),
                "physical_cores": psutil.cpu_count(logical=False), "execution_device": "cpu"}
    with core.run_lock(run):
        path = run / "identity.json"
        if path.exists() and core.read(path) != identity:
            raise ValueError("Kode, data, environment atau PC berubah; jangan campur run.")
        if not path.exists():
            core.write(path, identity)
        if (run / "complete.json").exists() and (run / "completion_manifest.json").exists():
            verify_files(run, core.read(run / "completion_manifest.json")["files"])
            event(run, "already_complete", message="Checksum cocok; training tidak diulang.")
            return run / "complete.json"
        frozen = run / "selection_frozen.json"
        with np.load(dataset / "train.npz", allow_pickle=False) as z:
            y, indices = z["truth"], z["row_index"].astype(int)
        morgan, desc = all_morgan[indices], all_desc[indices]
        if not frozen.exists():
            split = core.read(dataset / "splits.json")
            positions = {int(v): i for i, v in enumerate(indices)}
            groups = pd.read_csv(dataset / "groups.csv").group_id.to_numpy()[indices]
            recipes, robustness = {}, {}
            for key in (*BASELINES, *TUNED):
                folder = run / key
                folder.mkdir(exist_ok=True)
                recipe_path = folder / "recipe.json"
                if recipe_path.exists():
                    recipes[key] = core.read(recipe_path)
                    if key in TUNED and core.read(folder / "tuning_summary.json")["budget"]["status"] != "complete":
                        raise ValueError("Recipe exists before tuning completes.")
                else:
                    event(run, "prepare_candidate", candidate=key)
                    recipes[key] = dict(protocol["baseline_parameters"]) if key in BASELINES else tune(
                        run, key, key.split("_")[0], features(key, morgan, desc), y, meta, split, positions,
                        protocol, protocol["initial_recipes"][key.split("_")[0]])
                    core.write(recipe_path, recipes[key])
                    core.write(folder / "training_policy.json", {"morgan_bits": bits(key), "descriptors": 8,
                        "weighting": core.WEIGHTING_POLICY, "tuning_seconds": 57600 if key in TUNED else 0,
                        "execution_device": "cpu", "threads": 6})
                event(run, "repeated_grouped_cv", candidate=key, seeds=protocol["robustness"]["seeds"])
                robustness[key] = core.repeated_grouped_cv(run, key, key.split("_")[0], recipes[key],
                    features(key, morgan, desc), y, meta["labels"], groups, protocol)
            contrasts(run, protocol, robustness)
            budget_validation(run, protocol, recipes, robustness, morgan, desc, y, meta["labels"], groups)
            overall, selected, scores = select_candidates(robustness)
            paths = [p for k in CANDIDATES for p in (run / k).iterdir() if p.is_file() and not p.name.endswith(".tmp")]
            paths += [run / n for n in ("paired_feature_cv.json", "feature_deltas.csv", "budget_comparison.json", "budget_comparison.csv")]
            paths += [p for p in (run / "robustness").rglob("*") if p.is_file() and not p.name.endswith(".tmp")]
            core.write(frozen, {"protocol": RUN_ID, "timestamp": datetime.now(timezone.utc).isoformat(),
                "selected_by_algorithm": selected, "overall_selected_from_repeated_grouped_cv": overall,
                "validation_scores": scores, "test_used_for_selection": False,
                "files": {p.relative_to(run).as_posix(): digest(p) for p in paths}})
        freeze = core.read(frozen)
        verify_files(run, freeze["files"])
        # Test targets are loaded only after every recipe, CV result and threshold is frozen.
        with np.load(dataset / "test.npz", allow_pickle=False) as z:
            truth_test, test_indices = z["truth"], z["row_index"].astype(int)
        if test_indices.tolist() != core.read(dataset / "splits.json")["test"]:
            raise ValueError("Test row order changed.")
        for key in CANDIDATES:
            event(run, "final_fit_and_test", candidate=key)
            fit_final_models(run / key, key.split("_")[0], core.read(run / key / "recipe.json"),
                             features(key, morgan, desc), y, meta["labels"])
            evaluate_test(run, frozen, key, features(key, all_morgan[test_indices], all_desc[test_indices]), truth_test, meta["labels"])
        core.write(run / "complete.json", summaries(run, meta, protocol, freeze))
        excluded = {"completion_manifest.json", ".alignment.lock", "progress.json", "events.jsonl"}
        core.write(run / "completion_manifest.json", {"files": {
            p.relative_to(run).as_posix(): digest(p) for p in run.rglob("*")
            if p.is_file() and p.name not in excluded and not p.name.endswith(".tmp")}})
        event(run, "complete", selected=freeze["overall_selected_from_repeated_grouped_cv"])
    return run / "complete.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUTPUT)
    parser.add_argument("--features", type=Path, default=EXTRA)
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / RUN_ID)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        meta, _ = preflight(args.dataset, args.features)
        print({"status": "V9_INPUTS_VERIFIED", "candidates": list(CANDIDATES), "labels": len(meta["labels"]),
               "feature_dimensions": [1032, 2056], "tuning_hours_total": 64, "device": "cpu", "training_started": False})
    else:
        print(run_all(args.dataset, args.run, args.features))


if __name__ == "__main__":
    main()
