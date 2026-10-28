"""CV per label, prediksi OOF, dan ambang tambahan dari train saja."""
import numpy as np
from .runtime import read, write, save_npz, Limits, WEIGHTING_POLICY
from .training import fit
from .splits import POLICY, inner_folds
from .metrics import METRICS, binary_metrics, summarize

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
    """Evaluate one candidate across the fixed grouped-CV repetitions."""
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
