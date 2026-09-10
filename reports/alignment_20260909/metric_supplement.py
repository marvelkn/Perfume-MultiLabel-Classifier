"""Add explicit reporting metrics to saved predictions; never fit or retune."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import auc, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.metrics import evaluate

def alignment_metrics(Y, P, thresholds, labels):
    Y, P, thresholds = np.asarray(Y), np.asarray(P), np.asarray(thresholds)
    if Y.ndim != 2 or not all(Y.shape) or not np.isin(Y, [0, 1]).all():
        raise ValueError("Truth must be a non-empty binary matrix.")
    result = evaluate(Y, P, thresholds, labels)
    predicted = P >= thresholds
    accuracies, specificities, pr_areas = [], [], []
    for j, label in enumerate(labels):
        truth, pred = Y[:, j].astype(bool), predicted[:, j]
        tn = int(np.sum(~truth & ~pred))
        fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred))
        tp = int(np.sum(truth & pred))
        accuracy = (tp + tn) / len(truth)
        specificity = tn / (tn + fp) if tn + fp else None
        pr_area = None
        if truth.any():
            precision, recall, _ = precision_recall_curve(truth, P[:, j])
            pr_area = float(auc(recall, precision))
            pr_areas.append(pr_area)
        accuracies.append(accuracy)
        if specificity is not None:
            specificities.append(specificity)
        result["per_label"][label].update({
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": accuracy, "specificity": specificity,
            "auprc_trapezoid": pr_area,
        })
    result.update({
        "accuracy_macro_binary": float(np.mean(accuracies)),
        "specificity_macro": float(np.mean(specificities)) if specificities else None,
        "specificity_valid_labels": len(specificities),
        "auprc_trapezoid_macro": float(np.mean(pr_areas)) if pr_areas else None,
        "auprc_trapezoid_valid_labels": len(pr_areas),
        "thresholds": thresholds.tolist(),
        "metric_notes": {
            "accuracy": "Mean binary accuracy across labels, not subset accuracy.",
            "auprc_trapezoid": "Linear trapezoidal PR area; separate from sklearn average precision.",
            "undefined": "Specificity without negatives and PR area without positives are null; their macro means exclude nulls.",
            "paper_parity": "Not verified: Suh code, averaging, PR integration and decision threshold require confirmation.",
            "purpose": "Reporting only; the original macro-AP tuning objective is unchanged.",
        },
    })
    return result

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True,
                        help="Existing experiment/xgb or experiment/lgbm with completed test outputs")
    parser.add_argument("--output", help="New JSON output; defaults to model-dir/alignment_metrics.json")
    args = parser.parse_args()
    directory = Path(args.model_dir).resolve()
    manifest_path = directory / "model_manifest.json"
    predictions_path = directory / "test_predictions.npz"
    original_metrics = directory / "test_metrics.json"
    freeze_path = directory.parent / "selection_frozen.json"
    for path in (manifest_path, predictions_path, original_metrics, freeze_path):
        if not path.is_file():
            raise FileNotFoundError(f"Completed, frozen evaluation required: {path}")
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["model_manifests"].get(meta["algorithm"]) != digest(manifest_path):
        raise ValueError("Model manifest differs from selection freeze.")
    if [m["label"] for m in meta["models"]] != meta["labels"]:
        raise ValueError("Label/model order mismatch.")
    with np.load(predictions_path, allow_pickle=False) as saved:
        result = alignment_metrics(saved["truth"], saved["probabilities"],
            np.asarray([m["threshold"] for m in meta["models"]]), meta["labels"])
    old = json.loads(original_metrics.read_text(encoding="utf-8"))
    for key in ("average_precision_macro", "subset_accuracy", "hamming_loss"):
        if not np.isclose(result[key], old[key], rtol=0, atol=1e-12):
            raise ValueError(f"Saved predictions do not reproduce existing metric: {key}")
    result["reporting_provenance"] = {
        "algorithm": meta["algorithm"], "dataset_id": meta["dataset_id"],
        "threshold_source": meta.get("threshold_source"),
        "model_manifest_sha256": digest(manifest_path),
        "predictions_sha256": digest(predictions_path),
        "original_metrics_sha256": digest(original_metrics),
        "script_sha256": digest(Path(__file__)),
        "training_started": False, "thresholds_changed": False,
    }
    target = Path(args.output).resolve() if args.output else directory / "alignment_metrics.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(target)

if __name__ == "__main__":
    main()
