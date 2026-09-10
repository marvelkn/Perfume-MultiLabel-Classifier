"""Six Suh reporting metrics; actual classifier decisions must be supplied."""
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score

METRICS = ("accuracy", "auroc", "auprc", "specificity", "precision", "recall")


def binary_metrics(truth, probabilities, predicted):
    y, p, d = np.asarray(truth), np.asarray(probabilities), np.asarray(predicted)
    if y.ndim != 1 or not len(y) or p.shape != y.shape or d.shape != y.shape:
        raise ValueError("Expected equally sized non-empty vectors.")
    if not np.isin(y, [0, 1]).all() or not np.isin(d, [0, 1]).all():
        raise ValueError("Targets and decisions must be binary.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities.")
    both = np.unique(y).size == 2
    tn = int(np.sum((y == 0) & (d == 0)))
    fp = int(np.sum((y == 0) & (d == 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ap = float(average_precision_score(y, p))
    # sklearn 1.6 returns NaN AUROC for one class; AP still runs in the author's helper.
    return {"accuracy": float(accuracy_score(y, d)),
            "auroc": float(roc_auc_score(y, p)) if both else None,
            "auprc": ap,
            "specificity": tn / (tn + fp) if both else None,
            "precision": float(precision_score(y, d, zero_division=0)),
            "recall": float(recall_score(y, d, zero_division=0))}


def summarize(per_label, labels):
    if len(set(labels)) != len(labels) or any(label not in per_label for label in labels):
        raise ValueError("Summary label registry mismatch.")
    scores, coverage = {}, {}
    for metric in METRICS:
        valid = [label for label in labels if per_label[label].get(metric) is not None]
        values = [per_label[label][metric] for label in valid]
        if not all(np.isfinite(values)):
            raise ValueError("Nonfinite metric should be represented explicitly as None.")
        scores[metric] = float(np.mean(values)) if values else None
        coverage[metric] = valid
    return {"metrics": scores, "valid_labels": coverage, "requested_labels": list(labels)}


def compare(left, right):
    """Paired label coverage prevents a change of denominator from appearing as a gain."""
    if left["requested_labels"] != right["requested_labels"]:
        raise ValueError("Different summary cohorts.")
    if left["valid_labels"] != right["valid_labels"]:
        raise ValueError("Different valid-label coverage; require an explicitly shared cohort.")
    return {key: right["metrics"][key] - left["metrics"][key]
            if right["metrics"][key] is not None else None for key in METRICS}
