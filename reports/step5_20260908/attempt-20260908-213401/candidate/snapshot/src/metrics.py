"""Explicit multilabel metric definitions and threshold selection."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, hamming_loss
def average_precision(Y, P):
    return float(np.mean([average_precision_score(Y[:,j], P[:,j]) if Y[:,j].sum() else 0.0 for j in range(Y.shape[1])]))
def thresholds_for(Y, P):
    thresholds = []
    for j in range(Y.shape[1]):
        if np.unique(Y[:,j]).size < 2:
            raise ValueError(f"Threshold partition lacks both classes for label {j}")
        # All distinct decisions, including none positive; tie -> higher threshold.
        candidates = np.unique(np.r_[P[:,j], np.nextafter(P[:,j].max(), np.inf)])
        scores = [f1_score(Y[:,j], P[:,j] >= t, zero_division=0) for t in candidates]
        thresholds.append(float(candidates[np.flatnonzero(np.isclose(scores, max(scores), rtol=0, atol=1e-12))[-1]]))
    return np.asarray(thresholds)
def evaluate(Y, P, thresholds, labels):
    if P.shape != Y.shape or not np.isfinite(P).all() or np.any((P < 0) | (P > 1)):
        raise ValueError("Invalid probability matrix")
    if len(labels) != Y.shape[1] or len(set(labels)) != len(labels) or np.shape(thresholds) != (Y.shape[1],) or not np.isfinite(thresholds).all():
        raise ValueError("Invalid label/threshold schema")
    predicted = P >= thresholds
    per_label = {}
    aucs = []
    for j, label in enumerate(labels):
        auc = float(roc_auc_score(Y[:,j],P[:,j])) if np.unique(Y[:,j]).size == 2 else None
        if auc is not None:
            aucs.append(auc)
        per_label[label] = {"positive_support": int(Y[:,j].sum()),
                            "average_precision": float(average_precision_score(Y[:,j],P[:,j])) if Y[:,j].sum() else 0.0,
                            "roc_auc": auc, "f1": float(f1_score(Y[:,j],predicted[:,j],zero_division=0)),
                            "precision":float(precision_score(Y[:,j],predicted[:,j],zero_division=0)),
                            "recall":float(recall_score(Y[:,j],predicted[:,j],zero_division=0))}
    return {"average_precision_macro": average_precision(Y,P),
            "average_precision_micro": float(average_precision_score(Y,P,average="micro")),
            "roc_auc_macro": float(np.mean(aucs)) if aucs else None, "roc_auc_valid_labels": len(aucs),
            "subset_accuracy": float(np.all(predicted == Y, axis=1).mean()),
            "hamming_loss": float(hamming_loss(Y,predicted)),
            "f1_macro": float(f1_score(Y,predicted,average="macro",zero_division=0)),
            "f1_micro": float(f1_score(Y,predicted,average="micro",zero_division=0)),
            "precision_macro": float(precision_score(Y,predicted,average="macro",zero_division=0)),
            "recall_macro": float(recall_score(Y,predicted,average="macro",zero_division=0)),
            "per_label": per_label}
