"""Baca grup historis yang dibekukan; bentuk fold internal dengan aturan yang sama."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
POLICY = "structure-or-morgan-components-v1"
def inner_folds(train, target, groups, max_folds, seed):
    train = np.asarray(train, dtype=int)
    truth, train_groups = target[train], groups[train]
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("Grouped CV requires both target classes.")
    upper = min(max_folds, len(set(train_groups[truth == 1])), len(set(train_groups[truth == 0])))
    # Lower k only for class feasibility, never based on model scores. No random-split fallback.
    for k in range(upper, 1, -1):
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        candidates = list(cv.split(np.zeros(len(train)), truth, train_groups))
        if all(len(np.unique(truth[a])) == 2 and len(np.unique(truth[b])) == 2 for a,b in candidates):
            return [{"train":train[a].tolist(), "validation":train[b].tolist()} for a,b in candidates]
    raise ValueError("No feasible grouped CV with both classes; keep molecules and review the protocol.")


def validate_grouped_split(split, groups):
    groups = np.asarray(groups)
    if split.get("group_policy") != POLICY or split.get("groups") != groups.tolist():
        raise ValueError("Missing or inconsistent structure/feature groups.")
    universe = set(range(len(groups)))
    def partition(a, b, expected):
        if len(a) != len(set(a)) or len(b) != len(set(b)):
            raise ValueError("Duplicate split indices.")
        a, b = set(a), set(b)
        if not a or not b or a & b or a | b != expected:
            raise ValueError("Invalid grouped partition coverage.")
        if set(groups[list(a)]) & set(groups[list(b)]):
            raise ValueError("Structure/feature group leakage across partitions.")
    partition(split["train"], split["test"], universe)
    training = set(split["train"])
    if set(split["folds"]) != set(split["eligible_labels"]):
        raise ValueError("Fold label registry mismatch.")
    fold_count = 0
    for label, folds in split["folds"].items():
        if len(folds) < 2:
            raise ValueError("At least two grouped folds required.")
        seen = []
        for fold in folds:
            partition(fold["train"], fold["validation"], training)
            seen.extend(fold["validation"])
            fold_count += 1
        if sorted(seen) != sorted(training):
            raise ValueError("Grouped validation must cover training rows once.")
    counts = pd.Series(groups).value_counts()
    return {"group_policy":POLICY, "group_count":int(len(counts)),
            "largest_group":int(counts.max()), "group_overlap_train_test":0,
            "group_overlap_cv":0, "folds_verified":fold_count}
