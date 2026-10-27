"""Keep structural and representation equivalents together without deleting molecules."""
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

POLICY = "structure-or-morgan-components-v1"


def structure_feature_groups(smiles, morgan):
    if morgan.ndim != 2 or len(smiles) != len(morgan):
        raise ValueError("Grouping input shape mismatch.")
    parent = list(range(len(smiles)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    structures, fingerprints = {}, {}
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError("Grouping requires valid single molecules.")
        Chem.RemoveStereochemistry(mol)
        # Preserve isotope/charge; only remove stereo information for the grouping key.
        key = Chem.MolToSmiles(mol, isomericSmiles=True)
        fp = np.ascontiguousarray(morgan[i]).tobytes()
        if key in structures:
            union(i, structures[key])
        if fp in fingerprints:
            union(i, fingerprints[fp])
        structures.setdefault(key, i)
        fingerprints.setdefault(fp, i)
    roots = [find(i) for i in range(len(smiles))]
    registry = {root: j for j, root in enumerate(sorted(set(roots)))}
    return np.asarray([registry[r] for r in roots], dtype=np.int64)


def outer_split(groups, test_size, seed):
    # Test allocation uses structure groups only, never label scores or test performance.
    cv = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    a, b = next(cv.split(np.zeros(len(groups)), groups=groups))
    return np.sort(a), np.sort(b)


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
