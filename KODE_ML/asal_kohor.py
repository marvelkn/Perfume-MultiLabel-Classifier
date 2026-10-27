"""Audit asal kohor beku. Bukan pembangun dataset baru atau fitur model 1024.

Tiga fungsi historis dipertahankan persis secara AST dari source pembentuk kohor.
Berikan fingerprint 1024 tersimpan untuk merekonstruksi identitas grup historis.
Fitur model utama tetap dibuat oleh research.features: Morgan 2048 + 8.
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit
from research.splits import POLICY, inner_folds, validate_grouped_split

def extract_features(records, config):
    raise ValueError("Audit asal kohor wajib menerima morgan historis tersimpan; tidak membuat fitur lama.")

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


def encode_and_split(records, taxonomy, config, *, morgan=None):
    y = np.asarray([[int(label in row) for label in taxonomy] for row in records.labels], dtype=np.uint8)
    if morgan is None:
        morgan, _ = extract_features(records, config)
    groups = structure_feature_groups(records.smiles.tolist(), morgan)
    train, test = outer_split(groups, config["split"]["test_size"], config["split"]["seed"])
    positive = y[train].sum(axis=0)
    minimum = config["labels"]["min_train_positive"]
    chosen = [j for j in range(len(taxonomy)) if positive[j] >= minimum and len(train)-positive[j] >= 2]
    if not chosen:
        raise ValueError("No labels have enough training examples.")
    labels, target = [taxonomy[j] for j in chosen], y[:, chosen]
    folds, support = {}, []
    for j, label in enumerate(labels):
        pos = int(target[train,j].sum()); neg = len(train)-pos
        try:
            folds[label] = inner_folds(train, target[:,j], groups,
                                       config["split"]["max_folds"], config["split"]["seed"])
        except ValueError as exc:
            raise ValueError(f"Label {label}: {exc}") from exc
        support.append({"label":label,"train_positive":pos,"train_negative":neg,
                        "folds":len(folds[label]),
                        "positive_groups":len(set(groups[train][target[train,j] == 1])),
                        "negative_groups":len(set(groups[train][target[train,j] == 0]))})
    split = {"train":train.tolist(),"test":test.tolist(),"folds":folds,"eligible_labels":labels,
             "seed":config["split"]["seed"],"group_policy":POLICY,"groups":groups.tolist(),
             "outer_method":"GroupShuffleSplit","inner_method":"StratifiedGroupKFold",
             "test_size_unit":"groups"}
    validate_grouped_split(split, groups)
    selection = [{"label":label,"train_positive":int(positive[j]),"selected":j in chosen}
                 for j,label in enumerate(taxonomy)]
    return target, labels, split, pd.DataFrame(support), pd.DataFrame(selection)

