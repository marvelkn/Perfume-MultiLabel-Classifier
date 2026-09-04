"""Fixed multilabel partitions; no model selection may load test labels."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold
def split_indices(Y, fraction, seed=42):
    if not 0 < fraction < 1:
        raise ValueError("Split fraction must be between zero and one")
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    return next(splitter.split(np.zeros((len(Y), 1)), Y))
def scaffold_groups(smiles):
    # All acyclic molecules share the empty Murcko scaffold; do not silently split it.
    return np.asarray([MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s),includeChirality=True) for s in smiles])

def group_split_indices(Y, groups, fraction, seed=42):
    splitter=GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed)
    return next(splitter.split(np.zeros((len(Y),1)),Y,groups))

def stratified_split(X, Y):
    from .config import CONFIG
    return split_indices(Y, CONFIG["split"]["test_size"], CONFIG["seed"])
def development_partitions(Y, *, threshold_size=.15, n_folds=3, early_stopping_size=.15, seed=42, groups=None):
    if groups is not None:
        groups=np.asarray(groups)
        fit,threshold=group_split_indices(Y,groups,threshold_size,seed)
    else:
        fit, threshold = split_indices(Y, threshold_size, seed)
    folds = []
    cv = (GroupKFold(n_splits=n_folds,shuffle=True,random_state=seed) if groups is not None else
          MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed))
    for number, (train, score) in enumerate(cv.split(np.zeros((len(fit), 1)), Y[fit], groups[fit] if groups is not None else None)):
        inner_train, stop = (group_split_indices(Y[fit[train]],groups[fit[train]],early_stopping_size,seed+number+1)
                             if groups is not None else split_indices(Y[fit[train]], early_stopping_size, seed + number + 1))
        folds.append({"train": fit[train[inner_train]].tolist(),
                      "stop": fit[train[stop]].tolist(), "score": fit[score].tolist()})
    return {"fit": fit.tolist(), "threshold": threshold.tolist(), "folds": folds}
