"""Build the model-ready dataset from GoodScents, Leffingwell, Arctander, Sigma, Flavornet.

Run:  python -m src.build_dataset

Pipeline: load archives -> attach SMILES (CAS/CID bridge) -> harmonize all sources
onto the Leffingwell taxonomy -> canonicalize + dedupe (union labels) ->
Morgan fingerprints -> min-positive label floor -> iterative-stratified 80/20 split
-> save artifacts to data/processed/.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import CONFIG, path
from .data_acquisition import load_all
from .featurize import canonical_smiles, fingerprint_matrix
from .label_harmonization import LabelHarmonizer, normalize
from .splits import stratified_split


# ----- column detection (robust to minor schema variation) -------------------
def _find_col(df: pd.DataFrame, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _cid_col(df):
    c = _find_col(df, ["CID", "PubChemCID", "pubchem_cid"])
    if c is None:
        raise KeyError(f"No CID column; columns={list(df.columns)}")
    return c


def _smiles_col(df):
    c = _find_col(df, ["IsomericSMILES", "SMILES", "CanonicalSMILES", "smiles"])
    if c is None:
        raise KeyError(f"No SMILES column; columns={list(df.columns)}")
    return c


def _descriptor_col(df):
    c = _find_col(df, ["Descriptors", "descriptors", "odor", "odors", "tags"])
    if c is None:
        raise KeyError(f"No descriptor column; columns={list(df.columns)}")
    return c


def _truthy(v) -> bool:
    try:
        return float(v) > 0
    except (TypeError, ValueError):
        return str(v).strip().lower() not in ("", "0", "false", "nan", "none")


def _attach_smiles(beh: pd.DataFrame, sti: pd.DataFrame, mol: pd.DataFrame) -> pd.DataFrame:
    """Attach `__smiles__` to a behavior frame, robust to index scheme.

    Path A: behavior shares the Stimulus index with stimuli (which carries CID).
    Path B: the behavior index already IS the CID.
    """
    smi_by_cid = mol[_smiles_col(mol)]
    cid = _cid_col(sti)

    a = beh.join(sti[[cid]], how="left")
    a["__smiles__"] = a[cid].map(smi_by_cid)

    if a["__smiles__"].notna().mean() < 0.5:
        b = beh.copy()
        b["__smiles__"] = pd.Series(b.index, index=b.index).map(smi_by_cid)
        if b["__smiles__"].notna().mean() > a["__smiles__"].notna().mean():
            return b
    return a


# ----- per-source record extraction ------------------------------------------
def _records_leffingwell(dfs, taxonomy_norm):
    beh = dfs["leffingwell_behavior"]
    label_cols = list(beh.columns)
    norm = {c: normalize(c) for c in label_cols}
    tax = set(taxonomy_norm)
    frame = _attach_smiles(beh, dfs["leffingwell_stimuli"], dfs["leffingwell_molecules"])
    records = []
    for _, row in frame.iterrows():
        labels = {norm[c] for c in label_cols if _truthy(row[c])} & tax
        records.append((row["__smiles__"], labels))
    return records


def _records_goodscents(dfs, harmonizer):
    beh = dfs["goodscents_behavior"]
    desc = _descriptor_col(beh)
    frame = _attach_smiles(beh, dfs["goodscents_stimuli"], dfs["goodscents_molecules"])
    return [(row["__smiles__"], harmonizer.map_row(row[desc])) for _, row in frame.iterrows()]


def _records_arctander(dfs, harmonizer):
    """Arctander: behavior_1_sparse has 'Stimulus' + 'Labels' (semicolon-separated text).
    stimuli has 'Stimulus' + 'new_CID'; molecules indexed by CID with IsomericSMILES.
    """
    beh = dfs["arctander_behavior"]
    sti = dfs["arctander_stimuli"]
    mol = dfs["arctander_molecules"]

    # Build CID -> SMILES map
    smi_col = _smiles_col(mol)
    cid_col_name = "CID" if "CID" in mol.columns else mol.columns[0]
    smi_by_cid = dict(zip(mol[cid_col_name], mol[smi_col]))

    # Stimulus -> CID from stimuli
    stim_to_cid = dict(zip(sti["Stimulus"], sti["new_CID"]))

    records = []
    for _, row in beh.iterrows():
        label_cell = row.get("Labels", "")
        if not isinstance(label_cell, str) or not label_cell.strip():
            continue
        cid = stim_to_cid.get(row["Stimulus"])
        smiles = smi_by_cid.get(cid)
        if smiles:
            labels = harmonizer.map_row(label_cell)
            records.append((smiles, labels))
    return records


def _records_sigma(dfs, taxonomy_norm):
    """Sigma 2014: behavior is a binary matrix (True/False columns per label).
    stimuli has 'Stimulus' + 'CAS'; molecules has CID + IsomericSMILES.
    We bridge via CAS -> CID using the stimuli table if available.
    """
    beh = dfs["sigma_behavior"]
    mol = dfs["sigma_molecules"]
    sti = dfs["sigma_stimuli"]

    smi_col = _smiles_col(mol)
    cid_col_name = "CID" if "CID" in mol.columns else mol.columns[0]
    smi_by_cid = dict(zip(mol[cid_col_name], mol[smi_col]))

    tax = set(taxonomy_norm)
    from .label_harmonization import normalize

    # The boolean columns (skip 'Stimulus' and 'descriptors')
    skip_cols = {"stimulus", "descriptors"}
    label_cols = [c for c in beh.columns if c.lower() not in skip_cols]
    norm_map = {c: normalize(c) for c in label_cols}

    # stimuli: Stimulus -> CAS, but molecules indexed by CID not CAS
    # Use molecules directly by matching on index or CID
    # Sigma stimuli has no CID, so we match molecules by index position where possible
    # Better: use IsomericSMILES from molecules keyed by CID, joined via stimulus order
    # Since stimulus IDs are negative ints, we rely on mol having IsomericSMILES in order
    # Use the 'descriptors' column text as fallback SMILES source is not possible;
    # instead join stimuli -> mol via CID if available, else skip
    mol_by_stimulus = {}
    if "CID" in sti.columns:
        cid_sti = dict(zip(sti["Stimulus"], sti["CID"]))
        for stim, cid in cid_sti.items():
            smi = smi_by_cid.get(cid)
            if smi:
                mol_by_stimulus[stim] = smi
    else:
        # No CID in stimuli: try to match by order
        mol_smiles_list = mol[smi_col].tolist()
        for i, stim in enumerate(sti["Stimulus"]):
            if i < len(mol_smiles_list):
                mol_by_stimulus[stim] = mol_smiles_list[i]

    records = []
    for _, row in beh.iterrows():
        smiles = mol_by_stimulus.get(row["Stimulus"])
        if not smiles:
            continue
        labels = {norm_map[c] for c in label_cols if _truthy(row[c]) and norm_map[c] in tax}
        records.append((smiles, labels))
    return records


def _records_flavornet(dfs, harmonizer):
    """Flavornet: behavior has 'Stimulus' (= CID) + 'Descriptors' (semicolon text).
    stimuli has 'Stimulus' + 'CID'. molecules has CID + IsomericSMILES.
    """
    beh = dfs["flavornet_behavior"]
    mol = dfs["flavornet_molecules"]

    smi_col = _smiles_col(mol)
    cid_col_name = "CID" if "CID" in mol.columns else mol.columns[0]
    smi_by_cid = dict(zip(mol[cid_col_name], mol[smi_col]))

    records = []
    for _, row in beh.iterrows():
        cid = row.get("Stimulus")
        smiles = smi_by_cid.get(cid)
        desc = row.get("Descriptors", "")
        if smiles and isinstance(desc, str) and desc.strip():
            labels = harmonizer.map_row(desc)
            records.append((smiles, labels))
    return records


# ----- orchestration ---------------------------------------------------------
def build():
    min_pos = int(CONFIG["labels"]["min_positive_count"])

    print("[1/6] Loading archives via pyrfume ...")
    dfs = load_all(save_raw=True)

    taxonomy = sorted({normalize(c) for c in dfs["leffingwell_behavior"].columns})
    print(f"      Leffingwell taxonomy: {len(taxonomy)} classes")

    print("[2/6] Harmonizing labels + attaching SMILES ...")
    harmonizer = LabelHarmonizer(taxonomy)
    records = (
        _records_leffingwell(dfs, taxonomy)
        + _records_goodscents(dfs, harmonizer)
        + (_records_arctander(dfs, harmonizer) if "arctander_behavior" in dfs else [])
        + (_records_sigma(dfs, taxonomy) if "sigma_behavior" in dfs else [])
        + (_records_flavornet(dfs, harmonizer) if "flavornet_behavior" in dfs else [])
    )
    print(f"      raw rows across sources: {len(records)}")

    unmapped = harmonizer.unmapped_report()
    unmapped.to_csv(path("data_interim") / "unmapped_descriptors.csv", index=False)
    print(f"      unmapped GoodScents descriptors: {len(unmapped)} "
          "(data/interim/unmapped_descriptors.csv)")

    print("[3/6] Canonicalizing + deduping (union labels) ...")
    merged = defaultdict(set)
    parse_fail = 0
    for smi, labels in records:
        c = canonical_smiles(smi)
        if c is None:
            parse_fail += 1
            continue
        merged[c] |= labels
    print(f"      unique molecules: {len(merged)}  (SMILES parse failures: {parse_fail})")

    smiles = list(merged.keys())

    print("[4/6] Morgan fingerprints ...")
    X, mask = fingerprint_matrix(smiles)
    smiles = [s for s, m in zip(smiles, mask) if m]
    label_index = {lab: i for i, lab in enumerate(taxonomy)}
    Y = np.zeros((len(smiles), len(taxonomy)), dtype=np.uint8)
    for r, s in enumerate(smiles):
        for lab in merged[s]:
            j = label_index.get(lab)
            if j is not None:
                Y[r, j] = 1

    col_pos = Y.sum(axis=0)
    blocklist = {normalize(b) for b in CONFIG["labels"].get("blocklist", [])}
    keep = (col_pos >= min_pos) & np.array([t not in blocklist for t in taxonomy])
    taxonomy_kept = [t for t, k in zip(taxonomy, keep) if k]
    Y = Y[:, keep]
    print(f"[5/6] Label floor (>= {min_pos}) & blocklist: kept {len(taxonomy_kept)}/{len(taxonomy)} labels")

    row_has = Y.sum(axis=1) > 0
    X, Y = X[row_has], Y[row_has]
    smiles = [s for s, h in zip(smiles, row_has) if h]
    print(f"      final: X={X.shape}  Y={Y.shape}")

    print("[6/6] Iterative-stratified 80/20 split + save ...")
    train_idx, test_idx = stratified_split(X, Y)
    smiles_train = [smiles[i] for i in train_idx]
    smiles_test = [smiles[i] for i in test_idx]
    assert not (set(smiles_train) & set(smiles_test)), "LEAKAGE: molecule in both splits!"

    proc = path("data_processed")
    sp.save_npz(proc / "X_train.npz", sp.csr_matrix(X[train_idx]))
    sp.save_npz(proc / "X_test.npz", sp.csr_matrix(X[test_idx]))
    sp.save_npz(proc / "Y_train.npz", sp.csr_matrix(Y[train_idx]))
    sp.save_npz(proc / "Y_test.npz", sp.csr_matrix(Y[test_idx]))
    (proc / "label_names.json").write_text(json.dumps(taxonomy_kept, indent=2), encoding="utf-8")
    (proc / "smiles_train.txt").write_text("\n".join(smiles_train), encoding="utf-8")
    (proc / "smiles_test.txt").write_text("\n".join(smiles_test), encoding="utf-8")

    inter = pd.concat(
        [pd.DataFrame({"canonical_smiles": smiles}), pd.DataFrame(Y, columns=taxonomy_kept)],
        axis=1,
    )
    inter.to_csv(path("data_interim") / "dataset_harmonized.csv", index=False)

    train_pos = Y[train_idx].sum(axis=0)
    summary = {
        "seed": CONFIG["seed"],
        "fingerprint": CONFIG["fingerprint"],
        "n_records_raw": len(records),
        "n_unique_canonical": len(merged),
        "n_smiles_parse_fail": parse_fail,
        "n_labels_taxonomy": len(taxonomy),
        "n_labels_kept": len(taxonomy_kept),
        "min_positive_count": min_pos,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "x_n_features": int(X.shape[1]),
        "train_test_disjoint": True,
        "train_positives_per_label": {l: int(n) for l, n in zip(taxonomy_kept, train_pos)},
    }
    (proc / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done. Artifacts in", proc)
    return summary


if __name__ == "__main__":
    build()
