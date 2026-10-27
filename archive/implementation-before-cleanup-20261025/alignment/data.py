"""Prepare Suh's curated records without executing the author's notebook."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data/reference/suh2025"
OUTPUT = ROOT / "data/builds/suh-aligned-v1"
COLS = ["ifra1", "ifra2", "ifra3"]
DESCRIPTORS = ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA"]
EXPECTED = {
    "author_code.ipynb": "14df6a7ae7c8f46094467c17f7666633739d3c795340c47c63c6ac3c75ab2ea6",
    "Dataset2_final_w_molblock.csv": "c82d281a56006c06ca0a6232ee7941124f28439ba7f21850dd326edeceab1415",
    "Dataset1_data_preparation.xlsx": "295947d10c26a2401829799761db4d338e001bf326045f5ab25dcc69febee290",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def verify_reference(directory=REFERENCE):
    actual = {name: digest(Path(directory) / name) for name in EXPECTED}
    if actual != EXPECTED:
        raise ValueError("Reference bytes differ from verified OSF version 1.")
    return actual


def curate(raw):
    """Keep author row IDs through filtering; do not reset X/Y independently."""
    required = {"smiles", "molblock", *COLS}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing columns: {required - set(raw.columns)}")
    source = raw.copy()
    source.insert(0, "source_row", np.arange(len(source)))
    unique = source.drop_duplicates("smiles", keep="first")
    complete = unique.dropna(subset=["smiles", "molblock"]).copy()
    complete[COLS] = complete[COLS].apply(lambda s: s.str.upper())
    labels = sorted(set(complete[COLS].stack()))
    labeled = complete[complete[COLS].notna().any(axis=1)].reset_index(drop=True)
    # Match the author's alphabetically first target, not IFRA slot order.
    primary = labeled[COLS].apply(lambda row: min(row.dropna()), axis=1)
    keep = primary.map(primary.value_counts()) >= 2
    filtered = labeled.loc[keep].copy()
    filtered["stratify_label"] = primary.loc[keep]
    filtered = filtered.reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No records survive the reference strata filter.")
    wrong_rows = labeled.iloc[:len(filtered)]["source_row"].to_numpy()
    slots = labeled[COLS].stack().value_counts()
    summary_labels = sorted(slots[slots >= 30].index.tolist())
    audit = {
        "raw_rows": len(raw), "unique_smiles_rows": len(unique),
        "after_required_values": len(complete), "labeled_rows": len(labeled),
        "excluded_singleton_primary_strata": int((~keep).sum()),
        "benchmark_rows": len(filtered), "label_registry_count": len(labels),
        "paper_summary_label_count": len(summary_labels),
        "index_mapping_different_positions": int(np.sum(wrong_rows != filtered.source_row.to_numpy())),
        "author_actual_rows_absent_from_intended_cohort": len(set(wrong_rows) - set(filtered.source_row)),
        "excluded_source_rows": labeled.loc[~keep, "source_row"].astype(int).tolist(),
        "slot_count_rule": "Count non-null IFRA slots, uppercased, before singleton-stratum filtering; >=30.",
        "row_fix": "Filter records first; derive X, Y and split from the same retained row IDs.",
    }
    return filtered, labels, summary_labels, audit


def encode(frame, labels):
    lookup = {name: j for j, name in enumerate(labels)}
    y = np.zeros((len(frame), len(labels)), dtype=np.uint8)
    for i, row in enumerate(frame[COLS].itertuples(index=False, name=None)):
        for name in row:
            if pd.notna(name):
                if name not in lookup:
                    raise ValueError(f"Unknown label: {name}")
                y[i, lookup[name]] = 1
    return y


def features(frame):
    RDLogger.DisableLog("rdApp.warning")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=False)
    x = np.zeros((len(frame), 1024), dtype=np.uint8)
    desc = np.zeros((len(frame), 5), dtype=np.float32)
    identities, mismatches, bit_mismatches, fragments = [], [], [], []
    for i, row in enumerate(frame.itertuples(index=False)):
        smi_mol = Chem.MolFromSmiles(row.smiles)
        block_mol = Chem.MolFromMolBlock(row.molblock)
        if smi_mol is None or block_mol is None:
            raise ValueError(f"Invalid structure at source row {row.source_row}; do not silently zero-fill.")
        x[i] = generator.GetFingerprintAsNumPy(block_mol)
        desc[i] = [getattr(Descriptors, name)(smi_mol) for name in DESCRIPTORS]
        identities.append(Chem.MolToSmiles(smi_mol, isomericSmiles=True))
        if len(Chem.GetMolFrags(smi_mol)) > 1:
            fragments.append(int(row.source_row))
        if Chem.MolToSmiles(smi_mol, isomericSmiles=False) != Chem.MolToSmiles(block_mol, isomericSmiles=False):
            mismatches.append(int(row.source_row))
        if not np.array_equal(x[i], generator.GetFingerprintAsNumPy(smi_mol)):
            bit_mismatches.append(int(row.source_row))
    if not np.isfinite(desc).all():
        raise ValueError("Nonfinite descriptors.")
    if len(set(identities)) != len(identities):
        raise ValueError("Canonical duplicates require a group split before training.")
    return x, desc, {
        "canonical_duplicate_extra_rows": len(identities) - len(set(identities)),
        "multifragment_records": len(fragments),
        "multifragment_source_rows": fragments,
        "nonstereo_structure_mismatch_source_rows": mismatches,
        "morgan_smiles_molblock_mismatch_source_rows": bit_mismatches,
        "benchmark_scope": "Curated chemical records include disconnected structures; not validated perfume mixtures.",
        "inference_scope": "Existing public API remains connected-single-molecule only.",
    }


def partition(frame, y, labels, seed=42):
    train, test = train_test_split(np.arange(len(frame)), test_size=0.2,
                                  stratify=frame.stratify_label, random_state=seed)
    folds, eligible, support = {}, [], []
    for j, name in enumerate(labels):
        positive = int(y[train, j].sum())
        negative = len(train) - positive
        k = min(5, positive, negative)
        usable = k >= 2
        support.append({"label": name, "train_positive": positive, "train_negative": negative,
                        "folds": k if usable else 0, "eligible": usable})
        if usable:
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            folds[name] = [{"train": train[a].tolist(), "validation": train[b].tolist()}
                          for a, b in cv.split(train, y[train, j])]
            eligible.append(name)
    if set(train) & set(test):
        raise AssertionError("Train/test overlap.")
    return {"train": train.tolist(), "test": test.tolist(), "folds": folds,
            "eligible_labels": eligible, "seed": seed, "support": support}


def prepare(output=OUTPUT, reference=REFERENCE):
    output, reference = Path(output), Path(reference)
    verify_reference(reference)
    if output.exists():
        raise FileExistsError("Use a new build path; existing builds are immutable.")
    raw = pd.read_csv(reference / "Dataset2_final_w_molblock.csv")
    frame, labels, summary_labels, audit = curate(raw)
    y = encode(frame, labels)
    x, desc, structure_audit = features(frame)
    split = partition(frame, y, labels)
    audit.update(structure_audit)
    audit.update(train_rows=len(split["train"]), test_rows=len(split["test"]),
                 trainable_labels=len(split["eligible_labels"]),
                 untrainable_labels=sorted(set(labels) - set(split["eligible_labels"])))
    if not set(summary_labels).issubset(split["eligible_labels"]):
        raise ValueError("Primary summary labels are not all trainable.")
    output.mkdir(parents=True)
    # Keep held-out labels separate; CV/fit loaders never open test.npz.
    for part in ("train", "test"):
        rows = split[part]
        np.savez_compressed(output / f"{part}.npz", source_row=frame.source_row.to_numpy()[rows],
                            row_index=np.asarray(rows), morgan=x[rows], descriptors=desc[rows], truth=y[rows])
    frame.drop(columns=["molblock"]).to_csv(output / "records.csv", index=False)
    pd.DataFrame(split.pop("support")).to_csv(output / "train_label_support.csv", index=False)
    write_json(output / "splits.json", split)
    write_json(output / "audit.json", audit)
    spec = {"radius": 2, "n_bits": 1024, "include_chirality": False, "rdkit_version": rdBase.rdkitVersion,
            "morgan_source": "author MolBlock", "descriptor_source": "author SMILES",
            "descriptors": DESCRIPTORS, "dimensions": {"A_C": 1024, "B_D": 1029},
            "allow_disconnected_benchmark_records": True}
    manifest = {"protocol": "suh-aligned-v1", "reference_sha256": EXPECTED,
                "labels": labels, "summary_labels": summary_labels, "feature_spec": spec,
                "metric": "average_precision_macro_on_fixed_100_reference_labels",
                "split_policy": "author stratification, corrected row indices",
                "training_started": False, "test_evaluated": False,
                "files": {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / "dataset_manifest.json", manifest)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.reference), indent=2))
