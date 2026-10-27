"""Pasangkan fitur utama dengan target dan split beku, tanpa membaca fitur lama."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .runtime import ROOT, read, digest
from .features import EXTRA, N_BITS, DESCRIPTORS8, calculate, calculate_morgan
from .splits import POLICY, validate_grouped_split

CONFIG = ROOT / "research/config/dataset.json"
DATASET = ROOT / "data/builds/perfume-five-grouped-v2"


def verify(dataset=DATASET, extra=EXTRA, *, recompute=False):
    """Periksa identitas beku; recompute hanya menghitung ulang fitur 2048+8."""
    dataset, extra = Path(dataset), Path(extra)
    contract = read(CONFIG)
    for folder, entries in ((dataset, contract["dataset_files"]), (extra, contract["feature_files"])):
        for name, expected in entries.items():
            if digest(folder / name) != expected:
                raise ValueError(f"Frozen input changed: {folder / name}")
    meta = read(dataset / "dataset_manifest.json")
    split = read(dataset / "splits.json")
    feature_meta = read(extra / "manifest.json")
    for folder, manifest in ((dataset, meta), (extra, feature_meta)):
        for name, expected in manifest["files"].items():
            if digest(folder / name) != expected:
                raise ValueError(f"Original provenance hash changed: {name}")
    records = pd.read_csv(dataset / "records.csv")
    stored_groups = pd.read_csv(dataset / "groups.csv")
    labels = meta["labels"]
    if (len(records) != contract["rows"] or len(labels) != contract["label_count"]
            or len(set(labels)) != len(labels) or labels != split["eligible_labels"]
            or meta["group_policy"] != POLICY or records.smiles.isna().any()
            or records.smiles.duplicated().any()):
        raise ValueError("Frozen cohort or label registry changed.")
    if (stored_groups.smiles.tolist() != records.smiles.tolist()
            or stored_groups.source_row.tolist() != records.source_row.tolist()):
        raise ValueError("Group row identity differs from the molecular records.")
    validate_grouped_split(split, stored_groups.group_id.to_numpy())
    if feature_meta["dataset_manifest_sha256"] != digest(dataset / "dataset_manifest.json"):
        raise ValueError("Feature bundle belongs to another cohort.")
    spec = feature_meta["feature_spec"]
    if (spec["n_bits"] != N_BITS or spec["radius"] != 2
            or spec["include_chirality"] is not False or spec["descriptors"] != list(DESCRIPTORS8)):
        raise ValueError("Expected the main 2048+8 feature schema.")
    with np.load(extra / "features.npz", allow_pickle=False) as z:
        morgan, descriptors = z["morgan"], z["descriptors"]
        if (z["row_index"].tolist() != list(range(len(records)))
                or z["source_row"].tolist() != records.source_row.tolist()
                or z["names"].tolist() != list(DESCRIPTORS8)):
            raise ValueError("Feature order or molecular identity changed.")
    if (morgan.shape != (len(records), N_BITS) or morgan.dtype != np.uint8
            or descriptors.shape != (len(records), 8) or descriptors.dtype != np.float32
            or not np.isin(morgan, [0, 1]).all() or not np.isfinite(descriptors).all()):
        raise ValueError("Invalid feature values or dimensions.")
    for role in ("train", "test"):
        # Test labels are not decoded during preflight; file bytes are only hashed above.
        with np.load(dataset / f"{role}.npz", allow_pickle=False) as z:
            if (z["row_index"].tolist() != split[role]
                    or z["source_row"].tolist() != records.source_row.to_numpy()[split[role]].tolist()
                    or len(split[role]) != contract[f"{role}_rows"]):
                raise ValueError(f"{role}: frozen row order changed.")
            if role == "train":
                target = z["truth"]
                if target.shape != (len(split[role]), len(labels)) or not np.isin(target, [0, 1]).all():
                    raise ValueError("Invalid training target matrix.")
    positions = {row: i for i, row in enumerate(split["train"])}
    for j, name in enumerate(labels):
        for fold in split["folds"][name]:
            for part in ("train", "validation"):
                if np.unique(target[[positions[i] for i in fold[part]], j]).size != 2:
                    raise ValueError(f"Fold for {name} lacks one target class.")
    if recompute:
        smiles = records.smiles.tolist()
        if not np.array_equal(calculate_morgan(smiles), morgan) or not np.array_equal(calculate(smiles), descriptors):
            raise ValueError("2048+8 values differ from recomputed molecular features.")
    # The old manifest remains untouched. Return an in-memory view with actual model features.
    return dict(meta, feature_spec=spec), split, feature_meta, morgan, descriptors


def load_partition(role, dataset=DATASET, extra=EXTRA, *, verified=None):
    if role not in ("train", "test"):
        raise ValueError("Partition must be train or test.")
    meta, split, _, morgan, descriptors = verified or verify(dataset, extra)
    with np.load(Path(dataset) / f"{role}.npz", allow_pickle=False) as z:
        result = {key: z[key].copy() for key in ("row_index", "source_row", "truth")}
    rows = result["row_index"]
    if (rows.tolist() != split[role] or result["truth"].shape != (len(rows), len(meta["labels"]))
            or not np.isin(result["truth"], [0, 1]).all()):
        raise ValueError("Partition target or row order changed.")
    result.update(morgan=morgan[rows], descriptors=descriptors[rows])
    return result


def encode_frozen_records(records, dataset=DATASET, *, verified=None):
    """Encode labels with the frozen registry; never select new labels or shuffle rows."""
    meta, split, _, _, _ = verified or verify(dataset)
    original = pd.read_csv(Path(dataset) / "records.csv")
    for name in ("labels", "sources"):
        original[name] = original[name].map(json.loads)
    pd.testing.assert_frame_equal(records.reset_index(drop=True), original, check_dtype=True)
    target = np.asarray([[int(name in row) for name in meta["labels"]]
                         for row in records.labels], dtype=np.uint8)
    for role in ("train", "test"):
        with np.load(Path(dataset) / f"{role}.npz", allow_pickle=False) as z:
            if not np.array_equal(target[split[role]], z["truth"]):
                raise ValueError("Encoded labels differ from frozen targets.")
    return target, meta["labels"], split
