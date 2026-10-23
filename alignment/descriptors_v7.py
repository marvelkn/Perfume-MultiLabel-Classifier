"""Build Morgan 2048 plus eight descriptors on the unchanged frozen cohort."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from . import experiments as core
from .data import ROOT, digest
from .perfume_data import OUTPUT

N_BITS = 2048
RADIUS = 2
DESCRIPTORS5 = ("MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA")
DESCRIPTORS8 = DESCRIPTORS5 + ("NumRotatableBonds", "HeavyAtomCount", "RingCount")
EXTRA = ROOT / "data/builds/perfume-m2048-d8-v7"


def molecule(smiles, index):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"Invalid or multicomponent structure at row {index}; no rows may be dropped.")
    return mol


def calculate(smiles):
    values = np.empty((len(smiles), 8), dtype=np.float32)
    for i, value in enumerate(smiles):
        mol = molecule(value, i)
        values[i] = [getattr(Descriptors, name)(mol) for name in DESCRIPTORS8]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite descriptors; no silent imputation.")
    return values


def calculate_morgan(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, fpSize=N_BITS, includeChirality=False)
    return np.asarray([generator.GetFingerprintAsNumPy(molecule(smi, i))
                       for i, smi in enumerate(smiles)], dtype=np.uint8)


def validate_values(dataset, values, row_index, source_rows, morgan):
    records = pd.read_csv(Path(dataset) / "records.csv")
    split = core.read(Path(dataset) / "splits.json")
    if (values.shape != (len(records), 8) or not np.isfinite(values).all()
            or not np.array_equal(row_index, np.arange(len(records)))
            or not np.array_equal(source_rows, records.source_row.to_numpy())):
        raise ValueError("Feature row identity, descriptor shape, or values changed.")
    if morgan.shape != (len(records), N_BITS) or not np.isin(morgan, [0, 1]).all():
        raise ValueError("Expected binary Morgan 2048 for every row.")
    # With the same Morgan parameters, OR-folding verifies correspondence to v6.
    folded = np.maximum(morgan[:, :1024], morgan[:, 1024:])
    for role in ("train", "test"):
        # Feature identity checks never read test targets or model scores.
        with np.load(Path(dataset) / f"{role}.npz", allow_pickle=False) as saved:
            indices = saved["row_index"].astype(int)
            if indices.tolist() != split[role]:
                raise ValueError(f"{role}: row order differs from frozen split.")
            if not np.array_equal(values[indices, :5], saved["descriptors"]):
                raise ValueError(f"{role}: first five descriptors differ from v6.")
            if not np.array_equal(folded[indices], saved["morgan"]):
                raise ValueError(f"{role}: Morgan 2048 does not fold to the frozen 1024 bits.")
    return records


def verify(dataset=OUTPUT, extra=EXTRA):
    dataset, extra = Path(dataset), Path(extra)
    manifest = core.read(extra / "manifest.json")
    spec = manifest["feature_spec"]
    if (manifest["id"] != "perfume-m2048-d8-v7"
            or manifest["dataset_manifest_sha256"] != digest(dataset / "dataset_manifest.json")
            or spec["rdkit_version"] != rdBase.rdkitVersion
            or spec["descriptors"] != list(DESCRIPTORS8)
            or spec["n_bits"] != N_BITS or spec["radius"] != RADIUS
            or spec["include_chirality"] is not False):
        raise ValueError("Feature build belongs to another dataset, schema, or RDKit version.")
    for name, expected in manifest["files"].items():
        if digest(extra / name) != expected:
            raise ValueError(f"Feature artifact changed: {name}")
    with np.load(extra / "features.npz", allow_pickle=False) as saved:
        values, morgan = saved["descriptors"], saved["morgan"]
        if saved["names"].tolist() != list(DESCRIPTORS8):
            raise ValueError("Descriptor names changed.")
        records = validate_values(dataset, values, saved["row_index"], saved["source_row"], morgan)
    if manifest["rows"] != len(records):
        raise ValueError("Feature manifest row count changed.")
    return manifest, morgan, values


def prepare(dataset=OUTPUT, extra=EXTRA):
    dataset, extra = Path(dataset), Path(extra)
    if (extra / "manifest.json").exists():
        return verify(dataset, extra)[0]
    if extra.exists() and any(extra.iterdir()):
        raise ValueError("Incomplete feature directory; use a new output directory.")
    meta, _ = core.preflight(dataset)
    if meta["feature_spec"]["descriptors"] != list(DESCRIPTORS5):
        raise ValueError("Expected the original five-descriptor order.")
    records = pd.read_csv(dataset / "records.csv")
    values = calculate(records.smiles.tolist())
    morgan = calculate_morgan(records.smiles.tolist())
    indices = np.arange(len(records))
    validate_values(dataset, values, indices, records.source_row.to_numpy(), morgan)
    extra.mkdir(parents=True, exist_ok=True)
    core.save_npz(extra / "features.npz", morgan=morgan, descriptors=values,
                  row_index=indices, source_row=records.source_row.to_numpy(),
                  names=np.asarray(DESCRIPTORS8))
    frame = pd.DataFrame(values, columns=DESCRIPTORS8)
    frame.insert(0, "smiles", records.smiles)
    frame.insert(0, "source_row", records.source_row)
    frame.to_csv(extra / "descriptors.csv", index=False)
    manifest = {
        "id": "perfume-m2048-d8-v7",
        "dataset_manifest_sha256": digest(dataset / "dataset_manifest.json"),
        "rows": len(records),
        "feature_spec": {"n_bits":N_BITS, "radius":RADIUS, "include_chirality":False,
                         "descriptors":list(DESCRIPTORS8), "rdkit_version":rdBase.rdkitVersion,
                         "dtype":"float32", "fragment_policy":"single"},
        "dimensions": {"A_C":N_BITS, "B_D":N_BITS+8},
        "first_five_match_frozen_train_and_test": True,
        "morgan_2048_folds_exactly_to_frozen_1024": True,
        "group_policy": "Keep original 1024-based groups; any identical 2048 fingerprint also folds identically.",
        "rows_removed": 0, "labels_or_splits_changed": False,
        "descriptor_source": "Eight types from Suh et al. 2025; their combination with Morgan is a project experiment.",
        "files": {name: digest(extra / name) for name in ("features.npz", "descriptors.csv")},
    }
    core.write(extra / "manifest.json", manifest)
    verify(dataset, extra)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, default=EXTRA)
    args = parser.parse_args()
    print(prepare(args.dataset, args.output))


if __name__ == "__main__":
    main()
