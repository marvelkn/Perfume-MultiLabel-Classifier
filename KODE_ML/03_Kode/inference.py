"""Prediksi satu molekul memakai model penelitian tersimpan dan skema yang sama."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from . import runtime as core
from .runtime import ROOT, digest
from .features import DESCRIPTORS8, N_BITS


def predict(run, smiles):
    run = Path(run)
    bundle = core.read(run/"inference_bundle.json")
    completion = core.read(run/"completion_manifest.json")["files"]
    if digest(run/"inference_bundle.json") != completion["inference_bundle.json"]:
        raise ValueError("Inference bundle changed.")
    if bundle["environment"] != core.environment():
        raise ValueError("Use the pinned training environment.")
    spec = bundle["feature_spec"]
    names = spec["descriptors"]
    if (names not in ([], list(DESCRIPTORS8)) or spec["n_bits"] != N_BITS
            or bool(names) != bundle["use_descriptors"]):
        raise ValueError("Unsupported descriptor order.")
    if spec.get("rdkit_version", rdBase.rdkitVersion) != rdBase.rdkitVersion:
        raise ValueError("RDKit version differs.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Input must be one valid connected molecule.")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=spec["radius"], fpSize=spec["n_bits"], includeChirality=spec["include_chirality"])
    x = np.concatenate([generator.GetFingerprintAsNumPy(mol),
                        [getattr(Descriptors,name)(mol) for name in names]]).astype(np.float32)[None,:]
    if not np.isfinite(x).all() or x.shape[1] != spec["dimensions"]["input"]:
        raise ValueError("Feature schema mismatch.")
    output = []
    for label in bundle["labels"]:
        entry = bundle["models"][label]
        path = run/bundle["models_directory"]/entry["file"]
        if digest(path) != entry["sha256"]:
            raise ValueError("Model checksum changed.")
        model = joblib.load(path)
        if model.n_features_in_ != x.shape[1]:
            raise ValueError("Model expects a different feature count.")
        # The ndarray feature order is frozen and checked above.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names", category=UserWarning)
            score = float(model.predict_proba(x)[0,1])
        output.append({"label":label, "score":score, "predicted":score>=bundle["primary_threshold"]})
    return {"smiles":Chem.MolToSmiles(mol), "candidate":bundle["selected_candidate"],
            "descriptors":len(names), "threshold":bundle["primary_threshold"],
            "score_meaning":"uncalibrated model score, not aroma intensity",
            "labels":sorted(output,key=lambda row:row["score"],reverse=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=None)
    parser.add_argument("--smiles",required=True)
    args = parser.parse_args()
    from .results import main_run
    print(json.dumps(predict(args.run or main_run(),args.smiles),indent=2))


if __name__ == "__main__":
    main()
