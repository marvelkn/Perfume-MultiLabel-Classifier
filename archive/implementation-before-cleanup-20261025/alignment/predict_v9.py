"""Infer one connected molecule using the selected V9 research bundle."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from rdkit.Chem import rdFingerprintGenerator

from . import experiments as core
from .data import ROOT, digest
from .descriptors_v7 import DESCRIPTORS8, calculate, molecule
from .experiments_v9 import RUN_ID, verify_files


def predict(run, smiles):
    run = Path(run)
    bundle = core.read(run / "inference_bundle.json")
    if bundle["protocol"] != RUN_ID or bundle["environment"] != core.environment():
        raise ValueError("Bundle/environment mismatch.")
    if bundle["selection_sha256"] != digest(run / "selection_frozen.json"):
        raise ValueError("Selection changed.")
    spec = bundle["feature_spec"]
    if spec["n_bits"] not in (1024, 2048) or spec["descriptors"] != list(DESCRIPTORS8):
        raise ValueError("Wrong feature schema.")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=spec["radius"], fpSize=spec["n_bits"], includeChirality=spec["include_chirality"])
    fp = gen.GetFingerprintAsNumPy(molecule(smiles, 0))
    x = np.concatenate((fp, calculate([smiles])[0])).astype(np.float32)[None, :]
    folder = run / bundle["models_directory"]
    verify_files(folder, {v["file"]: v["sha256"] for v in bundle["models"].values()})
    scores = [float(joblib.load(folder / bundle["models"][n]["file"]).predict_proba(x)[0, 1]) for n in bundle["labels"]]
    return {"candidate": bundle["selected_candidate"], "smiles": smiles, "threshold": 0.5,
            "interpretation": bundle["score_interpretation"],
            "labels": [{"label": n, "score": s, "predicted": s >= .5} for n, s in zip(bundle["labels"], scores)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / RUN_ID)
    parser.add_argument("--smiles", required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.run, args.smiles), indent=2))
