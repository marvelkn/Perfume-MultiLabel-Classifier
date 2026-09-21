"""Check frozen features and sampled inference, without fitting or computing new metrics."""
from pathlib import Path
import json
import sys
import time
import warnings
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from rdkit import Chem
from api_light import SPEC, fingerprint_result
from src.artifacts import digest, write_json
from src.featurize import features_and_metadata, MoleculeError
from src.inference import Predictor
from src.release import DEFAULT_RELEASE, read

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / ".local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v6/perfume-campus-grouped-v6"
RUN = PACKAGE / "runs/perfume-five-grouped-v6"
DATA = PACKAGE / "data/builds/perfume-five-grouped-v2"
OUT = Path(__file__).resolve().parent
start = time.perf_counter()
identity = read(RUN / "identity.json")
assert digest(DATA / "dataset_manifest.json") == identity["dataset"]
meta = read(DATA / "dataset_manifest.json")
for name, expected in meta["files"].items():
    assert digest(DATA / name) == expected, name
records = pd.read_csv(DATA / "records.csv")
train = np.load(DATA / "train.npz", allow_pickle=False)
test = np.load(DATA / "test.npz", allow_pickle=False)
feature_rows = 0
for part in (train, test):
    for i, record_index in enumerate(part["row_index"]):
        vector, _ = features_and_metadata(records.iloc[int(record_index)].smiles, SPEC)
        expected = np.concatenate((part["morgan"][i], part["descriptors"][i])).astype(np.float32)
        np.testing.assert_array_equal(vector, expected)
        feature_rows += 1
print(f"Exact feature parity: {feature_rows} molecules", flush=True)
predictor = Predictor()
assert predictor.spec.schema_id == SPEC.schema_id
assert predictor.labels == meta["labels"]
indices = np.unique(np.linspace(0, len(test["row_index"])-1, 32, dtype=int))
smiles = [records.iloc[int(test["row_index"][i])].smiles for i in indices]
matrix = np.stack([features_and_metadata(s, predictor.spec)[0] for s in smiles])
with warnings.catch_warnings():
    warnings.simplefilter("error", UserWarning)
    scores = predictor.predict_features(matrix)
completion = read(RUN / "completion_manifest.json")["files"]
reference_path = "lgbm_D/test_predictions.npz"
assert digest(RUN / reference_path) == completion[reference_path]
with np.load(RUN / reference_path, allow_pickle=False) as stored:
    assert stored["labels"].tolist() == predictor.labels
    reference = stored["probabilities"][indices]
    max_error = float(np.max(np.abs(reference-scores)))
    np.testing.assert_allclose(scores, reference, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(scores >= .5, stored["decisions"][indices])
example = predictor.predict(smiles[0])
repeat = predictor.predict(smiles[0])
assert example == repeat
mol = Chem.MolFromSmiles(smiles[0])
alternate = Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=mol.GetNumAtoms()-1)
assert predictor.predict(alternate) == example
assert fingerprint_result(smiles[0])["fingerprint"] == matrix[0].tolist()
assert len(example["predictions"]) == 109
for invalid, expected in [("", "INVALID_SMILES"), ("bad!", "INVALID_SMILES"), ("CCO.O", "UNSUPPORTED_MIXTURE")]:
    try:
        predictor.predict(invalid)
    except MoleculeError as exc:
        assert exc.code == expected
    else:
        raise AssertionError("Invalid molecule accepted")
try:
    predictor.predict(smiles[0], feature_schema_id="legacy-schema")
except MoleculeError as exc:
    assert exc.code == "FEATURE_SCHEMA_MISMATCH"
else:
    raise AssertionError("Wrong schema accepted")
# Check that presentation permits zero positives, without altering a real estimator.
original_method = predictor.predict_features
predictor.predict_features = lambda matrix: np.zeros((len(matrix),109), dtype=np.float32)
assert predictor.predict(smiles[0])["predicted_labels"] == []
predictor.predict_features = original_method
result = {"status": "passed", "feature_parity": {"rows": feature_rows, "features": 1029, "comparison": "exact"},
          "prediction_parity": {"test_rows": len(indices), "labels": 109, "scores_compared": int(scores.size),
                                "tolerance_absolute": 1e-6, "max_absolute_error": max_error, "decisions_identical": True,
                                "sampling": "32 evenly spaced test positions; no performance-based selection",
                                "test_positions": indices.tolist()},
          "contracts": {"api_feature_parity": True, "equivalent_smiles": True, "repeatability": True,
                        "invalid_inputs_rejected": True, "wrong_schema_rejected": True, "empty_positive_set_allowed": True},
          "verified_dataset_files": len(meta["files"]), "verified_release_files": len(predictor.manifest["files"]),
          "model_count": len(predictor.models), "feature_schema_id": SPEC.schema_id,
          "release_manifest_sha256": digest(DEFAULT_RELEASE / "release.json"),
          "dataset_manifest_sha256": digest(DATA / "dataset_manifest.json"),
          "reference_predictions_sha256": digest(RUN / reference_path),
          "source_files": {name: digest(ROOT / name) for name in ("src/inference.py", "src/release.py", "src/featurize.py", "api_light.py", "feature_spec.json")},
          "duration_seconds": time.perf_counter()-start,
          "scope": "Implementation parity only; no training, new accuracy estimate, ONNX, Android or deployment"}
write_json(OUT / "verification.json", result)
write_json(OUT / "prediction_example.json", example)
print(json.dumps(result, indent=2))
