"""Local Python inference for the frozen model; no training or network calls."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from .artifacts import digest
from .featurize import MoleculeError, features_and_metadata
from .release import DEFAULT_RELEASE, contained, read, verify_release


class Predictor:
    def __init__(self, release=DEFAULT_RELEASE):
        self.directory = Path(release).resolve()
        self.manifest, self.spec = verify_release(self.directory)
        self.manifest_sha256 = digest(self.directory / "release.json")
        for package, expected in self.manifest["environment"].items():
            if version(package) != expected:
                raise ValueError(f"Inference requires {package}=={expected}; installed {version(package)}")
        self.labels = self.manifest["labels"]
        self.models = []
        for label in self.labels:
            path = contained(self.directory, self.manifest["models"][label])
            # Only deserialize local, checksum-verified project artifacts.
            model = joblib.load(path)
            if model.__class__.__module__ != "lightgbm.sklearn" or model.__class__.__name__ != "LGBMClassifier":
                raise ValueError(f"Unexpected estimator for {label}")
            if model.n_features_in_ != self.spec.n_features or list(model.classes_) != [0, 1]:
                raise ValueError(f"Model feature/class mismatch: {label}")
            self.models.append(model)

    def predict_features(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != self.spec.n_features:
            raise ValueError("Expected a non-empty matrix of 1029 features")
        if not np.isfinite(matrix).all():
            raise ValueError("Features must be finite")
        probabilities = []
        for model in self.models:
            # Names are assigned after ordering by the frozen schema, not guessed from input.
            frame = pd.DataFrame(matrix, columns=model.feature_name_)
            probabilities.append(model.predict_proba(frame, num_threads=1)[:, 1])
        scores = np.column_stack(probabilities).astype(np.float32)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Model produced invalid scores")
        return scores

    def predict(self, smiles, feature_schema_id=None):
        if feature_schema_id is not None and feature_schema_id != self.spec.schema_id:
            raise MoleculeError("FEATURE_SCHEMA_MISMATCH", "Features and model have different schemas.")
        if not isinstance(smiles, str) or len(smiles) > 10000:
            raise MoleculeError("INVALID_INPUT", "Enter a SMILES string of at most 10000 characters.")
        vector, metadata = features_and_metadata(smiles, self.spec)
        scores = self.predict_features(vector.reshape(1, -1))[0]
        threshold = 0.5
        predictions = [{"label": label, "score": float(score), "threshold": threshold,
                        "predicted": bool(score >= threshold)} for label, score in zip(self.labels, scores)]
        return {"status": "ok", **metadata, "model_id": self.manifest["release_id"],
                "selected_candidate": self.manifest["selected_candidate"],
                "model_manifest_sha256": self.manifest_sha256,
                "threshold_policy": "primary_fixed_0.5", "n_features": self.spec.n_features,
                "label_count": len(self.labels), "score_interpretation": self.manifest["score_interpretation"],
                "predictions": predictions,
                "predicted_labels": [row["label"] for row in predictions if row["predicted"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--feature-schema-id")
    args = parser.parse_args()
    try:
        # Reject invalid molecules before loading the model package.
        features_and_metadata(args.smiles)
        result = Predictor(args.release).predict(args.smiles, args.feature_schema_id)
    except (MoleculeError, ValueError, OSError) as exc:
        code = exc.code if isinstance(exc, MoleculeError) else "INFERENCE_UNAVAILABLE"
        print(json.dumps({"status": "error", "error": {"code": code, "message": str(exc)}}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
