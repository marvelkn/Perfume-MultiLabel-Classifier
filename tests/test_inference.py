"""Contract rejection tests without training or loading research estimators."""
import json
from pathlib import Path
import numpy as np
import pytest
from src.artifacts import digest, write_json
from src.featurize import DESCRIPTORS, FeatureSpec, features_and_metadata
from src.release import contained, runtime_spec, verify_release
from src.inference import Predictor


def bundle():
    return {"feature_spec": {"radius": 2, "n_bits": 1024, "include_chirality": False,
            "descriptors": list(DESCRIPTORS), "fragment_policy": "single",
            "rdkit_version": FeatureSpec().rdkit_version, "dimensions": {"B_D": 1029}},
            "use_descriptors": True}


def test_legacy_feature_schema_is_distinct():
    active, legacy = FeatureSpec(), FeatureSpec(n_bits=2048)
    assert active.n_features == 1029 and active.schema_id != legacy.schema_id


def test_reversed_descriptor_order_rejected():
    info = bundle()
    info["feature_spec"]["descriptors"].reverse()
    with pytest.raises(ValueError, match="Descriptor order"):
        runtime_spec(info)


def test_training_dimension_mismatch_rejected():
    info = bundle()
    info["feature_spec"]["dimensions"]["B_D"] = 2053
    with pytest.raises(ValueError, match="dimension"):
        runtime_spec(info)


@pytest.mark.parametrize("name", ["../outside.joblib", "models/../../outside.joblib"])
def test_artifact_path_cannot_escape_release(tmp_path, name):
    with pytest.raises(ValueError, match="escapes"):
        contained(tmp_path, name)


def test_corruption_rejected_before_deserialization(tmp_path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"original")
    manifest = {"format_version": 1, "selected_candidate": "lgbm_D", "files": {artifact.name: digest(artifact)}}
    write_json(tmp_path / "release.json", manifest)
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_release(tmp_path)


@pytest.mark.parametrize("matrix", [np.zeros((1, 2053)), np.full((1, 1029), np.nan), np.zeros((0, 1029))])
def test_bad_feature_matrix_rejected_before_prediction(matrix):
    predictor = Predictor.__new__(Predictor)
    predictor.spec = FeatureSpec()
    predictor.models = []
    with pytest.raises(ValueError):
        predictor.predict_features(matrix)


def test_equivalent_smiles_have_identical_features():
    first, meta = features_and_metadata("CCOC(=O)C1=CC=CC=C1")
    second, alternate = features_and_metadata("O=C(OCC)c1ccccc1")
    assert meta["smiles"] == alternate["smiles"]
    np.testing.assert_array_equal(first, second)
