"""Contract tests only: no model fitting, trial creation, or dataset writes."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from research import dataset, experiment, features, inference
from research.runtime import ROOT, digest, read
from research.results import main_run


@pytest.fixture(scope="module")
def verified():
    return dataset.verify()


def test_frozen_data_and_current_feature_pairing(verified):
    meta, split, _, morgan, descriptors = verified
    assert len(meta["labels"]) == 109
    assert morgan.shape == (6686, 2048)
    assert descriptors.shape == (6686, 8)
    for role in ("train", "test"):
        part = dataset.load_partition(role, verified=verified)
        assert part["row_index"].tolist() == split[role]
        assert np.array_equal(part["morgan"], morgan[split[role]])
        assert np.array_equal(part["descriptors"], descriptors[split[role]])


def test_no_historical_module_imports():
    for path in (ROOT / "research").glob("*.py"):
        importlib.import_module("research." + path.stem)
    assert not any(name == "alignment" or name.startswith("alignment.")
                   or name == "src" or name.startswith("src.") for name in sys.modules)


def test_archived_model_inference_matches_previous_path():
    expected = read(ROOT / "reports/research_cleanup_20261025/baseline_predictions.json")
    for smiles, result in expected.items():
        assert inference.predict(main_run(), smiles) == result


def test_old_run_rejected_before_writing():
    before = digest(main_run() / "identity.json")
    with pytest.raises(ValueError, match="Historical run"):
        experiment.run_all(dataset.DATASET, main_run())
    assert digest(main_run() / "identity.json") == before


def test_primary_config_preserves_method():
    old = read(ROOT / "archive/implementation-before-cleanup-20261025/alignment/protocol_v7.json")
    new = read(experiment.PROTOCOL)
    for key in ("environment", "algorithms", "conditions", "group_policy", "class_imbalance",
                "baseline_parameters", "tuning", "search_space", "robustness",
                "threshold_selection", "primary_reporting", "features", "initial_recipes"):
        assert old[key] == new[key], key


def test_single_molecule_schema_and_invalid_input():
    smiles = ["COc1cc(C=O)ccc1O"]
    morgan, descriptors = features.calculate_morgan(smiles), features.calculate(smiles)
    for condition in "ABCD":
        expected = 2056 if condition in "BD" else 2048
        assert features.features("lgbm_" + condition, morgan, descriptors).shape == (1, expected)
    with pytest.raises(ValueError):
        features.calculate(["CCO.O"])


def test_no_heavy_calls_in_learning_notebook():
    import ast
    notebook = read(ROOT / "KODE_ML/BELAJAR_KODE_ML_LENGKAP.ipynb")
    forbidden = {"fit", "fit_final_models", "run_all", "tune", "optimize", "ask", "prepare", "review_frozen_preprocessing"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
                assert name not in forbidden
