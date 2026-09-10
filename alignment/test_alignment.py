import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alignment.data import curate, encode, partition, features
from alignment.metrics import binary_metrics, summarize, compare


def test_filter_keeps_record_and_label_identity():
    raw = pd.DataFrame({"smiles": ["CC", "CO", "CCC", "CCO", "CCN"],
                        "molblock": ["x"] * 5,
                        "ifra1": ["A", "RARE", "A", "B", "B"],
                        "ifra2": [None] * 5, "ifra3": [None] * 5})
    frame, labels, summary, audit = curate(raw)
    assert frame.source_row.tolist() == [0, 2, 3, 4]
    y = encode(frame, labels)
    assert y[:, labels.index("A")].tolist() == [1, 1, 0, 0]
    assert audit["index_mapping_different_positions"] == 3
    assert audit["excluded_singleton_primary_strata"] == 1


def test_fold_adapts_to_rare_label_without_test_leakage():
    frame = pd.DataFrame({"stratify_label": ["A"] * 40})
    y = np.zeros((40, 2), dtype=np.uint8)
    y[:20, 0] = 1
    y[:4, 1] = 1
    split = partition(frame, y, ["A", "RARE"])
    for j, label in enumerate(["A", "RARE"]):
        folds = split["folds"][label]
        assert len(folds) == min(5, int(y[split["train"], j].sum()))
        assert sorted(i for f in folds for i in f["validation"]) == sorted(split["train"])
        for f in folds:
            assert not set(f["train"]) & set(f["validation"])
            assert not (set(f["train"]) | set(f["validation"])) & set(split["test"])


def test_average_precision_is_not_trapezoid_and_decision_is_supplied():
    value = binary_metrics([1, 0, 1], [.9, .8, .1], [1, 1, 0])
    assert value["auprc"] == pytest.approx(5/6)
    assert value["accuracy"] == pytest.approx(1/3)
    assert value["specificity"] == 0
    assert value["precision"] == .5
    assert value["recall"] == .5


@pytest.mark.parametrize("truth,pred,ap", [([0, 0], [0, 0], 0), ([1, 1], [1, 1], 1)])
def test_single_class_defined_and_undefined_metrics(truth, pred, ap):
    value = binary_metrics(truth, [.2, .8], pred)
    assert value["auroc"] is None and value["specificity"] is None
    assert value["auprc"] == ap


def test_summary_rejects_denominator_shift():
    a = binary_metrics([0, 1], [.2, .8], [0, 1])
    b = binary_metrics([0, 0], [.2, .8], [0, 1])
    left = summarize({"A": a}, ["A"])
    right = summarize({"A": b}, ["A"])
    with pytest.raises(ValueError, match="coverage"):
        compare(left, right)


def test_reject_invalid_probability_and_registry():
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [np.nan, .8], [0, 1])
    with pytest.raises(ValueError):
        summarize({}, ["missing"])


def test_feature_equivalence_to_author_morgan():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles("CCO")
    block = Chem.MolToMolBlock(mol)
    x, desc, audit = features(pd.DataFrame([{"smiles":"CCO", "molblock":block, "source_row":0}]))
    original = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromMolBlock(block), 2, 1024)
    expected = np.zeros(1024, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(original, expected)
    assert np.array_equal(x[0], expected)
    assert desc.shape == (1, 5) and np.isfinite(desc).all()


def test_baseline_weighting_is_reference_specific():
    from alignment.experiments import make_model
    xgb = make_model("xgb", {"n_estimators":100})
    lgbm = make_model("lgbm", {"n_estimators":100})
    assert xgb.get_params()["scale_pos_weight"] is None
    assert lgbm.get_params()["class_weight"] == "balanced"


def test_deadline_stops_before_fit():
    from alignment.experiments import Limits
    with pytest.raises(TimeoutError):
        Limits(deadline=0).check()


def test_check_host_rejects_original_laptop(monkeypatch):
    import alignment.experiments as e
    amendment = e.read(e.ROOT / "reports/campus_20260909/full/amendment.json")
    monkeypatch.setattr(e.platform, "node", lambda: amendment["excluded_laptop_host"])
    with pytest.raises(RuntimeError, match="original laptop"):
        e.check_host()


class FakeModel:
    def predict_proba(self, x):
        p = np.where(x[:, 0] > .5, .8, .2)
        return np.column_stack([1-p, p])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] > .5).astype(int)


def test_full_orchestration_freezes_all_candidates_before_test(tmp_path, monkeypatch):
    import alignment.experiments as e
    dataset, run = tmp_path / "data", tmp_path / "run"
    dataset.mkdir()
    x = np.array([[0], [1], [0], [1], [0], [1], [0], [1]], dtype=np.uint8)
    y = np.column_stack([x[:, 0], 1-x[:, 0]])
    np.savez(dataset / "train.npz", row_index=np.arange(8), truth=y, morgan=x, descriptors=x.astype(float))
    np.savez(dataset / "test.npz", truth=y, morgan=x, descriptors=x.astype(float))
    (dataset / "dataset_manifest.json").write_text("{}")
    folds = [{"train":[0,1,2,3], "validation":[4,5,6,7]},
             {"train":[4,5,6,7], "validation":[0,1,2,3]}]
    e.write(dataset / "splits.json", {"folds":{"A":folds,"B":folds}, "eligible_labels":["A","B"]})
    meta = {"labels":["A","B"], "summary_labels":["A","B"]}
    monkeypatch.setattr(e, "check_host", lambda: None)
    monkeypatch.setattr(e, "preflight", lambda _: (meta, {}))
    monkeypatch.setattr(e, "fit", lambda *a, **k: FakeModel())
    monkeypatch.setattr(e, "tuning", lambda *a, **k: {"n_estimators":100})
    original_load = e.np.load
    test_opens = []
    def checked_load(path, *args, **kwargs):
        if Path(path).name == "test.npz":
            freeze = e.read(run / "selection_frozen.json")
            assert len(freeze["files"]) == 32
            test_opens.append(str(path))
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(e.np, "load", checked_load)
    result = e.run_all(dataset, run)
    assert len(e.read(result)["test"]) == 8
    assert len(test_opens) == 1
    monkeypatch.setattr(e, "fit", lambda *a, **k: pytest.fail("A frozen run must not fit again."))
    e.run_all(dataset, run)
    (run / "xgb_A/test_per_label.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        e.run_all(dataset, run)


def test_interrupted_tuning_reservation_is_charged(tmp_path):
    import alignment.experiments as e
    key = "xgb_C"
    e.write(tmp_path / key / "tuning_budget.json",
            {"charged_seconds":3600, "active_reservation":3600})
    with pytest.raises(RuntimeError, match="no completed trial"):
        e.tuning(tmp_path, key, "xgb", None, None, None, None, None,
                 {"tuning":{"seconds_per_study":7200, "attempts_per_study":15}})
    assert e.read(tmp_path / key / "tuning_budget.json")["charged_seconds"] == 7200
