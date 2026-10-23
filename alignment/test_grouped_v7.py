"""Exercise v7 feature identity, selection, resume and inference on tiny fixtures."""
import copy
import json
import shutil
import zipfile

import joblib
import numpy as np
import pandas as pd
import pytest
from rdkit import rdBase

from alignment import descriptors_v7 as descriptors
from alignment import experiments_v7 as v7
from alignment import tuning_v6 as tuning
from alignment.perfume_data import OUTPUT


def test_real_preflight_descriptor_parity_and_no_row_loss():
    meta, protocol = v7.preflight(OUTPUT)
    manifest, morgan, values = descriptors.verify()
    assert morgan.shape == (6686, 2048)
    assert values.shape == (6686, 8) and len(meta["labels"]) == 109
    assert np.isin(morgan, [0, 1]).all()
    assert meta["feature_spec"]["n_bits"] == 2048
    assert v7.core.read(OUTPUT / "dataset_manifest.json")["feature_spec"]["n_bits"] == 1024
    assert manifest["rows_removed"] == 0
    assert protocol["tuning"]["seconds_total"] == 32 * 3600
    assert len(v7.CANDIDATES) == 8 and len(v7.TUNED) == 4
    assert {key[-1] for key in v7.CANDIDATES} == set("ABCD")
    assert {key[-1] for key in v7.TUNED} == set("CD")


def test_descriptor_values_and_invalid_molecules():
    values = descriptors.calculate(["CCO", "c1ccccc1"])
    assert values[0, 0] == pytest.approx(46.069, abs=.001)
    assert values[0, 6:].tolist() == [3, 0]
    assert values[1, 6:].tolist() == [6, 1]
    morgan = descriptors.calculate_morgan(["CCO", "c1ccccc1"])
    assert morgan.shape == (2, 2048) and morgan.dtype == np.uint8
    assert np.isin(morgan, [0, 1]).all() and np.any(morgan[0] != morgan[1])
    for smiles in ("bad_smiles", "CC.O"):
        for calculate in (descriptors.calculate, descriptors.calculate_morgan):
            with pytest.raises(ValueError):
                calculate([smiles])


def test_rows_descriptors_and_fingerprint_parity_cannot_change(tmp_path):
    records = pd.DataFrame({"source_row": [10, 20, 30, 40], "smiles": ["CCO"] * 4})
    records.to_csv(tmp_path / "records.csv", index=False)
    values = np.arange(32, dtype=np.float32).reshape(4, 8)
    morgan = np.zeros((4, 2048), dtype=np.uint8)
    morgan[:, 1200] = 1
    folded = np.bitwise_or(morgan[:, :1024], morgan[:, 1024:])
    v7.core.write(tmp_path / "splits.json", {"train": [2, 0], "test": [3, 1]})
    for role, indices in (("train", [2, 0]), ("test", [3, 1])):
        v7.core.save_npz(tmp_path / f"{role}.npz", row_index=np.asarray(indices),
                         descriptors=values[indices, :5], morgan=folded[indices])
    args = (np.arange(4), records.source_row.to_numpy(), morgan)
    descriptors.validate_values(tmp_path, values, *args)
    with pytest.raises(ValueError):
        descriptors.validate_values(tmp_path, values, np.array([1, 0, 2, 3]), args[1], morgan)
    with pytest.raises(ValueError):
        descriptors.validate_values(tmp_path, values, args[0], args[1][::-1], morgan)
    changed = values.copy()
    changed[0, 0] += 1
    with pytest.raises(ValueError):
        descriptors.validate_values(tmp_path, changed, *args)
    changed_morgan = morgan.copy()
    changed_morgan[0, 5] = 1
    with pytest.raises(ValueError):
        descriptors.validate_values(tmp_path, values, args[0], args[1], changed_morgan)


def test_ablation_adds_eight_features_and_rejects_legacy_fingerprints():
    morgan = np.zeros((4, 2048), dtype=np.uint8)
    values = np.arange(32, dtype=np.float32).reshape(4, 8)
    for algorithm in v7.ALGORITHMS:
        for left, right in (("A", "B"), ("C", "D")):
            plain = v7.features(f"{algorithm}_{left}", morgan, values)
            augmented = v7.features(f"{algorithm}_{right}", morgan, values)
            assert plain.shape == (4, 2048) and augmented.shape == (4, 2056)
            np.testing.assert_array_equal(plain, augmented[:, :2048])
            np.testing.assert_array_equal(augmented[:, 2048:], values)
            with pytest.raises(ValueError):
                v7.features(f"{algorithm}_{right}", morgan[:, :1024], values)
    with pytest.raises(ValueError):
        v7.features("xgb_UNKNOWN", morgan, values)
    with pytest.raises(ValueError):
        v7.features("xgb_B", morgan, values[:, :5])


def test_selection_ignores_test_and_prefers_baseline_on_exact_tie():
    values = {key: {"mean_metrics": {"auprc": .3},
                    "test_ap": .99 if key[-1] == "D" else .01}
              for key in v7.CANDIDATES}
    _, selected, _ = v7.select_candidates(values)
    assert selected == {"xgb": "xgb_A", "lgbm": "lgbm_A"}
    values["xgb_D"]["mean_metrics"]["auprc"] = .4
    assert v7.select_candidates(values)[0] == "xgb_D"


def test_fresh_equal_studies_and_resume_budget(tmp_path):
    protocol = v7.core.read(v7.PROTOCOL)
    for algorithm in v7.ALGORITHMS:
        for condition in "CD":
            key = f"{algorithm}_{condition}"
            study = tuning.initialize(tmp_path, key, protocol, protocol["initial_recipes"][algorithm])
            assert len(study.trials) == 1
            assert study.trials[0].params == {}
            budget = tmp_path / key / "budget.json"
            saved = v7.core.read(budget)
            assert saved["budget_seconds"] == 8 * 3600
            saved["charged_seconds"] = 120
            v7.core.write(budget, saved)
            tuning.initialize(tmp_path, key, protocol, protocol["initial_recipes"][algorithm])
            assert v7.core.read(budget)["charged_seconds"] == 120
            changed = copy.deepcopy(protocol)
            changed["tuning"]["seconds_per_study"] += 1
            with pytest.raises(ValueError, match="budget"):
                tuning.initialize(tmp_path, key, changed, protocol["initial_recipes"][algorithm])


def assert_each_candidate_can_predict(run, tmp_path):
    from alignment.predict_v7 import predict

    original_bundle = v7.core.read(run / "inference_bundle.json")
    morgan = descriptors.calculate_morgan(["CCO"])
    values = descriptors.calculate(["CCO"])
    for key in v7.CANDIDATES:
        fixture = tmp_path / f"inference_{key}"
        shutil.copytree(run / key, fixture / key)
        bundle = copy.deepcopy(original_bundle)
        count = v7.descriptor_count(key)
        bundle.update(selected_candidate=key, algorithm=key.split("_")[0],
                      models_directory=key, models=v7.core.read(run / key / "models.json"),
                      use_descriptors=bool(count))
        bundle["feature_spec"].update(n_bits=2048, descriptors=list(descriptors.DESCRIPTORS8[:count]),
                                      use_descriptors=bool(count), dimensions={"input": 2048 + count})
        bundle["feature_names"] = ([f"morgan_{index:04d}" for index in range(2048)]
                                    + list(descriptors.DESCRIPTORS8[:count]))
        v7.core.write(fixture / "inference_bundle.json", bundle)
        v7.core.write(fixture / "completion_manifest.json", {
            "files": {"inference_bundle.json": v7.digest(fixture / "inference_bundle.json")}})
        result = predict(fixture, "CCO")
        scores = {row["label"]: row["score"] for row in result["labels"]}
        assert set(scores) == set(bundle["labels"])
        expected_features = v7.features(key, morgan, values)
        for label in bundle["labels"]:
            model = joblib.load(fixture / key / bundle["models"][label]["file"])
            assert model.n_features_in_ == 2048 + count
            expected = model.predict_proba(expected_features)[0, 1]
            assert scores[label] == pytest.approx(expected)
        assert all(0 <= score <= 1 for score in scores.values())


def test_freeze_interrupt_resume_models_metrics_and_inference(tmp_path, monkeypatch):
    protocol = v7.core.read(v7.PROTOCOL)
    protocol["robustness"].update(seeds=[7, 42], folds=2)
    dataset, run, extra, reference = (tmp_path / name for name in ("dataset", "run", "extra", "reference"))
    for folder in (dataset, extra, reference):
        folder.mkdir()
    labels = ["sweet", "woody"]
    truth = np.array([[0, 0], [1, 0], [0, 1], [1, 1]] * 8, dtype=np.int8)
    morgan = np.zeros((40, 2048), dtype=np.uint8)
    morgan[:, :2] = np.concatenate([truth, truth[:8]])
    values = np.tile(np.arange(8, dtype=np.float32), (40, 1))
    # Deliberately unusable legacy matrices detect accidental fallback to 1024 bits.
    v7.core.save_npz(dataset / "train.npz", truth=truth, morgan=np.zeros((32, 1024), dtype=np.uint8),
                     descriptors=values[:32, :5], row_index=np.arange(32))
    v7.core.save_npz(dataset / "test.npz", truth=truth[:8], morgan=np.zeros((8, 1024), dtype=np.uint8),
                     descriptors=values[32:, :5], row_index=np.arange(32, 40))
    v7.core.write(dataset / "dataset_manifest.json", {"synthetic": True})
    v7.core.write(dataset / "splits.json", {"train": list(range(32)), "test": list(range(32, 40))})
    v7.core.write(extra / "manifest.json", {"synthetic": True})
    pd.DataFrame({"group_id": np.arange(40)}).to_csv(dataset / "groups.csv", index=False)
    v7.core.write(reference / "manifest.json", {"synthetic": True})
    history = [{"candidate": f"{algorithm}_{condition}", "cv_ap_mean": .2,
                **{metric: .2 for metric in v7.core.METRICS}}
               for algorithm in v7.ALGORITHMS for condition in "ABCD"]
    pd.DataFrame(history).to_csv(reference / "comparison_all_v6.csv", index=False)
    protocol_path = tmp_path / "protocol.json"
    v7.core.write(protocol_path, protocol)
    meta = {"labels": labels, "summary_labels": labels,
            "feature_spec": {"n_bits": 2048, "radius": 2, "include_chirality": False,
                             "rdkit_version": rdBase.rdkitVersion,
                             "descriptors": list(descriptors.DESCRIPTORS8)}}
    monkeypatch.setattr(v7, "PROTOCOL", protocol_path)
    monkeypatch.setattr(v7, "REFERENCE", reference)
    monkeypatch.setattr(v7, "preflight", lambda *args: (meta, protocol))
    monkeypatch.setattr(v7, "verify_descriptors", lambda *args: ({}, morgan, values))
    monkeypatch.setattr(v7.core, "check_host", lambda: None)
    monkeypatch.setattr(v7.core.Limits, "check", lambda *args, **kwargs: None)
    profile = v7.core.compute_profile()
    profile["threads_per_model"] = 1
    monkeypatch.setattr(v7.core, "compute_profile", lambda: profile)
    original_make = v7.core.make_model
    monkeypatch.setattr(v7.core, "make_model", lambda algorithm, params:
                        original_make(algorithm, {"n_estimators": 3, "max_depth": 2}))
    tuned = []

    def fake_tune(run, key, *args):
        tuned.append(key)
        v7.core.write(run / key / "tuning_summary.json", {"budget": {"status": "complete"}})
        return {"n_estimators": 3, "max_depth": 2}

    monkeypatch.setattr(v7, "tune", fake_tune)
    evaluate = v7.evaluate_test

    def interrupted(*args):
        assert (run / "selection_frozen.json").exists()
        raise KeyboardInterrupt

    monkeypatch.setattr(v7, "evaluate_test", interrupted)
    with pytest.raises(KeyboardInterrupt):
        v7.run_all(dataset, run, extra)
    frozen = v7.digest(run / "selection_frozen.json")
    assert len(tuned) == 4 and not (run / "complete.json").exists()
    monkeypatch.setattr(v7, "evaluate_test", evaluate)
    v7.run_all(dataset, run, extra)
    assert frozen == v7.digest(run / "selection_frozen.json")
    assert len(tuned) == 4
    comparison = pd.read_csv(run / "comparison_all.csv")
    assert len(comparison) == 8
    assert set(comparison.descriptors) == {0, 8}
    assert len(pd.read_csv(run / "descriptor_deltas.csv")) == 8
    assert v7.core.read(run / "paired_descriptor_cv.json")["identical_paired_folds"]
    assert_each_candidate_can_predict(run, tmp_path)
    v7.run_all(dataset, run, extra)
    assert len(tuned) == 4
    (run / "xgb_A" / "recipe.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        v7.run_all(dataset, run, extra)


def test_package_and_collection_roundtrip(tmp_path, monkeypatch):
    from alignment import package_campus_v7 as pack
    from alignment import collect_results_v7 as collector

    v7.core.write(tmp_path / "input.json", {"data": "included"})
    monkeypatch.setattr(pack, "paths", lambda root: {root / "input.json"})
    monkeypatch.setattr(collector, "paths", lambda root: {root / "input.json"})
    packaged = pack.build(tmp_path)
    assert packaged["archive"].endswith("perfume-campus-grouped-v7-m2048-d8.zip")
    with zipfile.ZipFile(packaged["archive"]) as archive:
        assert "input.json" in archive.namelist()
        assert archive.testzip() is None
        manifest = json.loads(archive.read("CAMPUS_PACKAGE_MANIFEST.json"))
        assert manifest["protocol"] == v7.RUN_ID
    run = tmp_path / "runs" / v7.RUN_ID
    v7.core.write(run / "progress.json", {"partial": True})
    collected = collector.collect(tmp_path)
    with zipfile.ZipFile(collected["archive"]) as archive:
        manifest = json.loads(archive.read("RESULTS_MANIFEST.json"))
        assert not manifest["complete"] and manifest["protocol"] == v7.RUN_ID
        assert f"runs/{v7.RUN_ID}/progress.json" in archive.namelist()
    v7.core.write(run / ".alignment.lock", {})
    with pytest.raises(RuntimeError, match="aktif"):
        collector.collect(tmp_path)
