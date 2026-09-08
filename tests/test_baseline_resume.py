import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from src import experiments
from src.artifacts import digest
from src.featurize import FeatureSpec
from src.runtime import ResourceLimit


@pytest.fixture
def simulated_baseline(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    X = rng.random((8, 13)).astype(np.float32)
    Y = np.column_stack([np.arange(8) % 2, 1 - np.arange(8) % 2]).astype(np.uint8)
    cfg = {"seed":42, "resources":{"threads":1}, "splits":{"folds":[
        {"train":[0,1,2,3], "stop":[4,5], "score":[6,7]},
        {"train":[4,5,6,7], "stop":[0,1], "score":[2,3]}]}}
    meta = {"dataset_id":"synthetic", "labels":["a","b"],
            "feature_spec":FeatureSpec(n_bits=8).to_dict()}
    @contextmanager
    def fake_context(run, *args):
        from pathlib import Path
        yield Path(run), cfg, X, Y, meta, SimpleNamespace(check=lambda **kw: None)
    monkeypatch.setattr(experiments, "guarded_context", fake_context)
    monkeypatch.setattr(experiments, "study_signature", lambda *args: "synthetic-code-signature")
    calls = {"completed":0, "stop_after":None}
    def fit(*args, **kwargs):
        if calls["stop_after"] is not None and calls["completed"] >= calls["stop_after"]:
            raise ResourceLimit("simulated interruption")
        calls["completed"] += 1
        return SimpleNamespace(predict_proba=lambda matrix: np.column_stack([1-matrix[:,0],matrix[:,0]])), 8
    monkeypatch.setattr(experiments, "fit_binary", fit)
    return cfg, calls


@pytest.mark.parametrize("algorithm", ["xgb", "lgbm"])
def test_baseline_resumes_each_completed_label_and_matches_uninterrupted(tmp_path, simulated_baseline, algorithm):
    _cfg, calls = simulated_baseline
    resumed = tmp_path / "resumed"; resumed.mkdir()
    calls["stop_after"] = 3
    with pytest.raises(ResourceLimit):
        experiments.baseline(resumed, algorithm)
    files = list((resumed / f"{algorithm}_baseline_checkpoints").rglob("*.npz"))
    assert len(files) == 3 and not (resumed / f"{algorithm}_baselines.json").exists()
    saved = {path: (digest(path), path.stat().st_mtime_ns) for path in files}
    calls["stop_after"] = None
    result = experiments.baseline(resumed, algorithm)
    assert calls["completed"] == 12
    assert saved == {path:(digest(path),path.stat().st_mtime_ns) for path in files}
    uninterrupted = tmp_path / "uninterrupted"; uninterrupted.mkdir()
    expected = experiments.baseline(uninterrupted, algorithm)
    assert result == expected
    with pytest.raises(FileExistsError):
        experiments.baseline(resumed, algorithm)


@pytest.mark.parametrize("corruption", ["seed", "code", "prediction", "legacy"])
def test_resume_rejects_changed_protocol_or_corrupt_predictions(tmp_path, simulated_baseline, monkeypatch, corruption):
    cfg, calls = simulated_baseline
    calls["stop_after"] = 1
    with pytest.raises(ResourceLimit):
        experiments.baseline(tmp_path, "xgb")
    if corruption == "seed":
        cfg["seed"] += 1
    elif corruption == "code":
        monkeypatch.setattr(experiments, "study_signature", lambda *args: "changed-code")
    elif corruption == "prediction":
        next((tmp_path/"xgb_baseline_checkpoints").rglob("*.npz")).write_bytes(b"corrupt")
    else:
        (tmp_path/"xgb_baseline_progress.json").write_text('{"none":{}}')
    calls["stop_after"] = None
    with pytest.raises(ValueError, match="differs|unversioned|integrity"):
        experiments.baseline(tmp_path, "xgb")
    assert calls["completed"] == 1


def test_resume_recomputes_scores_from_verified_predictions(tmp_path, simulated_baseline):
    _cfg, calls = simulated_baseline
    calls["stop_after"] = 4
    with pytest.raises(ResourceLimit):
        experiments.baseline(tmp_path, "xgb")
    path = tmp_path / "xgb_baseline_progress.json"
    progress = json.loads(path.read_text())
    progress["results"]["none"]["mean_ap"] = -999
    path.write_text(json.dumps(progress))
    calls["stop_after"] = None
    result = experiments.baseline(tmp_path, "xgb")
    assert 0 <= result["none"]["mean_ap"] <= 1
    assert calls["completed"] == 12
