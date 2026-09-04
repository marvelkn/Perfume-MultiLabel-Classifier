import json
from types import SimpleNamespace
import numpy as np
import optuna
import pytest
from src.featurize import FeatureSpec
from src.runtime import ResourceLimit
import src.experiments as experiments
from src.artifacts import digest

def test_interrupted_final_fit_resumes_without_retraining_completed_label(tmp_path,monkeypatch):
    rng=np.random.default_rng(1)
    X=rng.random((64,13)).astype(np.float32)
    Y=(X[:,:2]>.5).astype(np.uint8)
    cfg={"seed":42,"splits":{"fit":list(range(48)),"threshold":list(range(48,64))},"resources":{"threads":2}}
    spec=FeatureSpec(n_bits=8)
    meta={"dataset_id":"synthetic","labels":["a","b"],"feature_spec":spec.to_dict()}
    monkeypatch.setattr(experiments,"context",lambda run:(tmp_path,cfg,X,Y,meta))
    monkeypatch.setattr(experiments,"ResourceGuard",lambda *a:SimpleNamespace(check=lambda **kw:None))
    study=optuna.create_study(storage=f"sqlite:///{(tmp_path/'studies.sqlite3').as_posix()}",study_name="xgb",direction="maximize")
    study.set_user_attr("strategies",["none"])
    study.set_user_attr("signature",experiments.study_signature(meta,cfg,"xgb",["none"]))
    trial=optuna.trial.create_trial(value=.5,user_attrs={"model_params":{"max_depth":2},"imbalance":"none","best_iterations":[[8,8],[8,8],[8,8]]})
    study.add_trial(trial)
    original=experiments.fit_binary
    calls=[0]
    def interrupted(*a,**kw):
        calls[0]+=1
        if calls[0]==2:raise ResourceLimit("Synthetic interruption")
        return original(*a,**kw)
    monkeypatch.setattr(experiments,"fit_binary",interrupted)
    with pytest.raises(ResourceLimit):experiments.finalize(tmp_path,"xgb")
    first=tmp_path/"xgb"/"xgb_000.pkl"
    checksum=digest(first)
    assert not (tmp_path/"xgb"/"model_manifest.json").exists()
    monkeypatch.setattr(experiments,"fit_binary",original)
    output=experiments.finalize(tmp_path,"xgb")
    assert digest(first)==checksum
    assert len(json.loads((output/"model_manifest.json").read_text())["models"])==2
