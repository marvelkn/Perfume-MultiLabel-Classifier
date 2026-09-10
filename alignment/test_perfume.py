"""Regression tests for the active five-source pipeline."""
from collections import Counter
import json
import numpy as np
import pandas as pd
import pytest
from alignment import perfume_data as p
from alignment import experiments as e

def test_sources_are_exactly_the_requested_five(tmp_path):
    config=p.load_config()
    assert set(config["sources"]) == {"goodscents","ifra","leffingwell","arctander","sigma"}
    config["sources"]["flavornet"]={}
    path=tmp_path/"bad.json"; path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Exactly"):
        p.load_config(path)

def test_ifra_missing_descriptors_do_not_become_labels():
    frame=pd.DataFrame({"Descriptor 1":["Floral",None],"Descriptor 2":[np.nan,"Woody"],
                        "Descriptor 3":["",None]})
    assert p.taxonomy_from_ifra(frame) == ["floral","woody"]

def test_sigma_text_metadata_is_not_a_binary_label():
    row=pd.Series({"Stimulus":"1","descriptors":"sweet;woody","sweet":"1","woody":"0"})
    assert p.tokens_for("sigma",row,row.index) == ["sweet"]
    row["sweet"]="broken"
    with pytest.raises(ValueError,match="Invalid binary"):
        p.tokens_for("sigma",row,row.index)

def test_unknown_labels_are_logged_without_guessing():
    unknown,mappings=Counter(),Counter()
    result=p.map_labels(["wood",None,np.nan,"alien odor","woody"],{"woody"},"test",unknown,mappings)
    assert result == {"woody"}
    assert unknown == {("test","alien odor"):1}
    assert mappings[("test","wood","woody","alias")] == 1

def test_label_selection_uses_training_only_and_has_no_top25_cap(monkeypatch):
    config=p.load_config(); config["labels"]["min_train_positive"]=2
    taxonomy=[f"label{i}" for i in range(31)]+["test_only"]
    rows=[taxonomy[:31] if i%2==0 else [] for i in range(12)]+[["test_only"]]*4
    records=pd.DataFrame({"labels":rows})
    monkeypatch.setattr(p,"split_indices",lambda *a:(np.arange(12),np.arange(12,16)))
    y,labels,split,support,_=p.encode_and_split(records,taxonomy,config)
    assert len(labels)==31 and "test_only" not in labels and y.shape==(16,31)
    for folds in split["folds"].values():
        assert len(folds)==5
        assert sorted(i for f in folds for i in f["validation"])==list(range(12))
        assert all(not (set(f["train"])|set(f["validation"])) & set(range(12,16)) for f in folds)

def test_features_are_dynamic_and_invalid_structures_fail():
    config=p.load_config(); config["features"]["n_bits"]=128
    x,d=p.extract_features(pd.DataFrame({"smiles":["CCO","CO"]}),config)
    assert x.shape==(2,128) and d.shape==(2,5) and np.isfinite(d).all()
    with pytest.raises(ValueError,match="Invalid molecule"):
        p.extract_features(pd.DataFrame({"smiles":["CC.O"]}),config)

@pytest.mark.parametrize("algorithm",["xgb","lgbm"])
def test_real_estimators_optuna_and_resume_on_tiny_synthetic_data(tmp_path,monkeypatch,algorithm):
    # Four trees on synthetic data exercise callbacks/serialization without research training.
    x=np.tile(np.array([[0.,0.,1.],[1.,1.,0.]],dtype=np.float32),(24,1))
    y=np.column_stack([x[:,0],1-x[:,0],x[:,0]]).astype(np.uint8)
    labels=["one","two","three"]
    folds=[{"train":list(range(24)),"validation":list(range(24,48))},
           {"train":list(range(24,48)),"validation":list(range(24))}]
    split={"folds":{label:folds for label in labels}}
    key=algorithm+"_C"; (tmp_path/key).mkdir()
    monkeypatch.setattr(e,"suggest",lambda trial,a:{"n_estimators":4,"max_depth":2})
    protocol={"tuning":{"seconds_per_study":30,"attempts_per_study":1}}
    args=(tmp_path,key,algorithm,x,y,{"labels":labels,"summary_labels":labels},split,
          dict(enumerate(range(48))),protocol)
    best=e.tuning(*args)
    assert best["n_estimators"]==4
    assert e.tuning(*args)==best
    model=e.fit(algorithm,best,x,y[:,0],e.Limits())
    path=tmp_path/"model.joblib"; e.joblib.dump(model,path)
    restored=e.joblib.load(path)
    assert restored.predict_proba(x).shape==(48,2)
    assert np.isfinite(restored.predict_proba(x)).all()
