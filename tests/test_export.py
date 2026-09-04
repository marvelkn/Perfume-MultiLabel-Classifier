import json
import joblib
import numpy as np
import pytest
from src.experiments import fit_binary
from src.featurize import FeatureSpec
from src.artifacts import digest, write_json
from export_for_mobile import export_bundle
from src.explain_model import contributions

@pytest.mark.parametrize("algorithm",["xgb","lgbm"])
@pytest.mark.parametrize("early_stop",[False,True])
def test_synthetic_two_label_export_parity_and_fail_closed(tmp_path,algorithm,early_stop):
    # Small synthetic functional check only: 64 rows, 13 features, eight trees.
    rng=np.random.default_rng(5)
    X=rng.random((64,13)).astype(np.float32);X[:,:8]=X[:,:8]>.5
    Y=np.column_stack([(X[:,0]+X[:,8])>1,X[:,1]>0]).astype(int)
    source=tmp_path/"models";source.mkdir()
    entries=[]
    for j,label in enumerate(["a","b"]):
        params={"max_depth":2} if algorithm=="xgb" else {"num_leaves":4,"min_child_samples":2}
        model,_=fit_binary(algorithm,params,X,Y[:,j],rounds=8,threads=2,stop=(X,1-Y[:,j]) if early_stop else None)
        assert contributions(model,algorithm,X[:4]).shape == (4,13)
        path=source/f"{label}.pkl";joblib.dump(model,path)
        entries.append({"label":label,"filename":path.name,"sha256":digest(path),"threshold":.5})
    spec=FeatureSpec(n_bits=8)
    meta={"algorithm":algorithm,"dataset_id":"synthetic","labels":["a","b"],"models":entries,
          "feature_spec":spec.to_dict(),"feature_schema_id":spec.schema_id}
    write_json(source/"model_manifest.json",meta)
    bundle=export_bundle(source,tmp_path/"bundle",X)
    assert len(bundle["models"])==2
    assert all(p["threshold_disagreements"]==0 for p in bundle["parity"])
    meta["models"]=list(reversed(entries));write_json(source/"model_manifest.json",meta)
    with pytest.raises(ValueError,match="order"):export_bundle(source,tmp_path/"bad",X)
    assert not (tmp_path/"bad"/"manifest.json").exists()
