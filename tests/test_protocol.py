import numpy as np
import pandas as pd
import pytest
import optuna
from src.build_dataset import _attach_smiles, _attach_sigma
from src.splits import development_partitions
from src.resampling import resample
from src.experiments import suggest_parameters, require_selection_open
from src.metrics import thresholds_for, evaluate

def test_joins_follow_identifiers_after_shuffling():
    beh=pd.DataFrame({"Stimulus":["b","a",None],"Labels":["fruity","sweet","bad"]})
    sti=pd.DataFrame({"Stimulus":["a","b",None],"CID":["10","20","30"]})
    mol=pd.DataFrame({"CID":["20","30","10"],"SMILES":["CCO","CCC","CC"]})
    result=_attach_smiles(beh,sti.sample(frac=1),mol.sample(frac=1))
    assert result["__smiles__"].iloc[:2].tolist() == ["CCO","CC"]
    assert pd.isna(result["__smiles__"].iloc[2])
    with pytest.raises(ValueError,match="Conflicting"):
        _attach_smiles(beh,pd.concat([sti,pd.DataFrame({"Stimulus":["a"],"CID":["20"]})]),mol)

def test_sigma_all_cas_aliases_must_be_unambiguous():
    sti=pd.DataFrame({"Stimulus":["1","1","2","2","3","3"],"CAS":["a","b","a","c","a","missing"]})
    beh=pd.DataFrame({"Stimulus":["1","2","3"]})
    result,rejected=_attach_sigma(beh,sti,{"a":"CC","b":"CC","c":"CCO"})
    assert result["__smiles__"].iloc[0] == "CC"
    assert result["__smiles__"].iloc[1:].isna().all()
    assert len(rejected) == 2

def test_development_roles_are_disjoint_and_cover_fit():
    Y=np.random.default_rng(42).binomial(1,.4,(180,3))
    splits=development_partitions(Y)
    fit,threshold=set(splits["fit"]),set(splits["threshold"])
    assert not fit & threshold and len(fit|threshold)==len(Y)
    scores=[]
    for fold in splits["folds"]:
        train,stop,score=(set(fold[k]) for k in ("train","stop","score"))
        assert not train&stop and not train&score and not stop&score
        assert train|stop|score == fit
        scores.extend(score)
    assert set(scores)==fit and len(scores)==len(fit)
    assert splits == development_partitions(Y)

@pytest.mark.parametrize("strategy",["random_oversample","mlsmote"])
def test_resampling_retains_binary_bits_and_original_rows(strategy):
    rng=np.random.default_rng(1)
    X=np.column_stack([rng.integers(0,2,(30,8)),rng.random((30,5))*10]).astype(np.float32)
    Y=np.zeros((30,2),dtype=np.uint8);Y[:20,0]=1;Y[:4,1]=1
    xr,yr=resample(X,Y,strategy,n_bits=8)
    np.testing.assert_array_equal(xr[:30],X);np.testing.assert_array_equal(yr[:30],Y)
    assert len(xr)==45 and set(np.unique(xr[:,:8])) <= {0,1}
    assert np.all(yr[30:,1] == 1)

def test_model_spaces_are_specific_and_lgbm_bagging_is_enabled():
    sx=optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1)).ask()
    sl=optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1)).ask()
    x,l=suggest_parameters(sx,"xgb"),suggest_parameters(sl,"lgbm")
    assert "gamma" in x and "gamma" not in l
    assert "num_leaves" in l and "num_leaves" not in x
    assert l["num_leaves"] <= 2**l["max_depth"] and l["subsample_freq"] > 0

def test_thresholds_and_metrics_have_explicit_semantics():
    Y=np.array([[0,1],[0,0],[1,0],[1,1]])
    P=np.array([[.1,.8],[.2,.1],[.7,.2],[.9,.9]])
    T=thresholds_for(Y,P)
    metrics=evaluate(Y,P,T,["a","b"])
    assert metrics["f1_macro"]==metrics["average_precision_macro"]==1
    assert metrics["hamming_loss"]==0
    with pytest.raises(ValueError): thresholds_for(np.ones_like(Y),P)

def test_test_access_freezes_all_further_selection(tmp_path):
    require_selection_open(tmp_path)
    (tmp_path/"selection_frozen.json").write_text("{}")
    with pytest.raises(ValueError,match="frozen"): require_selection_open(tmp_path)
