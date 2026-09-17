from pathlib import Path
import copy
import time
import numpy as np
import optuna
import pytest
from alignment import experiments_v6 as v6
from alignment import tuning_v6 as tuning
from alignment.perfume_data import OUTPUT

def protocol(): return v6.core.read(v6.PROTOCOL)

def test_real_inputs_and_reference_recipes_are_valid():
    meta,p=v6.preflight(OUTPUT)
    assert len(meta['labels'])==109
    assert p['tuning']['seconds_total']==32*3600
    assert len(v6.CANDIDATES)==8 and len(v6.TUNED)==4

def test_feature_shapes_differ_only_by_descriptors():
    m=np.zeros((3,1024)); d=np.ones((3,5))
    for key in v6.CANDIDATES:
        assert v6.features(key,m,d).shape==(3,1029 if key[-1] in 'BD' else 1024)

def test_expanded_search_and_depth_constraints():
    p=protocol()
    refs=v6.core.read(v6.REFERENCE/'comparison.json')['initial_recipes']
    for key in v6.TUNED:
        params=dict(refs[key]['recipe'],n_estimators=1600,learning_rate=.005,colsample_bytree=.3)
        result=tuning.suggest(optuna.trial.FixedTrial(params),key.split('_')[0],p)
        assert result==params
    assert p['search_space']['common']['reg_lambda']=={'kind':'float','low':.001,'high':100,'log':True}

def test_fresh_study_and_resume_do_not_import_scores_or_reset_budget(tmp_path):
    p=protocol(); recipe=v6.core.read(v6.REFERENCE/'comparison.json')['initial_recipes']['xgb_D']['recipe']
    study=tuning.initialize(tmp_path,'xgb_D',p,recipe)
    assert len(study.trials)==1 and study.trials[0].state==optuna.trial.TrialState.WAITING
    path=tmp_path/'xgb_D/budget.json'; state=v6.core.read(path)
    state['charged_seconds']=321.5; v6.core.write(path,state)
    study=tuning.initialize(tmp_path,'xgb_D',p,recipe)
    assert len(study.trials)==1
    assert v6.core.read(path)['charged_seconds']==321.5
    p['tuning']['seconds_per_study']+=1
    with pytest.raises(ValueError,match='budget'): tuning.initialize(tmp_path,'xgb_D',p,recipe)

def test_selection_uses_validation_and_prefers_baseline_on_exact_tie():
    robust={k:{'mean_metrics':{'auprc':.3}} for k in v6.CANDIDATES}
    _,chosen,_=v6.select_candidates(robust)
    assert chosen=={'xgb':'xgb_A','lgbm':'lgbm_A'}
    robust['xgb_C']['mean_metrics']['auprc']=.4
    overall,chosen,_=v6.select_candidates(robust)
    assert overall=='xgb_C'

@pytest.mark.parametrize('algorithm',['xgb','lgbm'])
def test_tuning_real_estimator_trial_timeout_and_resume(tmp_path,monkeypatch,algorithm):
    p=protocol(); p['tuning']['seconds_per_study']=100
    recipe=v6.core.read(v6.REFERENCE/'comparison.json')['initial_recipes'][algorithm+'_D']['recipe']
    original=v6.core.cross_validate
    clock=[0.]
    monkeypatch.setattr(tuning.time,'perf_counter',lambda:clock[0])
    monkeypatch.setattr(v6.core.Limits,'check',lambda *args,**kw:None)
    monkeypatch.setattr(tuning,'suggest',lambda *args:{'n_estimators':3,'max_depth':2})
    calls=[]
    def cv(*args,**kwargs):
        calls.append(1)
        if len(calls)>1:
            clock[0]=100.1
            raise TimeoutError('budget')
        result=original(*args,**kwargs)
        clock[0]=1.
        return result
    monkeypatch.setattr(v6.core,'cross_validate',cv)
    y=np.array([[0],[1]]*8,dtype=np.int8)
    x=np.column_stack([y[:,0],np.arange(16)%3]).astype(np.float32)
    split={'folds':{'sweet':[{'train':list(range(8)), 'validation':list(range(8,16))},{'train':list(range(8,16)),'validation':list(range(8))}]}}
    meta={'labels':['sweet'],'summary_labels':['sweet']}
    args=(tmp_path,algorithm+'_D',algorithm,x,y,meta,split,{i:i for i in range(16)},p,recipe)
    result=tuning.tune(*args)
    assert result['n_estimators']==3
    summary=v6.core.read(tmp_path/(algorithm+'_D')/'tuning_summary.json')
    assert summary['complete_trials']==1 and summary['failed_trials']==1
    assert summary['budget']['charged_seconds']==100
    assert tuning.tune(*args)==result
    assert len(calls)==2

def test_pipeline_freeze_resume_completion_and_tamper_detection(tmp_path,monkeypatch):
    p=protocol(); p['robustness']['seeds']=[7,42]; p['robustness']['folds']=2
    dataset=tmp_path/'dataset'; dataset.mkdir()
    run=tmp_path/'run'
    labels=['sweet','woody']
    truth=np.array([[0,0],[1,0],[0,1],[1,1]]*8,dtype=np.int8)
    m=truth.astype(np.float32); d=np.ones((len(truth),1),dtype=np.float32)
    v6.core.save_npz(dataset/'train.npz',truth=truth,morgan=m,descriptors=d,row_index=np.arange(len(truth)))
    v6.core.save_npz(dataset/'test.npz',truth=truth[:8],morgan=m[:8],descriptors=d[:8])
    v6.core.write(dataset/'dataset_manifest.json',{'synthetic':True})
    v6.core.write(dataset/'splits.json',{})
    import pandas as pd
    pd.DataFrame({'group_id':np.arange(len(truth))}).to_csv(dataset/'groups.csv',index=False)
    meta={'labels':labels,'summary_labels':labels,'feature_spec':{'n_bits':2,'descriptors':['dummy']}}
    monkeypatch.setattr(v6,'preflight',lambda path:(meta,p))
    monkeypatch.setattr(v6.core,'check_host',lambda:None)
    monkeypatch.setattr(v6.core.Limits,'check',lambda *args,**kw:None)
    # Small real estimators exercise feature dimensions, persistence, and predictions.
    original_make=v6.core.make_model
    monkeypatch.setattr(v6.core,'make_model',lambda alg,params:original_make(alg,{'n_estimators':3,'max_depth':2}))
    tuned=[]
    def fake_tune(run,key,*args):
        tuned.append(key)
        v6.core.write(run/key/'tuning_summary.json',{'budget':{'status':'complete'}})
        return {'n_estimators':3,'max_depth':2}
    monkeypatch.setattr(v6,'tune',fake_tune)
    original_evaluate=v6.evaluate_test
    def interrupt(*args,**kwargs): raise KeyboardInterrupt()
    monkeypatch.setattr(v6,'evaluate_test',interrupt)
    with pytest.raises(KeyboardInterrupt): v6.run_all(dataset,run)
    assert (run/'selection_frozen.json').exists() and not (run/'complete.json').exists()
    frozen_hash=v6.digest(run/'selection_frozen.json')
    assert len(tuned)==4
    monkeypatch.setattr(v6,'evaluate_test',original_evaluate)
    result=v6.run_all(dataset,run)
    assert v6.digest(run/'selection_frozen.json')==frozen_hash
    assert len(tuned)==4
    assert len(v6.core.read(result)['comparison'])==8
    assert len(pd.read_csv(run/'ablation_deltas.csv'))==10
    v6.run_all(dataset,run)
    assert len(tuned)==4
    (run/'xgb_A/recipe.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='changed'): v6.run_all(dataset,run)
