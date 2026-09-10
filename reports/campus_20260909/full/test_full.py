"""Pipeline checks use synthetic records and fake fitting, never the research dataset."""
import copy
import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from contextlib import contextmanager
import json
import time
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from campus_support import charge, admit, best_complete, write, KillJob
import run_all as controller
import campus_worker as worker


def test_budget_includes_reserved_interruption_and_all_states():
    entries=[{"operation":"tune","algorithm":"xgb","reserved_seconds":1200},
             {"operation":"tune","algorithm":"xgb","reserved_seconds":1200,"elapsed_seconds":12},
             {"operation":"baseline","algorithm":"xgb","reserved_seconds":1200,"elapsed_seconds":100},
             {"operation":"tune","algorithm":"lgbm","reserved_seconds":1200,"elapsed_seconds":700}]
    assert charge(entries,"xgb")==1212
    assert charge(entries,"lgbm")==700


@pytest.mark.parametrize("attempts,used,expected",[(29,0,True),(30,0,False),(0,14400,False),(2,14399.5,False),(2,14398,True)])
def test_admission(attempts,used,expected):
    entries=[{"operation":"tune","algorithm":"xgb","reserved_seconds":1200,"elapsed_seconds":used}]
    assert admit(entries,"xgb",attempts)==expected


def test_best_trial_ignores_failed_and_breaks_exact_tie_by_number():
    trials=[SimpleNamespace(state=SimpleNamespace(name=s),value=v,number=n) for s,v,n in
            [("FAIL",None,0),("COMPLETE",.7,3),("COMPLETE",.7,1),("COMPLETE",.6,2)]]
    assert best_complete(trials).number==1
    assert best_complete(trials[:1]) is None


def test_tune_persists_rng_before_fit_and_marks_failure(tmp_path,monkeypatch):
    import optuna
    import numpy as np
    from src import experiments as ex
    @contextmanager
    def context(*args):
        yield tmp_path,{"seed":42,"splits":{}},np.zeros((2,2)),np.zeros((2,2)),{"feature_spec":{}},object()
    monkeypatch.setattr(ex,"guarded_context",context)
    monkeypatch.setattr(ex,"study_signature",lambda *a:"test-signature")
    monkeypatch.setattr(ex,"FeatureSpec",SimpleNamespace(from_dict=lambda s:object()))
    monkeypatch.setattr(ex,"suggest_parameters",lambda trial,algorithm:{"learning_rate":trial.suggest_float("learning_rate",.01,.2)})
    def score(*args):
        assert (tmp_path/'xgb_sampler.pkl').exists()
        raise RuntimeError("Synthetic fit failure")
    monkeypatch.setattr(ex,"score_recipe",score)
    with pytest.raises(RuntimeError,match="Synthetic"):
        worker.tune_one(tmp_path,"xgb")
    study=optuna.load_study(storage=f"sqlite:///{(tmp_path/'studies.sqlite3').as_posix()}",study_name="xgb")
    assert [t.state.name for t in study.trials]==['FAIL']
    monkeypatch.setattr(ex,"score_recipe",lambda *a:([.2,.3,.4],[[2,2]]*3))
    worker.tune_one(tmp_path,"xgb")
    assert [t.state.name for t in study.trials]==['FAIL','COMPLETE']
    assert study.best_trial.value==pytest.approx(.3)


def test_running_trial_recovery(tmp_path):
    import optuna
    study=optuna.create_study(storage=f"sqlite:///{(tmp_path/'studies.sqlite3').as_posix()}",study_name="lgbm")
    study.ask()
    trials=controller.reconcile_trials(tmp_path,"lgbm")
    assert trials[0].state.name=='FAIL'


def test_job_close_kills_sleeping_child():
    process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        job=KillJob(process)
        job.close()
        process.wait(timeout=5)
        assert process.poll() is not None
    finally:
        if process.poll() is None: process.kill(); process.wait(timeout=5)


def test_live_resource_lock_is_not_removed(tmp_path):
    import os
    path=tmp_path/'.resource-session.lock'
    write(path,{'pid':os.getpid()})
    with pytest.raises(RuntimeError,match='still exists'): controller.clean_stale_lock(tmp_path)
    assert path.exists()


def test_pipeline_order_without_fitting(tmp_path,monkeypatch):
    p=controller.Pipeline.__new__(controller.Pipeline)
    p.directory=tmp_path; p.run=tmp_path/'experiment'; p.run.mkdir(); p.path=tmp_path/'pipeline.json'
    p.state={'stages':{},'entries':[],'tuning':{},'identity':{}}
    events=[]
    def stage(operation,algorithm='xgb',variant=None):
        events.append((operation,algorithm,variant))
        p.state['stages']['/'.join(filter(None,[operation,algorithm,variant]))]={'status':'OK'}
        if operation=='fit': write(p.run/algorithm/'model_manifest.json',{'synthetic':True})
        if operation=='evaluate':
            assert len([e for e in events if e[0]=='fit'])==2
            assert len([e for e in events if e[0]=='sensitivity'])==8
            assert (p.run/'pretest_sensitivity_status.json').exists()
        return True
    def tuning(algorithm):
        events.append(('tune',algorithm,None))
        p.state['tuning'][algorithm]={'best_trial':1,'resource_truncated':False}
    p.stage=stage; p.tuning=tuning
    p.execute()
    assert p.state['status']=='COMPLETE'
    assert [e[0] for e in events[:5]]==['init','baseline','baseline','tune','tune']
    assert not any('smoke' in e for e in events)


def test_no_progress_timeout_is_not_retried(tmp_path):
    p=controller.Pipeline.__new__(controller.Pipeline)
    p.run=tmp_path; p.path=tmp_path/'pipeline.json'; p.state={'stages':{}}
    calls=[]
    def invoke(*a,**k): calls.append(1); return {'status':'RESOURCE','reason':'TIME_LIMIT'}
    p.invoke=invoke
    assert p.stage('baseline','xgb') is False
    assert len(calls)==1


def test_sampler_target_does_not_replace_failures(tmp_path,monkeypatch):
    p=controller.Pipeline.__new__(controller.Pipeline)
    p.run=tmp_path; p.path=tmp_path/'pipeline.json'; p.state={'tuning':{},'entries':[]}
    trials=[SimpleNamespace(state=SimpleNamespace(name='FAIL'),value=None,number=i) for i in range(30)]
    monkeypatch.setattr(controller,'reconcile_trials',lambda *a:trials)
    p.invoke=lambda *a,**k:pytest.fail('Must not create a replacement trial')
    p.tuning('xgb')
    assert p.state['tuning']['xgb']['attempts']==30
    assert p.state['tuning']['xgb']['best_trial'] is None


def test_sensitivity_fixed_rounds_and_training_only(tmp_path,monkeypatch):
    import numpy as np
    from src import experiments as ex
    variant={'seed':123,'grouped':False,'folds':[{'train':[2,3,4,5],'score':[0,1]},
             {'train':[0,1,4,5],'score':[2,3]},{'train':[0,1,2,3],'score':[4,5]}]}
    write(tmp_path/'reports/step3_20260908/sensitivity_execution_partitions.json',{'random_seed_123':variant})
    run=tmp_path/'run'; run.mkdir()
    write(run/'xgb/model_manifest.json',{'models':[{'rounds':2},{'rounds':3}], 'imbalance':'none','feature_spec':{'n_bits':2},'model_params':{}})
    X=np.arange(16).reshape(8,2); Y=np.array([[0,1],[1,0]]*4)
    @contextmanager
    def context(*a): yield run,{'splits':{'fit':list(range(6))}},X,Y,{},SimpleNamespace(check=lambda **kw:None)
    monkeypatch.setattr(ex,'guarded_context',context)
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    seen=[]
    def resample(x,y,strategy,seed,n_bits):
        assert not np.any(x[:,0]>=12)
        seen.append(('resample',seed))
        return x,y
    def fit(algorithm,params,x,y,**kwargs):
        assert kwargs['seed']==123 and kwargs['rounds'] in (2,3) and 'stop' not in kwargs
        seen.append(('fit',kwargs['rounds']))
        return SimpleNamespace(predict_proba=lambda x:np.tile([.4,.6],(len(x),1))),kwargs['rounds']
    monkeypatch.setattr(ex,'resample',resample); monkeypatch.setattr(ex,'fit_binary',fit)
    result=worker.sensitivity(run,'xgb','random_seed_123')
    assert len(result['fold_macro_ap'])==3
    assert [v for k,v in seen if k=='resample']==[123,124,125]
    seen.clear(); worker.sensitivity(run,'xgb','random_seed_123')
    assert seen==[]


def test_invoke_waits_for_job_release_and_records_result(tmp_path,monkeypatch):
    fake=tmp_path/'tooling'; fake.mkdir()
    (fake/'campus_worker.py').write_text("import json,sys,time\nfrom pathlib import Path\nr=json.loads(Path(sys.argv[1]).read_text())\nwhile not Path(r['gate']).exists(): time.sleep(.01)\nPath(r['result']).write_text(json.dumps({'status':'OK','synthetic':True}))\n")
    monkeypatch.setattr(controller,'HERE',fake)
    monkeypatch.setattr(controller.psutil,'virtual_memory',lambda:SimpleNamespace(available=16*1024**3))
    p=controller.Pipeline.__new__(controller.Pipeline)
    p.directory=tmp_path/'session'; p.directory.mkdir(); p.run=p.directory/'experiment'
    p.path=p.directory/'pipeline.json'; p.state={'entries':[]}
    result=p.invoke('init',seconds=10)
    assert result['status']=='OK' and result['synthetic']
    assert p.state['entries'][0]['elapsed_seconds']>0
    for path in p.directory.glob('sessions/*/resources.jsonl'):
        assert all(json.loads(line)['cpu_c'] is None for line in path.read_text().splitlines())


def test_invoke_kills_deadline_worker(tmp_path,monkeypatch):
    fake=tmp_path/'tooling'; fake.mkdir()
    (fake/'campus_worker.py').write_text('import time; time.sleep(60)')
    monkeypatch.setattr(controller,'HERE',fake)
    monkeypatch.setattr(controller.psutil,'virtual_memory',lambda:SimpleNamespace(available=16*1024**3))
    p=controller.Pipeline.__new__(controller.Pipeline)
    p.directory=tmp_path/'session'; p.directory.mkdir(); p.run=p.directory/'experiment'
    p.path=p.directory/'pipeline.json'; p.state={'entries':[]}
    result=p.invoke('init',seconds=.3)
    assert result['status']=='RESOURCE' and result['reason']=='TIME_LIMIT'
    assert p.state['entries'][0]['elapsed_seconds']<5


def test_full_policy_isolated_process_keeps_ram_and_threads(tmp_path):
    code = """
import sys
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
from campus_support import install_policy, PROFILE
cfg=install_policy(30)
from src import runtime
assert cfg['resources']['threads']==2
assert cfg['resources']['profile']==PROFILE
assert cfg['resources']['require_temperature'] is False
runtime.validate_training_resources(cfg['resources'])
g=runtime.ResourceGuard(cfg['resources'])
assert g.temperature() is None
assert g.settings['session_minutes']<=25/60
runtime.psutil.virtual_memory=lambda:SimpleNamespace(available=1024**3)
runtime.psutil.Process=lambda:SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=100),children=lambda **kw:[])
try:
    g.check(force=True)
except runtime.ResourceLimit as error:
    assert 'RAM' in str(error)
else:
    raise AssertionError('RAM guard bypassed')
"""
    result=subprocess.run([sys.executable,'-c',code,str(HERE)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_completed_baseline_recovers_without_repeating_fit(tmp_path):
    write(tmp_path/'xgb_baselines.json',{name:{'fold_ap':[.1,.2,.3],'mean_ap':.2} for name in ['none','class_weight','random_oversample']})
    p=controller.Pipeline.__new__(controller.Pipeline)
    p.run=tmp_path; p.path=tmp_path/'pipeline.json'; p.state={'stages':{}}
    p.invoke=lambda *a,**k:pytest.fail('Already completed fitting must not be repeated')
    assert p.stage('baseline','xgb')
    assert p.state['stages']['baseline/xgb']['status']=='OK'
