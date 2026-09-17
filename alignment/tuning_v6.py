"""Fresh TPE studies with equal wall-clock budgets and atomic sampler checkpoints."""
from datetime import datetime, timezone
from pathlib import Path
import time
import joblib
import optuna
from . import experiments as core
from .metrics import summarize

def atomic_dump(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    joblib.dump(value, temp)
    temp.replace(path)

def event(run, stage, **details):
    value = {'time': datetime.now(timezone.utc).isoformat(), 'stage':stage, **details}
    core.write(Path(run)/'progress.json', value)
    with (Path(run)/'events.jsonl').open('a',encoding='utf-8') as f:
        import json
        f.write(json.dumps(value)+'\n')
    print(f"[{value['time']}] {stage}: {details}", flush=True)

def suggest(trial, algorithm, protocol):
    space = protocol['search_space']
    params = {}
    for name, spec in {**space['common'], **space[algorithm]}.items():
        kwargs = {k:v for k,v in spec.items() if k!='kind'}
        params[name] = (trial.suggest_int if spec['kind']=='int' else trial.suggest_float)(name, **kwargs)
    if algorithm == 'lgbm':
        leaves = space['lgbm_num_leaves']
        high = min(leaves['high'], 2**params['max_depth']) if leaves['cap_by_depth'] else leaves['high']
        params['num_leaves'] = trial.suggest_int('num_leaves', leaves['low'], high)
        params.update(space['lgbm_fixed'])
    return params

def initialize(run, key, protocol, recipe):
    folder = Path(run)/key
    folder.mkdir(parents=True,exist_ok=True)
    state_path = folder/'budget.json'
    policy = protocol['tuning']
    sampler_path = folder/'sampler.joblib'
    sampler = optuna.samplers.TPESampler(seed=policy['seed'],n_startup_trials=policy['n_startup_trials'],n_ei_candidates=policy['n_ei_candidates'],multivariate=policy['multivariate'])
    study = optuna.create_study(study_name=key,storage=f"sqlite:///{(folder/'optuna.sqlite3').as_posix()}",direction='maximize',sampler=sampler,load_if_exists=True)
    if state_path.exists():
        state = core.read(state_path)
        if state['budget_seconds'] != policy['seconds_per_study']:
            raise ValueError('Cannot change budget while resuming.')
        if not sampler_path.exists(): raise ValueError('Missing sampler checkpoint.')
    else:
        if any(t.state != optuna.trial.TrialState.WAITING for t in study.trials):
            raise ValueError('Study exists without budget evidence; do not import historical trials.')
        queued = {k:v for k,v in recipe.items() if k!='subsample_freq'}
        study.enqueue_trial(queued,skip_if_exists=True)
        atomic_dump(sampler,sampler_path)
        core.write(state_path,{'budget_seconds':policy['seconds_per_study'],'charged_seconds':0.,'status':'initialized','active_trial':None,'source_completed_trials':0})
    study.sampler=joblib.load(sampler_path)
    return study

class BudgetLimits(core.Limits):
    def __init__(self, deadline, callback, interval):
        super().__init__(deadline)
        self.callback, self.interval, self.last = callback, interval, 0.
    def check(self, force=False):
        now=time.perf_counter()
        if now-self.last>=self.interval:
            self.callback(now)
            self.last=now
        super().check(force)

def tune(run, key, algorithm, x, y, meta, split, positions, protocol, initial_recipe):
    folder=Path(run)/key
    study=initialize(run,key,protocol,initial_recipe)
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            study.tell(t.number,state=optuna.trial.TrialState.FAIL)
    state_path=folder/'budget.json'
    state=core.read(state_path)
    policy=protocol['tuning']
    remaining=max(0.,policy['seconds_per_study']-state['charged_seconds'])
    if remaining>1 and len(study.trials)<policy['safety_max_trials']:
        started=time.perf_counter(); charged=state['charged_seconds']; last_log=started
        def heartbeat(now):
            nonlocal last_log
            state.update(charged_seconds=min(policy['seconds_per_study'],charged+now-started),last_heartbeat_at=datetime.now(timezone.utc).isoformat(),status='running')
            core.write(state_path,state)
            if now-last_log>=60:
                event(run,'tuning',candidate=key,trial=state['active_trial'],hours_used=round(state['charged_seconds']/3600,3),hours_budget=policy['seconds_per_study']/3600)
                last_log=now
        limits=BudgetLimits(started+remaining,heartbeat,policy['heartbeat_seconds'])
        try:
            while time.perf_counter()<limits.deadline and len(study.trials)<policy['safety_max_trials']:
                trial=study.ask()
                state['active_trial']=trial.number
                heartbeat(time.perf_counter())
                try:
                    params=suggest(trial,algorithm,protocol)
                    trial.set_user_attr('resolved_parameters',params)
                    atomic_dump(study.sampler,folder/'sampler.joblib')
                    event(run,'trial_started',candidate=key,trial=trial.number)
                    values=core.cross_validate(algorithm,params,x,y,meta['labels'],split['folds'],positions,limits,meta['summary_labels'])
                    score=summarize(values,meta['summary_labels'])['metrics']['auprc']
                    study.tell(trial,score)
                    event(run,'trial_completed',candidate=key,trial=trial.number,macro_ap=score,best_macro_ap=study.best_value)
                except TimeoutError:
                    trial.set_user_attr('stop_reason','budget_exhausted')
                    study.tell(trial,state=optuna.trial.TrialState.FAIL)
                    break
                except BaseException:
                    trial.set_user_attr('stop_reason','interrupted_or_error')
                    study.tell(trial,state=optuna.trial.TrialState.FAIL)
                    raise
                finally:
                    state['active_trial']=None
                    atomic_dump(study.sampler,folder/'sampler.joblib')
                    study.trials_dataframe().to_csv(folder/'trials.csv',index=False)
                    heartbeat(time.perf_counter())
        finally:
            state.update(charged_seconds=min(policy['seconds_per_study'],charged+time.perf_counter()-started),active_trial=None)
            state['status']='complete' if state['charged_seconds']>=policy['seconds_per_study']-1 else 'paused'
            state['stop_reason']='time_budget_exhausted' if state['status']=='complete' else 'interrupted_or_safety_limit'
            core.write(state_path,state)
            atomic_dump(study.sampler,folder/'sampler.joblib')
            study.trials_dataframe().to_csv(folder/'trials.csv',index=False)
    else:
        study.trials_dataframe().to_csv(folder/'trials.csv',index=False)
    done=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    if not done: raise RuntimeError(f'{key}: no completed trial; preserve run for investigation.')
    if state['charged_seconds']<policy['seconds_per_study']-1:
        raise RuntimeError('Safety trial cap reached before full budget; do not treat as complete.')
    best=max(done,key=lambda t:(t.value,-t.number))
    core.write(folder/'tuning_summary.json',{'complete_trials':len(done),'failed_trials':sum(t.state==optuna.trial.TrialState.FAIL for t in study.trials),'best_trial':best.number,'best_macro_ap':best.value,'completed_trials_after_best':sum(t.number>best.number for t in done),'budget':state,'initial_recipe_recomputed':True})
    return best.user_attrs['resolved_parameters']
