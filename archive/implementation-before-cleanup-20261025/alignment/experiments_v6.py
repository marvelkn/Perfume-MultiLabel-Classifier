"""Expanded C/D search and a complete eight-candidate exploratory comparison."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil
import warnings
import joblib
import numpy as np
import pandas as pd
import psutil
from . import experiments as core
from .data import ROOT, digest
from .experiments_v4 import paired_label_bootstrap, paired_seed_comparison
from .experiments_v5 import fit_final_models
from .metrics import binary_metrics, summarize
from .perfume_data import OUTPUT
from .tuning_v6 import tune, event, suggest

warnings.filterwarnings('ignore',message='X does not have valid feature names, but LGBMClassifier was fitted with feature names',category=UserWarning)
RUN_ID='perfume-five-grouped-v6'
PROTOCOL=ROOT/'alignment/protocol_v6.json'
REFERENCE=ROOT/'v6_reference'
ALGORITHMS=('xgb','lgbm')
CANDIDATES=tuple(f'{a}_{c}' for a in ALGORITHMS for c in 'ABCD')
TUNED=tuple(k for k in CANDIDATES if k[-1] in 'CD')

def features(key, morgan, descriptors):
    return morgan if key[-1] in 'AC' else np.column_stack([morgan,descriptors])

def verify_files(root, files):
    for name,expected in files.items():
        if digest(Path(root)/name)!=expected: raise ValueError(f'Artifact changed: {name}')

def preflight(dataset):
    meta,_=core.preflight(dataset)
    p=core.read(PROTOCOL)
    if p['id']!=RUN_ID or tuple(p['algorithms'])!=ALGORITHMS or p['conditions']!=list('ABCD'):
        raise ValueError('Wrong v6 protocol.')
    if p['environment']!=core.environment() or p['dataset_manifest_sha256']!=digest(Path(dataset)/'dataset_manifest.json'):
        raise ValueError('V6 dataset/environment mismatch.')
    if p['group_policy']!=core.POLICY or p['class_imbalance']['policy']!=core.WEIGHTING_POLICY:
        raise ValueError('V6 grouped split or weighting mismatch.')
    if p['tuning']['seconds_total']!=4*p['tuning']['seconds_per_study']:
        raise ValueError('Four equal budgets required.')
    if not 5<=p['tuning']['heartbeat_seconds']<=60: raise ValueError('Invalid heartbeat.')
    verify_files(REFERENCE,core.read(REFERENCE/'manifest.json')['files'])
    references=core.read(REFERENCE/'comparison.json')
    if set(references['initial_recipes'])!=set(TUNED): raise ValueError('Missing reference recipe.')
    for source in references['source_runs'].values():
        if source['dataset_manifest_sha256']!=p['dataset_manifest_sha256']: raise ValueError('Historical cohort differs.')
    # Validate queued recipes against the expanded distributions, including leaf/depth constraints.
    import optuna
    for key in TUNED:
        recipe=references['initial_recipes'][key]['recipe']
        with warnings.catch_warnings():
            warnings.simplefilter('error',UserWarning)
            if suggest(optuna.trial.FixedTrial(recipe),key.split('_')[0],p)!=recipe:
                raise ValueError('Historical recipe cannot be re-evaluated in this search space.')
    return meta,p

def select_candidates(robustness):
    scores={k:v['mean_metrics']['auprc'] for k,v in robustness.items()}
    # Exact ties prefer the simpler untuned condition; no test outcome is inspected.
    chosen={a:max((k for k in CANDIDATES if k.startswith(a+'_')),key=lambda k:(scores[k],-'ABCD'.index(k[-1]))) for a in ALGORITHMS}
    overall=max(chosen.values(),key=lambda k:(scores[k],k))
    return overall,chosen,scores

def evaluate_test(run, frozen, key, x, truth, labels):
    folder=run/key
    models=core.read(folder/'models.json')
    verify_files(folder,{v['file']:v['sha256'] for v in models.values()})
    primary=folder/'test_summary.json'
    if primary.exists():
        result=core.read(primary)
        if result['selection_sha256']!=digest(frozen): raise ValueError('Selection changed.')
        verify_files(folder,result['artifact_sha256'])
        with np.load(folder/'test_predictions.npz',allow_pickle=False) as z:
            prob=z['probabilities']
            if not np.array_equal(z['truth'],truth) or z['labels'].tolist()!=labels:
                raise ValueError('Saved test cohort differs.')
    else:
        prob=np.column_stack([joblib.load(folder/models[name]['file']).predict_proba(x)[:,1] for name in labels]).astype(np.float32)
        decisions=(prob>=.5).astype(np.int8)
        per_label={name:binary_metrics(truth[:,j],prob[:,j],decisions[:,j]) for j,name in enumerate(labels)}
        core.save_npz(folder/'test_predictions.npz',probabilities=prob,decisions=decisions,truth=truth,labels=np.asarray(labels))
        core.write(folder/'test_per_label.json',per_label)
        result={'role':'exploratory v6, fixed threshold 0.5','summary':summarize(per_label,labels),'diagnostics':core.prediction_diagnostics(truth,decisions,labels),'selection_sha256':digest(frozen),'artifact_sha256':{n:digest(folder/n) for n in ('test_predictions.npz','test_per_label.json')}}
        core.write(primary,result)
    thresholds_path=run/'robustness'/key/'thresholds.json'
    secondary=folder/'test_summary_validation_thresholds.json'
    if secondary.exists():
        extra=core.read(secondary)
        if extra['selection_sha256']!=digest(frozen) or extra['threshold_source_sha256']!=digest(thresholds_path): raise ValueError('Threshold source changed.')
        verify_files(folder,extra['artifact_sha256'])
    else:
        threshold_info=core.read(thresholds_path)
        thresholds=np.asarray([threshold_info['labels'][n]['threshold'] for n in labels])
        decisions=(prob>=thresholds[None,:]).astype(np.int8)
        per_label={name:binary_metrics(truth[:,j],prob[:,j],decisions[:,j]) for j,name in enumerate(labels)}
        core.save_npz(folder/'test_predictions_validation_thresholds.npz',probabilities=prob,decisions=decisions,truth=truth,labels=np.asarray(labels),thresholds=thresholds)
        core.write(folder/'test_per_label_validation_thresholds.json',per_label)
        extra={'role':'secondary; thresholds chosen using training OOF only','summary':summarize(per_label,labels),'diagnostics':core.prediction_diagnostics(truth,decisions,labels),'selection_sha256':digest(frozen),'threshold_source_sha256':digest(thresholds_path),'artifact_sha256':{n:digest(folder/n) for n in ('test_predictions_validation_thresholds.npz','test_per_label_validation_thresholds.json')}}
        core.write(secondary,extra)
    return result,extra

def summaries(run, meta, protocol, freeze):
    references=core.read(REFERENCE/'comparison.json')['historical']
    rows=[]
    for key in CANDIDATES:
        rob=core.read(run/'robustness'/key/'robustness_summary.json')
        test=core.read(run/key/'test_summary.json')
        extra=core.read(run/key/'test_summary_validation_thresholds.json')
        rows.append({'candidate':key,'cv_ap_mean':rob['mean_metrics']['auprc'],'cv_ap_sd':rob['std_metrics']['auprc'],**test['summary']['metrics'],'macro_f1_0_5':test['diagnostics']['macro_f1'],'macro_f1_validation_threshold':extra['diagnostics']['macro_f1'],'selected_for_algorithm':key in freeze['selected_by_algorithm'].values(),'selected_overall':key==freeze['overall_selected_from_repeated_grouped_cv']})
    pd.DataFrame(rows).to_csv(run/'comparison_all.csv',index=False)
    deltas=[]
    for a in ALGORITHMS:
        for left,right,meaning in [('A','B','descriptors_without_tuning'),('A','C','tuning_on_morgan'),('B','D','tuning_on_combined_features'),('C','D','descriptors_with_equal_tuning_budget'),('A','D','combined_modifications')]:
            l=next(r for r in rows if r['candidate']==f'{a}_{left}')
            r=next(r for r in rows if r['candidate']==f'{a}_{right}')
            deltas.append({'algorithm':a,'contrast':f'{right}-{left}','meaning':meaning,'cv_ap_delta':r['cv_ap_mean']-l['cv_ap_mean'],**{m:r[m]-l[m] for m in core.METRICS}})
    pd.DataFrame(deltas).to_csv(run/'ablation_deltas.csv',index=False)
    history=[]
    for row in rows:
        for key,old in references.items():
            if key.rsplit('_v',1)[0]!=row['candidate']: continue
            value={'candidate':row['candidate'],'reference':key}
            if 'test' in old: value.update({f'test_delta_{m}':row[m]-old['test'][m] for m in core.METRICS})
            if 'repeated_cv' in old: value['cv_ap_delta']=row['cv_ap_mean']-old['repeated_cv']['auprc']
            history.append(value)
    pd.DataFrame(history).to_csv(run/'comparison_history.csv',index=False)
    selected=freeze['selected_by_algorithm']
    seed_comparison=paired_seed_comparison(*(core.read(run/'robustness'/selected[a]/'robustness_summary.json') for a in ALGORITHMS))
    label_comparison=paired_label_bootstrap(*(core.read(run/selected[a]/'test_per_label.json') for a in ALGORITHMS),meta['labels'])
    core.write(run/'paired_seed_comparison.json',seed_comparison)
    core.write(run/'paired_label_bootstrap_test.json',label_comparison)
    overall=freeze['overall_selected_from_repeated_grouped_cv']
    core.write(run/'inference_bundle.json',{'selected_candidate':overall,'algorithm':overall.split('_')[0],'models_directory':overall,'models':core.read(run/overall/'models.json'),'feature_spec':meta['feature_spec'],'use_descriptors':overall[-1] in 'BD','labels':meta['labels'],'primary_threshold':0.5,'secondary_thresholds_path':f'robustness/{overall}/thresholds.json','environment':core.environment(),'selection_sha256':digest(run/'selection_frozen.json'),'score_interpretation':'Uncalibrated model score; not intensity or mixture composition.'})
    lines=['# Hasil grouped-v6','',f'Model terpilih dari validasi: **{overall}**.','',protocol['confirmatory_status'],'','| Kandidat | CV AP | SD | Test AP | AUROC | Precision | Recall |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['candidate']} | {r['cv_ap_mean']:.5f} | {r['cv_ap_sd']:.5f} | {r['auprc']:.5f} | {r['auroc']:.5f} | {r['precision']:.5f} | {r['recall']:.5f} |")
    lines+=['','Perbandingan lengkap: comparison_all.csv, ablation_deltas.csv, comparison_history.csv.','Hasil baru tidak otomatis lebih baik; laporkan kenaikan dan penurunan.','Interval bootstrap label bersifat deskriptif dan tidak menggantikan validasi molekul independen.','Gunakan inference_bundle.json untuk menemukan model terpilih dan skema fitur yang cocok.']
    (run/'RESULTS_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'protocol':RUN_ID,'completed_at':datetime.now(timezone.utc).isoformat(),'selected_by_algorithm':selected,'overall_selected_from_repeated_grouped_cv':overall,'comparison':rows,'tuning_outcome':{k:core.read(run/k/'tuning_summary.json') for k in TUNED},'paired_seed_comparison':seed_comparison,'paired_label_bootstrap_test':label_comparison,'test_used_for_selection':False,'confirmatory_status':protocol['confirmatory_status']}

def run_all(dataset, run):
    core.check_host()
    dataset,run=Path(dataset).resolve(),Path(run).resolve()
    meta,protocol=preflight(dataset)
    run.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(run).free<10*1024**3: raise RuntimeError('Need at least 10 GiB free disk for models and result ZIP.')
    identity={'protocol':digest(PROTOCOL),'dataset':digest(dataset/'dataset_manifest.json'),'reference':digest(REFERENCE/'manifest.json'),'source':{p.name:digest(p) for p in (ROOT/'alignment').glob('*.py')},'environment':core.environment(),'compute_profile':digest(core.COMPUTE_PROFILE),'host':platform.node(),'physical_cores':psutil.cpu_count(logical=False)}
    with core.run_lock(run):
        path=run/'identity.json'
        if path.exists() and core.read(path)!=identity: raise ValueError('Code, environment, host, or input changed; do not mix runs.')
        if not path.exists(): core.write(path,identity)
        if (run/'complete.json').exists() and (run/'completion_manifest.json').exists():
            verify_files(run,core.read(run/'completion_manifest.json')['files'])
            event(run,'already_complete',message='Verified; no training repeated.')
            return run/'complete.json'
        frozen=run/'selection_frozen.json'
        with np.load(dataset/'train.npz',allow_pickle=False) as z:
            y,morgan,desc,indices=z['truth'],z['morgan'],z['descriptors'],z['row_index'].astype(int)
        if not frozen.exists():
            refs=core.read(REFERENCE/'comparison.json')['initial_recipes']
            split=core.read(dataset/'splits.json')
            positions={int(v):i for i,v in enumerate(indices)}
            recipes={}
            for key in CANDIDATES:
                folder=run/key; folder.mkdir(exist_ok=True)
                done=folder/'recipe.json'
                if done.exists():
                    recipes[key]=core.read(done)
                    if key in TUNED and core.read(folder/'tuning_summary.json')['budget']['status']!='complete': raise ValueError('Recipe exists before tuning completes.')
                    continue
                event(run,'prepare_candidate',candidate=key)
                recipes[key]=dict(protocol['baseline_parameters']) if key not in TUNED else tune(run,key,key.split('_')[0],features(key,morgan,desc),y,meta,split,positions,protocol,refs[key]['recipe'])
                core.write(folder/'training_policy.json',{'condition':key[-1],'weighting':core.WEIGHTING_POLICY,'tuning_budget_seconds':protocol['tuning']['seconds_per_study'] if key in TUNED else 0})
                core.write(done,recipes[key])
            groups=pd.read_csv(dataset/'groups.csv').group_id.to_numpy()[indices]
            robustness={}
            for key in CANDIDATES:
                event(run,'repeated_grouped_cv',candidate=key,seeds=protocol['robustness']['seeds'])
                robustness[key]=core.repeated_grouped_cv(run,key,key.split('_')[0],recipes[key],features(key,morgan,desc),y,meta['labels'],groups,protocol)
            overall,selected,scores=select_candidates(robustness)
            paths=[p for k in CANDIDATES for p in (run/k).iterdir() if p.is_file() and not p.name.endswith('.tmp')]
            paths.extend(p for p in (run/'robustness').rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
            core.write(frozen,{'protocol':RUN_ID,'timestamp':datetime.now(timezone.utc).isoformat(),'selected_by_algorithm':selected,'overall_selected_from_repeated_grouped_cv':overall,'validation_scores':scores,'test_used_for_selection':False,'files':{p.relative_to(run).as_posix():digest(p) for p in paths}})
        freeze=core.read(frozen)
        verify_files(run,freeze['files'])
        # No tuning/CV mutations after freeze: safe resume even after test evaluation starts.
        with np.load(dataset/'test.npz',allow_pickle=False) as z:
            yt,mt,dt=z['truth'],z['morgan'],z['descriptors']
        for key in CANDIDATES:
            event(run,'final_fit_and_test',candidate=key)
            fit_final_models(run/key,key.split('_')[0],core.read(run/key/'recipe.json'),features(key,morgan,desc),y,meta['labels'])
            evaluate_test(run,frozen,key,features(key,mt,dt),yt,meta['labels'])
        result=summaries(run,meta,protocol,freeze)
        core.write(run/'complete.json',result)
        excluded={'completion_manifest.json','.alignment.lock','progress.json','events.jsonl'}
        files={p.relative_to(run).as_posix():digest(p) for p in run.rglob('*') if p.is_file() and p.name not in excluded and not p.name.endswith('.tmp')}
        core.write(run/'completion_manifest.json',{'files':files})
        event(run,'complete',selected=result['overall_selected_from_repeated_grouped_cv'])
    return run/'complete.json'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=OUTPUT)
    parser.add_argument('--run',type=Path,default=ROOT/f'runs/{RUN_ID}')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:
        meta,p=preflight(args.dataset)
        print({'status':'V6_INPUTS_VERIFIED','candidates':list(CANDIDATES),'labels':len(meta['labels']),'hours_per_study':p['tuning']['seconds_per_study']/3600,'total_tuning_hours':p['tuning']['seconds_total']/3600,'training_started':False})
    else: print(run_all(args.dataset,args.run))

if __name__=='__main__': main()
