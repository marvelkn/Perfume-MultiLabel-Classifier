"""Read-only result audit; writes evidence only alongside this script. No fitting."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

ROOT = Path(r'C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier')
PACKAGE = ROOT / '.local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v9-bits-d8/perfume-campus-grouped-v9-bits-d8'
RUN = PACKAGE / 'runs/perfume-five-grouped-v9-bits-d8'
DATA = PACKAGE / 'data/builds/perfume-five-grouped-v2'
OUT = Path(__file__).resolve().parent
SEEDS = [7,21,42,84,100]
CANDIDATES = [f'{a}_m{b}_{s}' for a in ('xgb','lgbm') for b in (1024,2048) for s in ('base','tuned')]
VARIANTS = CANDIDATES + [k+'_8h' for k in CANDIDATES if k.endswith('tuned')]
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
cache = {}

def sha(path):
    path = path.resolve()
    if path not in cache:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024), b''):
                h.update(b)
        cache[path] = h.hexdigest()
    return cache[path]

def verify(folder, files):
    for name, expected in files.items():
        path = (folder / name).resolve()
        assert path.is_relative_to(folder.resolve()), name
        assert path.is_file() and sha(path) == expected, f'Checksum mismatch: {path}'
    return len(files)

def close(a, b, what):
    assert np.isclose(a,b,atol=1e-10,rtol=1e-8), (what,a,b)

def div(a,b):
    return np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)

manifest = read(RUN/'completion_manifest.json')
checked = verify(RUN,manifest['files'])
print(f'Completion checksums verified: {checked}',flush=True)
identity = read(RUN/'identity.json')
for name,path in {'protocol':PACKAGE/'alignment/protocol_v9.json',
                  'dataset':DATA/'dataset_manifest.json',
                  'features':PACKAGE/'data/builds/perfume-m2048-d8-v7/manifest.json',
                  'compute_profile':PACKAGE/'alignment/campus_compute_profile.json'}.items():
    assert sha(path)==identity[name],name
source_count = verify(PACKAGE,identity['source'])
package_count = verify(PACKAGE,read(PACKAGE/'CAMPUS_PACKAGE_MANIFEST.json')['files'])
verify(DATA,read(DATA/'dataset_manifest.json')['files'])
freeze = read(RUN/'selection_frozen.json')
freeze_count = verify(RUN,freeze['files'])
assert freeze['test_used_for_selection'] is False
with np.load(DATA/'test.npz',allow_pickle=False) as z:
    truth = z['truth']; test_indices=z['row_index']
    labels = z['labels'].tolist() if 'labels' in z.files else read(DATA/'dataset_manifest.json')['labels']
with np.load(DATA/'train.npz',allow_pickle=False) as z:
    train_y=z['truth']; train_indices=z['row_index']
assert truth.shape==(1333,109) and train_y.shape==(5353,109)
assert len(set(train_indices)&set(test_indices))==0
allgroups=pd.read_csv(DATA/'groups.csv').group_id.to_numpy()
groups=allgroups[train_indices]
assert not set(groups)&set(allgroups[test_indices])
comparison=pd.read_csv(RUN/'comparison_all.csv').set_index('candidate')
recomputed=[]; cv_scores={}; fold_counts=set(); model_count=0; cv_storage_deltas=[]
fold_cache={}
for seed in SEEDS:
    assert len({sha(RUN/f'robustness/{k}/seed_{seed}/folds.json') for k in VARIANTS})==1
    folds=read(RUN/f'robustness/lgbm_m2048_tuned/seed_{seed}/folds.json')['folds']
    fold_cache[seed]=folds
    for label,parts in folds.items():
        fold_counts.add(len(parts)); heldout=[]
        for part in parts:
            a=np.asarray(part['train']); b=np.asarray(part['validation'])
            assert not set(groups[a])&set(groups[b])
            assert not set(a)&set(b) and len(a)+len(b)==len(train_y)
            heldout.extend(b)
        assert sorted(heldout)==list(range(len(train_y))), (seed,label)
print('Identical folds and group separation verified for all 12 variants.',flush=True)
selected_labels=[]
for key in CANDIDATES:
    models=read(RUN/key/'models.json')
    assert set(models)==set(labels)
    model_count+=len(models)
    verify(RUN/key,{v['file']:v['sha256'] for v in models.values()})
    primary_prob=None
    for secondary in (False,True):
        suffix='_validation_thresholds' if secondary else ''
        with np.load(RUN/key/f'test_predictions{suffix}.npz',allow_pickle=False) as z:
            prob=z['probabilities']; decisions=z['decisions']
            assert np.array_equal(z['truth'],truth) and z['labels'].tolist()==labels
            assert prob.shape==truth.shape and np.isfinite(prob).all()
            assert ((prob>=0)&(prob<=1)).all()
            thresholds=z['thresholds'] if secondary else np.full(109,.5)
            assert np.array_equal(decisions,(prob>=thresholds).astype(int))
            if secondary:
                assert np.array_equal(prob,primary_prob)
                info=read(RUN/'robustness'/key/'thresholds.json')
                assert np.array_equal(thresholds,[info['labels'][n]['threshold'] for n in labels])
            else:
                primary_prob=prob.copy()
        tp=((truth==1)&(decisions==1)).sum(axis=0)
        tn=((truth==0)&(decisions==0)).sum(axis=0)
        fp=((truth==0)&(decisions==1)).sum(axis=0)
        fn=((truth==1)&(decisions==0)).sum(axis=0)
        ap=np.array([average_precision_score(truth[:,j],prob[:,j]) for j in range(109)])
        result={'accuracy':float(((tp+tn)/len(truth)).mean()),
                'auroc':float(np.mean([roc_auc_score(truth[:,j],prob[:,j]) for j in range(109)])),
                'auprc':float(ap.mean()),'specificity':float(div(tn,tn+fp).mean()),
                'precision':float(div(tp,tp+fp).mean()),'recall':float(div(tp,tp+fn).mean()),
                'macro_f1':float(div(2*tp,2*tp+fp+fn).mean()),
                'micro_f1':float(div(np.asarray(2*tp.sum()),np.asarray(2*tp.sum()+fp.sum()+fn.sum()))),
                'mean_predicted_labels':float(decisions.sum(axis=1).mean()),
                'exact_match_ratio':float(np.all(decisions==truth,axis=1).mean())}
        saved=read(RUN/key/f'test_summary{suffix}.json')
        assert saved['selection_sha256']==sha(RUN/'selection_frozen.json')
        for metric,value in saved['summary']['metrics'].items():
            close(result[metric],value,f'{key}{suffix}:{metric}')
            if not secondary:
                close(result[metric],comparison.loc[key,metric],f'CSV:{key}:{metric}')
        close(result['macro_f1'],saved['diagnostics']['macro_f1'],f'{key}{suffix}:F1')
        close(result['macro_f1'],comparison.loc[key,'macro_f1_oof_threshold' if secondary else 'macro_f1_0_5'],'CSV F1')
        recomputed.append({'candidate':key,'threshold':'OOF' if secondary else '0.5',**result})
        if key=='lgbm_m2048_tuned':
            selected_labels.extend([{'label':label,'threshold':'OOF' if secondary else '0.5',
                'positives':int(truth[:,j].sum()),'ap':float(ap[j]),
                'precision':float(div(tp,tp+fp)[j]),'recall':float(div(tp,tp+fn)[j]),
                'f1':float(div(2*tp,2*tp+fp+fn)[j]),'tp':int(tp[j]),'fp':int(fp[j]),
                'fn':int(fn[j]),'tn':int(tn[j])} for j,label in enumerate(labels)])
    print(f'Test metrics independently verified: {key}',flush=True)

for key in VARIANTS:
    values=[]; seed_probs=[]
    for seed in SEEDS:
        folder=RUN/f'robustness/{key}/seed_{seed}'
        recorded=read(folder/'cv_fold_metrics.json')
        from_saved=float(np.mean([np.mean([f['metrics']['auprc'] for f in recorded[n]]) for n in labels]))
        with np.load(folder/'cv_oof_predictions.npz',allow_pickle=False) as z:
            assert np.array_equal(z['truth'],train_y) and z['labels'].tolist()==labels
            scores=z['probabilities']
            assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
        raw=float(np.mean([np.mean([average_precision_score(train_y[part['validation'],j],scores[part['validation'],j])
                    for part in fold_cache[seed][label]]) for j,label in enumerate(labels)]))
        # Saved OOF scores are float32; LightGBM computes fold metrics before this cast.
        # Record rank-tie changes caused by quantization rather than demand bitwise metric equality.
        assert abs(raw-from_saved)<1e-4, (f'Raw OOF AP:{key}:{seed}',raw,from_saved)
        cv_storage_deltas.append({'candidate':key,'seed':seed,'raw_ap':raw,'recorded_ap':from_saved,'delta':raw-from_saved})
        values.append(from_saved); seed_probs.append(scores)
    rob=read(RUN/'robustness'/key/'robustness_summary.json')
    close(np.mean(values),rob['mean_metrics']['auprc'],f'CV mean:{key}')
    close(np.std(values,ddof=1),rob['std_metrics']['auprc'],f'CV SD:{key}')
    if key in CANDIDATES:
        close(np.mean(values),comparison.loc[key,'cv_ap_mean'],f'CSV CV mean:{key}')
        close(np.std(values,ddof=1),comparison.loc[key,'cv_ap_sd'],f'CSV CV SD:{key}')
    cv_scores[key]=dict(zip(map(str,SEEDS),values))
    with np.load(RUN/'robustness'/key/'pooled_oof_predictions.npz',allow_pickle=False) as z:
        pooled=z['probabilities']
        assert np.array_equal(z['truth'],train_y)
    assert np.allclose(pooled,np.mean(seed_probs,axis=0),rtol=1e-6,atol=1e-7)
    info=read(RUN/'robustness'/key/'thresholds.json')
    for j,label in enumerate(labels):
        chosen=info['labels'][label]['threshold']
        assert chosen in info['grid']
        target=train_y[:,j].astype(bool)
        pred=pooled[:,j,None]>=np.array(info['grid'])[None,:]
        tp=(pred&target[:,None]).sum(axis=0)
        f1s=div(2*tp,pred.sum(axis=0)+target.sum())
        close(f1_score(target,pooled[:,j]>=chosen,zero_division=0),max(f1s),'OOF threshold maximum')
    print(f'Raw CV AP and OOF thresholds independently verified: {key}',flush=True)

selected=max(CANDIDATES,key=lambda k:np.mean(list(cv_scores[k].values())))
assert selected==freeze['overall_selected_from_repeated_grouped_cv']=='lgbm_m2048_tuned'
bundle=read(RUN/'inference_bundle.json')
assert bundle['selected_candidate']==selected and bundle['labels']==labels
assert bundle['feature_spec']['dimensions']['input']==2056
assert bundle['selection_sha256']==sha(RUN/'selection_frozen.json')
contrasts=read(RUN/'paired_feature_cv.json')['contrasts']
for row in contrasts:
    ds=[cv_scores[row['right']][str(s)]-cv_scores[row['left']][str(s)] for s in SEEDS]
    assert np.allclose(ds,row['delta_right_minus_left'])
    close(np.mean(ds),row['mean_delta'],'feature contrast')
budgets=read(RUN/'budget_comparison.json')['rows']
for row in budgets:
    key=row['candidate']
    ds=[cv_scores[key][str(s)]-cv_scores[key+'_8h'][str(s)] for s in SEEDS]
    assert np.allclose(ds,row['delta_per_seed'])
    close(np.mean(ds),row['delta_cv_ap_16h_minus_8h'],'budget contrast')

events=[json.loads(line) for line in (RUN/'events.jsonl').read_text().splitlines()]
begin=datetime.fromisoformat(events[0]['time'])
end=datetime.fromisoformat(read(RUN/'complete.json')['completed_at'])
frozen_time=datetime.fromisoformat(freeze['timestamp'])
first_final=min(datetime.fromisoformat(e['time']) for e in events if e['stage']=='final_fit_and_test')
assert frozen_time<=first_final
v7run=ROOT/'.local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v7-m2048-d8/perfume-campus-grouped-v7-m2048-d8/runs/perfume-five-grouped-v7-m2048-d8'
old=pd.read_csv(v7run/'comparison_all.csv').set_index('candidate')
v7comp=[]
for a in ('xgb','lgbm'):
    for old_name,new_name in ((a+'_B',a+'_m2048_base'),(a+'_D',a+'_m2048_tuned')):
        row={'v7':old_name,'v9':new_name}
        for col in ('cv_ap_mean','auprc','auroc','macro_f1_0_5'):
            row[col+'_v7']=float(old.loc[old_name,col]); row[col+'_v9']=float(comparison.loc[new_name,col])
            row[col+'_delta']=row[col+'_v9']-row[col+'_v7']
        row['f1_oof_v7']=float(old.loc[old_name,'macro_f1_validation_threshold'])
        row['f1_oof_v9']=float(comparison.loc[new_name,'macro_f1_oof_threshold'])
        with np.load(v7run/old_name/'test_predictions.npz') as z, np.load(RUN/new_name/'test_predictions.npz') as n:
            assert np.array_equal(z['truth'],n['truth']) and np.array_equal(z['labels'],n['labels'])
            row['identical_predictions']=bool(np.array_equal(z['probabilities'],n['probabilities']))
            if new_name.endswith('base'):
                assert row['identical_predictions']
        v7comp.append(row)

gap=[cv_scores['lgbm_m2048_tuned'][str(s)]-cv_scores['xgb_m1024_tuned'][str(s)] for s in SEEDS]
report={'status':'Verified completed artifacts and recomputed metrics; interpretation has caveats',
        'source_run':str(RUN),'source_manifest_sha256':sha(RUN/'completion_manifest.json'),
        'completion_files_verified':checked,'source_files_verified':source_count,'package_files_verified':package_count,
        'freeze_files_verified':freeze_count,'model_count':model_count,
        'dataset':{'rows':len(train_y)+len(truth),'train_rows':len(train_y),'test_rows':len(truth),'labels':len(labels),
          'group_overlap_train_test':0,'group_overlap_cv':0,'fold_counts':sorted(fold_counts),
          'mean_test_positive_labels':float(truth.sum(axis=1).mean()),'all_negative_accuracy':float((truth==0).mean())},
        'selected':selected,'cv_scores_by_seed':cv_scores,'cv_contrasts':contrasts,'budget_comparison':budgets,
        'selected_vs_runner_up':{'delta_per_seed':gap,'mean_delta':float(np.mean(gap)),'positive_seeds':sum(v>0 for v in gap)},
        'test_metrics':recomputed,'v7_comparison':v7comp,'start':begin.isoformat(),'completed':end.isoformat(),
        'elapsed_hours':(end-begin).total_seconds()/3600,'selected_before_final_fit':True,
        'raw_oof_ap_recomputed_for':VARIANTS,'oof_thresholds_verified_for':VARIANTS,
        'cv_float32_storage_comparison':cv_storage_deltas,
        'max_abs_cv_storage_delta':max(abs(v['delta']) for v in cv_storage_deltas),
        'caveats':['Historical test repeatedly inspected; exploratory follow-up, not independent confirmation.',
          'Repeated CV reuses development data for tuning; not nested CV. Seeds are correlated.',
          'No statistical-significance or globally-optimal-hyperparameters claim.',
          'AP tuning does not guarantee improvement in F1 or precision at a chosen threshold.',
          'Missing aroma annotations are encoded zero, not confirmed sensory negatives.',
          'Group separation uses existing exact structure/feature groups, not a scaffold split.',
          'No training, source pipeline edits, release promotion or thesis edits during analysis.']}
pd.DataFrame(recomputed).to_csv(OUT/'recomputed_metrics.csv',index=False)
pd.DataFrame(selected_labels).to_csv(OUT/'selected_model_per_label.csv',index=False)
pd.DataFrame(v7comp).to_csv(OUT/'v7_comparison.csv',index=False)
(OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ('status','completion_files_verified','source_files_verified','package_files_verified',
    'freeze_files_verified','model_count','dataset','selected','selected_vs_runner_up','elapsed_hours','v7_comparison')},indent=2))
