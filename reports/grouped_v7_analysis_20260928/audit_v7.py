"""Read-only audit of returned v7 artifacts; write only this analysis directory."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PACKAGE = ROOT / ".local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v7-m2048-d8/perfume-campus-grouped-v7-m2048-d8"
RUN = PACKAGE / "runs/perfume-five-grouped-v7-m2048-d8"
DATA = PACKAGE / "data/builds/perfume-five-grouped-v2"
CANDIDATES = [f"{a}_{c}" for a in ("xgb", "lgbm") for c in "ABCD"]
SEEDS = [7,21,42,84,100]
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
cache = {}
def sha(path):
    path = path.resolve()
    if path not in cache:
        h=hashlib.sha256()
        with path.open("rb") as f:
            for b in iter(lambda:f.read(1024*1024),b""):
                h.update(b)
        cache[path]=h.hexdigest()
    return cache[path]
def verify(folder, files):
    for name, expected in files.items():
        path=(folder/name).resolve()
        assert path.is_relative_to(folder.resolve()), name
        assert path.is_file() and sha(path)==expected, f"Checksum mismatch: {path}"
    return len(files)
def close(a,b,what):
    assert np.isclose(a,b,atol=1e-10,rtol=1e-8), (what,a,b)

completion=read(RUN/"completion_manifest.json")
checked=verify(RUN,completion["files"])
print(f"Completion checksums verified: {checked}",flush=True)
identity=read(RUN/"identity.json")
for key,path in {"protocol":PACKAGE/"alignment/protocol_v7.json",
                 "dataset":DATA/"dataset_manifest.json",
                 "features":PACKAGE/"data/builds/perfume-m2048-d8-v7/manifest.json",
                 "reference":PACKAGE/"v7_reference/manifest.json",
                 "compute_profile":PACKAGE/"alignment/campus_compute_profile.json"}.items():
    assert sha(path)==identity[key],key
source_count=verify(PACKAGE,identity["source"])
package_count=verify(PACKAGE,read(PACKAGE/"CAMPUS_PACKAGE_MANIFEST.json")["files"])
verify(DATA,read(DATA/"dataset_manifest.json")["files"])
freeze=read(RUN/"selection_frozen.json")
freeze_count=verify(RUN,freeze["files"])
assert freeze["test_used_for_selection"] is False
with np.load(DATA/"test.npz",allow_pickle=False) as z:
    truth=z["truth"]; labels=z["labels"].tolist() if "labels" in z.files else read(DATA/"dataset_manifest.json")["labels"]
    test_indices=z["row_index"]
with np.load(DATA/"train.npz",allow_pickle=False) as z:
    train_y=z["truth"]; train_indices=z["row_index"]
assert truth.shape==(1333,109) and train_y.shape==(5353,109)
assert len(set(train_indices)&set(test_indices))==0
allgroups=pd.read_csv(DATA/"groups.csv").group_id.to_numpy()
groups=allgroups[train_indices]
assert not set(groups)&set(allgroups[test_indices])
comparison=pd.read_csv(RUN/"comparison_all.csv").set_index("candidate")
recomputed=[]; cv_scores={}; fold_counts=set(); selected_labels=[]; model_count=0
for seed in SEEDS:
    assert len({sha(RUN/f"robustness/{k}/seed_{seed}/folds.json") for k in CANDIDATES})==1
    folds=read(RUN/f"robustness/lgbm_D/seed_{seed}/folds.json")["folds"]
    for label, parts in folds.items():
        fold_counts.add(len(parts))
        heldout=[]
        for part in parts:
            a=np.asarray(part["train"]);b=np.asarray(part["validation"])
            assert not set(groups[a])&set(groups[b])
            assert len(set(a)&set(b))==0 and len(a)+len(b)==len(train_y)
            heldout.extend(b)
        assert sorted(heldout)==list(range(len(train_y))), (seed,label)
for key in CANDIDATES:
    models=read(RUN/key/"models.json")
    assert set(models)==set(labels)
    model_count+=len(models)
    for item in models.values():
        assert sha(RUN/key/item["file"])==item["sha256"]
    for secondary in (False,True):
        suffix="_validation_thresholds" if secondary else ""
        with np.load(RUN/key/f"test_predictions{suffix}.npz",allow_pickle=False) as z:
            prob=z["probabilities"]; decisions=z["decisions"]
            assert np.array_equal(z["truth"],truth) and z["labels"].tolist()==labels
            assert prob.shape==truth.shape and np.isfinite(prob).all()
            assert ((prob>=0)&(prob<=1)).all()
            thresholds=z["thresholds"] if secondary else np.full(109,.5)
            assert np.array_equal(decisions,(prob>=thresholds).astype(int))
        tp=((truth==1)&(decisions==1)).sum(axis=0)
        tn=((truth==0)&(decisions==0)).sum(axis=0)
        fp=((truth==0)&(decisions==1)).sum(axis=0)
        fn=((truth==1)&(decisions==0)).sum(axis=0)
        div=lambda a,b:np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)
        ap=np.array([average_precision_score(truth[:,j],prob[:,j]) for j in range(109)])
        result={"accuracy":float(((tp+tn)/len(truth)).mean()),
                "auroc":float(np.mean([roc_auc_score(truth[:,j],prob[:,j]) for j in range(109)])),
                "auprc":float(ap.mean()),"specificity":float(div(tn,tn+fp).mean()),
                "precision":float(div(tp,tp+fp).mean()),"recall":float(div(tp,tp+fn).mean()),
                "macro_f1":float(div(2*tp,2*tp+fp+fn).mean()),
                "mean_predicted_labels":float(decisions.sum(axis=1).mean())}
        saved=read(RUN/key/f"test_summary{suffix}.json")
        for metric,value in saved["summary"]["metrics"].items():
            close(result[metric],value,f"{key}{suffix}:{metric}")
        close(result["macro_f1"],saved["diagnostics"]["macro_f1"],f"{key}{suffix}:F1")
        if not secondary:
            for metric in saved["summary"]["metrics"]:
                close(result[metric],comparison.loc[key,metric],f"CSV:{key}:{metric}")
            if key=="lgbm_D":
                selected_labels=[{"label":label,"positives":int(truth[:,j].sum()),
                                 "prevalence":float(truth[:,j].mean()),"ap":float(ap[j]),
                                 "precision":float(div(tp,tp+fp)[j]),"recall":float(div(tp,tp+fn)[j]),
                                 "f1":float(div(2*tp,2*tp+fp+fn)[j])} for j,label in enumerate(labels)]
        close(result["macro_f1"],comparison.loc[key,"macro_f1_validation_threshold" if secondary else "macro_f1_0_5"],"CSV F1")
        recomputed.append({"candidate":key,"threshold":"OOF" if secondary else "0.5",**result})
    values=[]
    for seed in SEEDS:
        folder=RUN/f"robustness/{key}/seed_{seed}"
        recorded=read(folder/"cv_fold_metrics.json")
        from_saved=float(np.mean([np.mean([fold["metrics"]["auprc"] for fold in recorded[label]]) for label in labels]))
        if key in ("xgb_D","lgbm_D"):
            folds=read(folder/"folds.json")["folds"]
            with np.load(folder/"cv_oof_predictions.npz",allow_pickle=False) as z:
                assert np.array_equal(z["truth"],train_y) and z["labels"].tolist()==labels
                scores=z["probabilities"]
            raw=float(np.mean([np.mean([average_precision_score(train_y[part["validation"],j],scores[part["validation"],j])
                                      for part in folds[label]]) for j,label in enumerate(labels)]))
            close(raw,from_saved,f"Raw OOF AP:{key}:{seed}")
        values.append(from_saved)
    close(np.mean(values),comparison.loc[key,"cv_ap_mean"],f"CV mean:{key}")
    close(np.std(values,ddof=1),comparison.loc[key,"cv_ap_sd"],f"CV SD:{key}")
    cv_scores[key]=dict(zip(map(str,SEEDS),values))
    print(f"Metrics verified: {key}",flush=True)
pd.DataFrame(recomputed).to_csv(OUT/"recomputed_metrics.csv",index=False)
pd.DataFrame(selected_labels).sort_values("ap",ascending=False).to_csv(OUT/"selected_model_per_label.csv",index=False)

# Verify chosen thresholds against training OOF scores, without selecting from test.
threshold_info=read(RUN/"robustness/lgbm_D/thresholds.json")
with np.load(RUN/"robustness/lgbm_D/pooled_oof_predictions.npz",allow_pickle=False) as z:
    pooled=z["probabilities"]
    assert np.array_equal(z["truth"],train_y)
seed_probs=[]
for seed in SEEDS:
    with np.load(RUN/f"robustness/lgbm_D/seed_{seed}/cv_oof_predictions.npz",allow_pickle=False) as z:
        seed_probs.append(z["probabilities"])
assert np.allclose(pooled,np.mean(seed_probs,axis=0),rtol=1e-6,atol=1e-7)
for j,label in enumerate(labels):
    chosen=threshold_info["labels"][label]["threshold"]
    assert chosen in threshold_info["grid"]
    best=max(f1_score(train_y[:,j],pooled[:,j]>=t,zero_division=0) for t in threshold_info["grid"])
    close(f1_score(train_y[:,j],pooled[:,j]>=chosen,zero_division=0),best,"OOF threshold maximum")
selection=max(CANDIDATES,key=lambda k:np.mean(list(cv_scores[k].values())))
assert selection==freeze["overall_selected_from_repeated_grouped_cv"]=="lgbm_D"
bundle=read(RUN/"inference_bundle.json")
assert bundle["selected_candidate"]==selection
assert bundle["feature_spec"]["dimensions"]["input"]==2056 and bundle["labels"]==labels
assert bundle["selection_sha256"]==sha(RUN/"selection_frozen.json")

tuning={}
for key in ("xgb_C","xgb_D","lgbm_C","lgbm_D"):
    table=pd.read_csv(RUN/key/"trials.csv"); done=table[table.state=="COMPLETE"]
    saved=read(RUN/key/"tuning_summary.json"); failed=table[table.state=="FAIL"]
    assert len(done)==saved["complete_trials"] and len(failed)==saved["failed_trials"]
    assert failed.user_attrs_stop_reason.eq("budget_exhausted").all()
    assert saved["budget"]["status"]=="complete" and saved["budget"]["charged_seconds"]==28800
    best=done.sort_values(["value","number"],ascending=[False,True]).iloc[0]
    assert int(best.number)==saved["best_trial"]
    close(best.value,saved["best_macro_ap"],"Best Optuna trial")
    with sqlite3.connect((RUN/key/"optuna.sqlite3").as_uri()+"?mode=ro",uri=True) as con:
        states=dict(con.execute("SELECT state,COUNT(*) FROM trials GROUP BY state").fetchall())
    assert states=={"COMPLETE":len(done),"FAIL":len(failed)}
    tuning[key]={**saved,"sql_trial_states":states,"last_failure_reason":"budget_exhausted",
                 "recipe":read(RUN/key/"recipe.json"),
                 "completed_trial_numbers":done.number.astype(int).tolist()}
events=[json.loads(line) for line in (RUN/"events.jsonl").read_text().splitlines()]
begin=datetime.fromisoformat(events[0]["time"])
end=datetime.fromisoformat(read(RUN/"complete.json")["completed_at"])
frozen_time=datetime.fromisoformat(freeze["timestamp"])
first_final=min(datetime.fromisoformat(e["time"]) for e in events if e["stage"]=="final_fit_and_test")
assert frozen_time<=first_final
contrasts={}
for a in ("xgb","lgbm"):
    for lo,hi in (("A","B"),("C","D"),("A","C"),("B","D")):
        ds=[cv_scores[f"{a}_{hi}"][str(s)]-cv_scores[f"{a}_{lo}"][str(s)] for s in SEEDS]
        contrasts[f"{a}_{hi}-{lo}"]={"mean":float(np.mean(ds)),"deltas":ds,"positive_seeds":sum(v>0 for v in ds)}
gap=[cv_scores["lgbm_D"][str(s)]-cv_scores["xgb_D"][str(s)] for s in SEEDS]
report={
"status":"Completed artifacts and principal metrics verified; interpret with caveats",
"source_run":str(RUN),"source_manifest_sha256":sha(RUN/"completion_manifest.json"),
"completion_files_verified":checked,"source_files_verified":source_count,"package_files_verified":package_count,
"freeze_files_verified":freeze_count,"model_count":model_count,
"dataset":{"train_rows":len(train_y),"test_rows":len(truth),"labels":len(labels),
           "group_overlap_train_test":0,"group_overlap_cv":0,"fold_counts":sorted(fold_counts),
           "mean_test_positive_labels":float(truth.sum(axis=1).mean()),
           "all_negative_accuracy":float((truth==0).mean())},
"selected":selection,"cv_scores_by_seed":cv_scores,"cv_contrasts":contrasts,
"lgbm_minus_xgb_D":{"mean":float(np.mean(gap)),"per_seed":gap,"positive_seeds":sum(v>0 for v in gap)},
"test_metrics":recomputed,"tuning":tuning,
"start":begin.isoformat(),"completed":end.isoformat(),"elapsed_hours":(end-begin).total_seconds()/3600,
"selected_oof_thresholds_verified":True,"selected_before_final_fit":True,
"raw_oof_ap_recomputed_for":["xgb_D","lgbm_D"],
"caveats":["Historical test already inspected; exploratory evaluation.",
"Repeated CV is not nested CV; no independence/significance claim for seeds.",
"AP optimization does not guarantee improved F1 at threshold 0.5.",
"Eight-hour budgets are equal, completed trial counts differ.",
"XGBoost completed 12 trials per study; LGBM D best was its last completed trial.",
"V7 does not isolate five versus eight descriptors at a fixed Morgan length.",
"Unrecorded odor labels treated as negatives; label harmonization not expert-validated here.",
"No retraining, source pipeline edits, release promotion, or thesis edits during this audit."]}
(OUT/"audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({"verified_files":checked,"models":model_count,"selected":selection,
                  "hours":report["elapsed_hours"],"cv_fold_counts":sorted(fold_counts),
                  "all_negative_accuracy":report["dataset"]["all_negative_accuracy"],
                  "D_gap":report["lgbm_minus_xgb_D"],
                  "top_labels":sorted(selected_labels,key=lambda x:-x["ap"])[:5],
                  "bottom_labels":sorted(selected_labels,key=lambda x:x["ap"])[:5]},indent=2))
