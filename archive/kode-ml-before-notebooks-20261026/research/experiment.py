"""Delapan kandidat penelitian utama. Default CLI hanya memeriksa input."""
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
from . import runtime as core
from .validation import repeated_grouped_cv, prediction_diagnostics
from .metrics import METRICS
from .splits import POLICY
from .dataset import DATASET, verify as verify_dataset
from .features import descriptor_count, features
from .runtime import ROOT, digest
from .features import DESCRIPTORS8, N_BITS, EXTRA, verify as verify_descriptors
from .training import fit_final_models
from .metrics import binary_metrics, summarize
OUTPUT = DATASET
from .tuning import tune, event, suggest

warnings.filterwarnings('ignore',message='X does not have valid feature names, but LGBMClassifier was fitted with feature names',category=UserWarning)
RUN_ID='perfume-main-research'
PROTOCOL=ROOT/'research/config/experiment.json'
ALGORITHMS=('xgb','lgbm')
CANDIDATES=tuple(f'{a}_{c}' for a in ALGORITHMS for c in 'ABCD')
TUNED=tuple(k for k in CANDIDATES if k[-1] in 'CD')

def verify_files(root, files):
    for name,expected in files.items():
        if digest(Path(root)/name)!=expected: raise ValueError(f'Artifact changed: {name}')

def preflight(dataset, extra=EXTRA):
    meta, _, feature_meta, _, _ = verify_dataset(dataset, extra)
    p = core.read(PROTOCOL)
    if p["id"] != RUN_ID or tuple(p["algorithms"]) != ALGORITHMS or p["conditions"] != list("ABCD"):
        raise ValueError("Wrong active research protocol.")
    if (p["environment"] != core.environment()
            or p["dataset_manifest_sha256"] != digest(Path(dataset)/"dataset_manifest.json")):
        raise ValueError("Dataset/environment mismatch.")
    if p["group_policy"] != POLICY or p["class_imbalance"]["policy"] != core.WEIGHTING_POLICY:
        raise ValueError("Grouped split or weighting mismatch.")
    if p["tuning"]["seconds_total"] != len(TUNED)*p["tuning"]["seconds_per_study"]:
        raise ValueError("Four equal tuning budgets required.")
    if not 5 <= p["tuning"]["heartbeat_seconds"] <= 60:
        raise ValueError("Invalid heartbeat.")
    if p["feature_manifest_sha256"] != digest(Path(extra)/"manifest.json"):
        raise ValueError("Descriptor build differs from the protocol.")
    if (p["features"]["descriptors"] != list(DESCRIPTORS8) or p["features"]["n_bits"] != N_BITS
            or p["features"]["radius"] != 2 or p["features"]["include_chirality"] is not False):
        raise ValueError("Unexpected descriptor definitions.")
    import optuna
    for algorithm in ALGORITHMS:
        recipe = p["initial_recipes"][algorithm]
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            if suggest(optuna.trial.FixedTrial(recipe), algorithm, p) != recipe:
                raise ValueError("Starting recipe is outside the locked search space.")
    meta = dict(meta, feature_spec=feature_meta["feature_spec"])
    return meta, p


def select_candidates(robustness):
    scores={k:v['mean_metrics']['auprc'] for k,v in robustness.items()}
    # Exact ties prefer A, B, C, D; model choice uses validation only.
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
        result={'role':'exploratory research, fixed threshold 0.5','summary':summarize(per_label,labels),'diagnostics':prediction_diagnostics(truth,decisions,labels),'selection_sha256':digest(frozen),'artifact_sha256':{n:digest(folder/n) for n in ('test_predictions.npz','test_per_label.json')}}
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
        extra={'role':'secondary; thresholds chosen using training OOF only','summary':summarize(per_label,labels),'diagnostics':prediction_diagnostics(truth,decisions,labels),'selection_sha256':digest(frozen),'threshold_source_sha256':digest(thresholds_path),'artifact_sha256':{n:digest(folder/n) for n in ('test_predictions_validation_thresholds.npz','test_per_label_validation_thresholds.json')}}
        core.write(secondary,extra)
    return result,extra

def paired_cv(run, protocol):
    """Describe paired differences; repeated seeds are not independent experiments."""
    rows = []
    for seed in protocol["robustness"]["seeds"]:
        paths = [run/"robustness"/key/f"seed_{seed}"/"folds.json" for key in CANDIDATES]
        if len({digest(path) for path in paths}) != 1:
            raise ValueError("Candidates used different validation folds.")
    for algorithm in ALGORITHMS:
        for left, right, meaning in (("A", "B", "eight_descriptors_at_fixed_2048_baseline"),
                                      ("C", "D", "eight_descriptors_at_fixed_2048_equal_tuning_budget")):
            a, b = f"{algorithm}_{left}", f"{algorithm}_{right}"
            va = core.read(run/"robustness"/a/"robustness_summary.json")
            vb = core.read(run/"robustness"/b/"robustness_summary.json")
            deltas = [vb["per_seed"][str(s)]["metrics"]["auprc"]
                      - va["per_seed"][str(s)]["metrics"]["auprc"]
                      for s in protocol["robustness"]["seeds"]]
            rows.append({"algorithm": algorithm, "contrast": f"{right}-{left}",
                         "interpretation": meaning, "seeds": protocol["robustness"]["seeds"],
                         "delta_per_seed": deltas, "mean_delta": float(np.mean(deltas)),
                         "sd_delta": float(np.std(deltas, ddof=1)),
                         "added_features_wins": sum(d > 0 for d in deltas),
                         "morgan_only_wins": sum(d < 0 for d in deltas),
                         "ties": sum(d == 0 for d in deltas)})
    result = {"contrasts": rows, "identical_paired_folds": True,
              "caveat": "Descriptive repeated-CV differences, not independent samples, p-values or equivalence tests."}
    core.write(run/"paired_descriptor_cv.json", result)
    return result


def summaries(run, meta, protocol, freeze):
    rows = []
    for key in CANDIDATES:
        rob = core.read(run/"robustness"/key/"robustness_summary.json")
        test = core.read(run/key/"test_summary.json")
        extra = core.read(run/key/"test_summary_validation_thresholds.json")
        rows.append({"candidate": key, "descriptors": descriptor_count(key),
                     "tuned": key in TUNED, "cv_ap_mean": rob["mean_metrics"]["auprc"],
                     "cv_ap_sd": rob["std_metrics"]["auprc"], **test["summary"]["metrics"],
                     "macro_f1_0_5": test["diagnostics"]["macro_f1"],
                     "macro_f1_validation_threshold": extra["diagnostics"]["macro_f1"],
                     "selected_for_algorithm": key in freeze["selected_by_algorithm"].values(),
                     "selected_overall": key == freeze["overall_selected_from_repeated_grouped_cv"]})
    pd.DataFrame(rows).to_csv(run/"comparison_all.csv", index=False)
    paired = core.read(run/"paired_descriptor_cv.json")
    deltas = []
    label_rows = []
    for algorithm in ALGORITHMS:
        for left, right, meaning in (("A","B","eight_descriptors_fixed_2048_baseline"),
                                     ("C","D","eight_descriptors_fixed_2048_equal_tuning_budget"),
                                     ("A","C","tuning_morgan_2048"),
                                     ("B","D","tuning_morgan_2048_plus_eight")):
            a, b = f"{algorithm}_{left}", f"{algorithm}_{right}"
            va = next(r for r in rows if r["candidate"] == a)
            vb = next(r for r in rows if r["candidate"] == b)
            deltas.append({"algorithm": algorithm, "contrast": f"{right}-{left}", "meaning": meaning,
                           **{f"delta_{m}": vb[m]-va[m] for m in
                              ("cv_ap_mean", *METRICS, "macro_f1_0_5", "macro_f1_validation_threshold")}})
            if (left,right) in (("A","B"),("C","D")):
                ta, tb = (core.read(run/k/"test_per_label.json") for k in (a,b))
                for label in meta["labels"]:
                    label_rows.append({"algorithm":algorithm, "contrast":f"{right}-{left}",
                                       "label":label, "test_ap_morgan":ta[label]["auprc"],
                                       "test_ap_8":tb[label]["auprc"],
                                       "test_ap_delta":tb[label]["auprc"]-ta[label]["auprc"]})
    pd.DataFrame(deltas).to_csv(run/"descriptor_deltas.csv", index=False)
    pd.DataFrame(label_rows).to_csv(run/"descriptor_deltas_per_label_test.csv", index=False)
    overall = freeze["overall_selected_from_repeated_grouped_cv"]
    count = descriptor_count(overall)
    spec = dict(meta["feature_spec"])
    spec.update(descriptors=list(DESCRIPTORS8[:count]), use_descriptors=bool(count),
                dimensions={"input":meta["feature_spec"]["n_bits"]+count},
                dtype="float32", fragment_policy="single")
    feature_names = [f"morgan_{i:04d}" for i in range(spec["n_bits"])] + list(DESCRIPTORS8[:count])
    core.write(run/"inference_bundle.json", {
        "protocol":RUN_ID, "selected_candidate":overall, "algorithm":overall.split("_")[0],
        "models_directory":overall, "models":core.read(run/overall/"models.json"),
        "feature_spec":spec, "feature_names":feature_names, "use_descriptors":bool(count),
        "labels":meta["labels"], "primary_threshold":0.5,
        "secondary_thresholds_path":f"robustness/{overall}/thresholds.json",
        "environment":core.environment(), "selection_sha256":digest(run/"selection_frozen.json"),
        "score_interpretation":"Uncalibrated score, not intensity or mixture composition.",
        "release_policy":"Research only; ONNX and Android remain deferred.",
        "prediction_command":"python -m research.inference --run NEW_RUN_DIRECTORY --smiles SMILES"})
    lines = ["# Hasil penelitian: Morgan 2048 dengan/tanpa delapan deskriptor", "",
             f"Pemenang berdasarkan mean macro AP validasi: **{overall}** ({count} deskriptor).", "",
             protocol["confirmatory_status"], "",
             "| Kandidat | Deskriptor | CV AP | SD | Test AP | AUROC | Macro F1 (0,5) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['candidate']} | {r['descriptors']} | {r['cv_ap_mean']:.6f} | {r['cv_ap_sd']:.6f} | {r['auprc']:.6f} | {r['auroc']:.6f} | {r['macro_f1_0_5']:.6f} |")
    lines += ["", "## Selisih validasi: penambahan delapan deskriptor pada Morgan 2048", ""]
    for row in paired["contrasts"]:
        lines.append(f"- {row['algorithm']} {row['contrast']}: {row['mean_delta']:+.6f}; "
                     f"Fitur tambahan unggul pada {row['added_features_wins']}/{len(row['seeds'])} seed.")
    lines += ["", "Angka tertinggi menentukan kandidat; kemenangan kecil/tidak konsisten belum membuktikan superioritas.",
              "B-A menguji delapan fitur pada baseline tetap. D-C membandingkan representasi setelah tuning masing-masing.",
              "AP di sini adalah sklearn Average Precision, bukan integrasi trapezoid kurva PR.",
              "Tidak ada p-value dari lima seed; seed/fold saling berkaitan dan CV ini bukan nested CV.",
              
              "Lihat descriptor_deltas.csv dan paired_descriptor_cv.json.",
              
              "ONNX/Android ditunda. Hasil arsip utama tidak diganti otomatis."]
    (run/"RESULTS_SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return {"protocol":RUN_ID, "completed_at":datetime.now(timezone.utc).isoformat(),
            "selected_by_algorithm":freeze["selected_by_algorithm"],
            "overall_selected_from_repeated_grouped_cv":overall, "comparison":rows,
            "paired_descriptor_cv":paired, "test_used_for_selection":False,
            "tuning_outcome":{k:core.read(run/k/"tuning_summary.json") for k in TUNED},
            "confirmatory_status":protocol["confirmatory_status"]}


def reject_historical_run(run):
    run = Path(run).resolve()
    if run.name.startswith("perfume-five-grouped-v") or ".local-tools" in run.parts or "archive" in run.parts:
        raise ValueError("Historical run is read-only. Use a new run directory for refactored source.")
    if run.exists() and any(run.iterdir()):
        identity = core.read(run / "identity.json") if (run / "identity.json").exists() else {}
        if identity.get("protocol") != digest(PROTOCOL) or not identity.get("source") or not all(k.startswith("research/") for k in identity["source"]):
            raise ValueError("Cannot resume a run with a historical or unknown source identity.")


def run_all(dataset, run, extra=EXTRA):
    dataset,run=Path(dataset).resolve(),Path(run).resolve()
    reject_historical_run(run)
    core.check_host()
    meta,protocol=preflight(dataset, extra)
    _, morgan2048, descriptors8 = verify_descriptors(dataset, extra)
    run.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(run).free<10*1024**3: raise RuntimeError('Need at least 10 GiB free disk for models and result ZIP.')
    identity={'protocol':digest(PROTOCOL),'dataset':digest(dataset/'dataset_manifest.json'),'features':digest(Path(extra)/'manifest.json'),'source':{p.relative_to(ROOT).as_posix():digest(p) for folder in ('research',) for p in (ROOT/folder).rglob('*.py')},'environment':core.environment(),'compute_profile':digest(core.COMPUTE_PROFILE),'host':platform.node(),'physical_cores':psutil.cpu_count(logical=False)}
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
            y,indices=z['truth'],z['row_index'].astype(int)
        morgan=morgan2048[indices]
        desc=descriptors8[indices]
        if not frozen.exists():
            refs=protocol['initial_recipes']
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
                recipes[key]=dict(protocol['baseline_parameters']) if key not in TUNED else tune(run,key,key.split('_')[0],features(key,morgan,desc),y,meta,split,positions,protocol,refs[key.split('_')[0]])
                core.write(folder/'training_policy.json',{'condition':key[-1],'weighting':core.WEIGHTING_POLICY,'tuning_budget_seconds':protocol['tuning']['seconds_per_study'] if key in TUNED else 0})
                core.write(done,recipes[key])
            groups=pd.read_csv(dataset/'groups.csv').group_id.to_numpy()[indices]
            robustness={}
            for key in CANDIDATES:
                event(run,'repeated_grouped_cv',candidate=key,seeds=protocol['robustness']['seeds'])
                robustness[key]=repeated_grouped_cv(run,key,key.split('_')[0],recipes[key],features(key,morgan,desc),y,meta['labels'],groups,protocol)
            paired_cv(run, protocol)
            overall,selected,scores=select_candidates(robustness)
            paths=[p for k in CANDIDATES for p in (run/k).iterdir() if p.is_file() and not p.name.endswith('.tmp')]
            paths.append(run/'paired_descriptor_cv.json')
            paths.extend(p for p in (run/'robustness').rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
            core.write(frozen,{'protocol':RUN_ID,'timestamp':datetime.now(timezone.utc).isoformat(),'selected_by_algorithm':selected,'overall_selected_from_repeated_grouped_cv':overall,'validation_scores':scores,'test_used_for_selection':False,'files':{p.relative_to(run).as_posix():digest(p) for p in paths}})
        freeze=core.read(frozen)
        verify_files(run,freeze['files'])
        # No tuning/CV mutations after freeze: safe resume even after test evaluation starts.
        with np.load(dataset/'test.npz',allow_pickle=False) as z:
            yt,test_indices=z['truth'],z['row_index'].astype(int)
        if test_indices.tolist()!=core.read(dataset/'splits.json')['test']:
            raise ValueError('Test row order changed.')
        mt=morgan2048[test_indices]
        dt=descriptors8[test_indices]
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
    parser.add_argument('--train',action='store_true',help='Explicit opt-in for a separately authorized new run.')
    parser.add_argument('--features','--descriptors',dest='features',type=Path,default=EXTRA)
    args=parser.parse_args()
    if args.check or not args.train:
        meta,p=preflight(args.dataset,args.features)
        print({'status':'MAIN_M2048_D8_INPUTS_VERIFIED','candidates':list(CANDIDATES),'labels':len(meta['labels']),'feature_dimensions':[N_BITS,N_BITS+8],'hours_per_study':p['tuning']['seconds_per_study']/3600,'total_tuning_hours':p['tuning']['seconds_total']/3600,'training_started':False})
    else: print(run_all(args.dataset,args.run,args.features))

if __name__=='__main__': main()
