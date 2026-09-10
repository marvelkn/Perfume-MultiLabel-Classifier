"""Run the complete campus experiment sequentially, without smoke or thermal telemetry."""
import argparse
import ctypes
import json
from datetime import datetime
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

from campus_support import (ROOT, HERE, ALGORITHMS, PROFILE, TARGET_ATTEMPTS, TUNING_SECONDS,
                            SESSION_SECONDS, KillJob, read, write, sha, charge, admit, best_complete)
sys.path.insert(0,str(ROOT))
import psutil


def reconcile_trials(run, algorithm):
    import optuna
    if not (run/"studies.sqlite3").exists(): return []
    try:
        study=optuna.load_study(storage=f"sqlite:///{(run/'studies.sqlite3').as_posix()}",study_name=algorithm)
    except KeyError: return []
    for trial in study.trials:
        if trial.state.name=="RUNNING":
            study.tell(trial.number,state=optuna.trial.TrialState.FAIL)
    study.trials_dataframe().to_csv(run/f"{algorithm}_trials.csv",index=False)
    return study.trials


def clean_stale_lock(directory):
    path=directory/".resource-session.lock"
    if not path.exists(): return
    value=read(path)
    if psutil.pid_exists(value["pid"]):
        raise RuntimeError(f"Resource lock PID {value['pid']} still exists; no second runner started")
    write(path.with_name("recovered-lock-"+uuid.uuid4().hex+".json"),value)
    path.unlink()


def progress(run, algorithm, operation, variant=None):
    if operation=="baseline": path=run/f"{algorithm}_baseline_progress.json"; key="entries"
    elif operation=="fit": path=run/algorithm/"fit_progress.json"; key="models"
    elif operation=="sensitivity": path=run/"sensitivity"/algorithm/variant/f"{algorithm}_baseline_progress.json"; key="entries"
    else: return 0
    return len(read(path).get(key,[])) if path.exists() else 0


def completed_artifact(run, algorithm, operation, variant=None):
    """Atomic completion markers take precedence over a lost worker acknowledgement."""
    try:
        if operation=="init":
            return read(run/"run.json").get("campus_amendment_sha256")==sha(HERE/"amendment.json")
        if operation=="baseline":
            values=read(run/f"{algorithm}_baselines.json")
            return set(values)=={"none","class_weight","random_oversample"} and all(len(v["fold_ap"])==3 for v in values.values())
        if operation=="fit":
            meta=read(run/algorithm/"model_manifest.json")
            return len(meta["models"])==25 and [m["label"] for m in meta["models"]]==meta["labels"] and all(
                Path(m["filename"]).name==m["filename"] and sha(run/algorithm/m["filename"])==m["sha256"] for m in meta["models"])
        if operation=="sensitivity":
            path=run/"sensitivity"/algorithm/variant
            return len(read(path/"metrics.json")["fold_macro_ap"])==3 and (path/"oof_predictions.npz").exists()
        if operation=="evaluate":
            return "average_precision_macro" in read(run/algorithm/"test_metrics.json") and (run/algorithm/"test_predictions.npz").exists()
        if operation=="export":
            index=read(run/algorithm/"onnx_location.json")
            directory=run/algorithm/index["directory"]
            meta=read(directory/"manifest.json")
            return len(meta["models"])==25 and all(sha(directory/m["filename"])==m["sha256"] for m in meta["models"])
        if operation=="benchmark": return read(run/algorithm/"desktop_benchmark.json")["model_count"]==25
    except (OSError,ValueError,KeyError,TypeError): return False
    return False


class Pipeline:
    def __init__(self, directory):
        self.directory=directory
        self.run=directory/"experiment"
        self.path=directory/"pipeline.json"
        identity={"amendment_sha256":sha(HERE/"amendment.json"),"host":platform.node(),
                  "python":platform.python_version()}
        if self.path.exists():
            self.state=read(self.path)
            if self.state["identity"]!=identity: raise ValueError("Run host or amendment changed")
        else:
            self.state={"identity":identity,"started_at":datetime.now().astimezone().isoformat(),
                        "status":"PREPARED","entries":[],"stages":{},"tuning":{},
                        "hardware":{"processor":platform.processor(),"logical_cpus":psutil.cpu_count(),
                                    "physical_cores":psutil.cpu_count(logical=False),"ram_bytes":psutil.virtual_memory().total,
                                    "platform":platform.platform()}}
            self.save()
        for entry in self.state["entries"]:
            if "elapsed_seconds" not in entry:
                entry["recovery"]="Controller interrupted; full reserved duration charged conservatively"
        self.save()

    def save(self): write(self.path,self.state)

    def invoke(self, operation, algorithm="xgb", seconds=SESSION_SECONDS, variant=None):
        started=time.perf_counter()
        session_id=uuid.uuid4().hex
        folder=self.directory/"sessions"/session_id; folder.mkdir(parents=True)
        request={"id":session_id,"operation":operation,"algorithm":algorithm,"variant":variant,
                 "run":str(self.run),"seconds":seconds,"gate":str(folder/"release"),"result":str(folder/"result.json")}
        write(folder/"request.json",request)
        entry={"id":session_id,"operation":operation,"algorithm":algorithm,"variant":variant,
               "reserved_seconds":seconds,"started_at":datetime.now().astimezone().isoformat()}
        self.state["entries"].append(entry); self.save()
        process=job=None
        outcome={"status":"FAILED","reason":"Worker was not launched"}
        try:
            clean_stale_lock(self.run)
            with (folder/"console.log").open("x",encoding="utf-8") as log, (folder/"resources.jsonl").open("x",encoding="utf-8") as samples:
                process=subprocess.Popen([sys.executable,"-u",str(HERE/"campus_worker.py"),str(folder/"request.json")],
                                         cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                job=KillJob(process)
                Path(request["gate"]).write_text("JOB_ATTACHED",encoding="utf-8")
                reason=None
                while process.poll() is None:
                    elapsed=time.perf_counter()-started
                    available=psutil.virtual_memory().available
                    parent=psutil.Process()
                    rss=parent.memory_info().rss
                    for child in parent.children(recursive=True):
                        try: rss+=child.memory_info().rss
                        except psutil.NoSuchProcess: pass
                    samples.write(json.dumps({"elapsed_seconds":elapsed,"rss_bytes":rss,"free_ram_bytes":available,"cpu_c":None})+"\n")
                    samples.flush()
                    if elapsed>=seconds: reason="TIME_LIMIT"
                    elif available<4*1024**3: reason="FREE_RAM_LIMIT"
                    elif rss>4*1024**3: reason="PROCESS_RAM_LIMIT"
                    if reason: break
                    time.sleep(.25)
                if reason:
                    outcome={"status":"RESOURCE","reason":reason}
                elif Path(request["result"]).exists():
                    outcome=read(request["result"])
                    if process.returncode!=0 and outcome["status"]=="OK": outcome={"status":"FAILED","reason":"Worker exit code contradicts result"}
                else: outcome={"status":"FAILED","reason":f"Worker exited {process.returncode} without result"}
        except KeyboardInterrupt:
            outcome={"status":"INTERRUPTED","reason":"User interrupted"}
            raise
        except Exception as exc:
            outcome={"status":"FAILED","reason":str(exc)}
        finally:
            if job: job.close()
            if process:
                if process.poll() is None: process.kill()
                process.wait(timeout=15)
            # Charge launch/import/checkpoint/repair overhead as active tuning time.
            if operation=="tune":
                try: reconcile_trials(self.run,algorithm)
                except Exception as exc: outcome={"status":"FAILED","reason":"Study reconciliation failed: "+str(exc)}
            entry.update(elapsed_seconds=time.perf_counter()-started,outcome=outcome)
            self.save()
        return outcome

    def stage(self, operation, algorithm="xgb", variant=None):
        key="/".join(filter(None,[operation,algorithm,variant]))
        if key in self.state["stages"]: return self.state["stages"][key]["status"]=="OK"
        if completed_artifact(self.run,algorithm,operation,variant):
            self.state["stages"][key]={"status":"OK","recovered":"Verified completion artifact"}
            self.save()
            return True
        print(f"START {key}",flush=True)
        result=None
        for attempt in range(6):
            before=progress(self.run,algorithm,operation,variant)
            result=self.invoke(operation,algorithm,variant=variant)
            if result["status"]=="OK": break
            if completed_artifact(self.run,algorithm,operation,variant):
                result={"status":"OK","recovered":"Completion artifact verified after interrupted acknowledgement", "worker_outcome":result}
                break
            timed=result["status"]=="RESOURCE" and ("time budget" in result.get("reason","").lower() or result.get("reason")=="TIME_LIMIT")
            if not timed or progress(self.run,algorithm,operation,variant)<=before: break
            print(f"RESUME checkpoint {key}",flush=True)
        self.state["stages"][key]=result; self.save()
        print(f"{result['status']} {key}",flush=True)
        return result["status"]=="OK"

    def tuning(self, algorithm):
        if algorithm in self.state["tuning"]: return
        overhead_started=time.perf_counter()
        consecutive=0; reason="TARGET_ATTEMPTS"
        trials=reconcile_trials(self.run,algorithm)
        while True:
            self.state["entries"].append({"id":uuid.uuid4().hex,"operation":"tune","algorithm":algorithm,
                "reserved_seconds":0,"elapsed_seconds":time.perf_counter()-overhead_started,"kind":"controller_overhead"})
            self.save()
            if not admit(self.state["entries"],algorithm,len(trials)): break
            remaining=TUNING_SECONDS-charge(self.state["entries"],algorithm)
            print(f"TUNE {algorithm}: attempt {len(trials)+1}/{TARGET_ATTEMPTS}; remaining {remaining/60:.1f} min",flush=True)
            before=len(trials)
            result=self.invoke("tune",algorithm,seconds=min(SESSION_SECONDS,remaining))
            overhead_started=time.perf_counter()
            trials=reconcile_trials(self.run,algorithm)
            if result["status"]=="RESOURCE": consecutive+=1
            else: consecutive=0
            if result["status"] not in {"OK","RESOURCE"}: reason="WORKER_FAILURE"; break
            if result["status"]=="RESOURCE" and ("RAM" in result.get("reason","") or "memory" in result.get("reason","").lower()):
                reason="RAM_LIMIT"; break
            if consecutive>=2: reason="TWO_CONSECUTIVE_RESOURCE_STOPS"; break
            if len(trials)==before: reason="NO_TRIAL_CREATED"; break
        # Include the final reconciliation even when an error exits the loop.
        if 'result' in locals() and reason != "TARGET_ATTEMPTS":
            self.state["entries"].append({"id":uuid.uuid4().hex,"operation":"tune","algorithm":algorithm,
                "reserved_seconds":0,"elapsed_seconds":time.perf_counter()-overhead_started,"kind":"controller_overhead"})
        if charge(self.state["entries"],algorithm)>=TUNING_SECONDS-1: reason="TIME_BUDGET"
        best=best_complete(trials)
        self.state["tuning"][algorithm]={"stop_reason":reason,"attempts":len(trials),
            "target_attempts":TARGET_ATTEMPTS,"complete":sum(t.state.name=="COMPLETE" for t in trials),
            "pruned":sum(t.state.name=="PRUNED" for t in trials),"failed":sum(t.state.name=="FAIL" for t in trials),
            "active_seconds":charge(self.state["entries"],algorithm),"best_trial":best.number if best else None,
            "best_cv_macro_ap":best.value if best else None,"resource_truncated":len(trials)<TARGET_ATTEMPTS}
        self.save()

    def execute(self):
        self.state["status"]="RUNNING"; self.save()
        if not self.stage("init"): raise RuntimeError("Run initialization failed")
        for algorithm in ALGORITHMS: self.stage("baseline",algorithm)
        for algorithm in ALGORITHMS: self.tuning(algorithm)
        for algorithm in ALGORITHMS:
            if self.state["tuning"][algorithm]["best_trial"] is not None: self.stage("fit",algorithm)
        variants=read(ROOT/"protocols/essenza_v1_20260908/protocol.json")["sensitivity"]["variant_order"]
        for algorithm in ALGORITHMS:
            if self.state["stages"].get(f"fit/{algorithm}",{}).get("status")=="OK":
                for variant in variants: self.stage("sensitivity",algorithm,variant)
        # Record every sensitivity outcome before any test access, including failures.
        write(self.run/"pretest_sensitivity_status.json",{k:v for k,v in self.state["stages"].items() if k.startswith("sensitivity/")})
        if all(self.state["stages"].get(f"fit/{a}",{}).get("status")=="OK" for a in ALGORITHMS):
            for algorithm in ALGORITHMS: self.stage("evaluate",algorithm)
        for algorithm in ALGORITHMS:
            if self.state["stages"].get(f"fit/{algorithm}",{}).get("status")=="OK":
                if self.stage("export",algorithm): self.stage("benchmark",algorithm)
        all_ok=all(v["status"]=="OK" for v in self.state["stages"].values())
        models=all((self.run/a/"model_manifest.json").exists() for a in ALGORITHMS)
        truncated=any(v["resource_truncated"] for v in self.state["tuning"].values())
        self.state["status"]="COMPLETE" if all_ok and models and not truncated else "FINISHED_WITH_LIMITATIONS"
        self.save()

    def package(self):
        summary={"status":self.state["status"],"tuning":self.state["tuning"],
                 "temperature_monitored":False,"smoke_executed":False,"models":{},
                 "stages":{k:{"status":v["status"],"reason":v.get("reason")} for k,v in self.state["stages"].items()}}
        for a in ALGORITHMS:
            model=self.run/a/"model_manifest.json"
            metrics=self.run/a/"test_metrics.json"
            summary["models"][a]={"native_model_available":model.exists(),"test_metrics":read(metrics) if metrics.exists() else None}
        summary["sensitivity"]={}
        for a in ALGORITHMS:
            variants={}
            for p in (self.run/"sensitivity"/a).glob("*/metrics.json"):
                variants[p.parent.name]=read(p)
            summary["sensitivity"][a]=variants
        if all(len(summary["sensitivity"][a])==4 for a in ALGORITHMS):
            import statistics
            summaries={}
            seeds=["random_seed_42","random_seed_123","random_seed_2026"]
            for a in ALGORITHMS:
                values=[summary["sensitivity"][a][s]["mean_fold_macro_ap"] for s in seeds]
                summaries[a]={"random_seed_mean":statistics.mean(values),"random_seed_sample_sd":statistics.stdev(values),
                              "scaffold_minus_random42":summary["sensitivity"][a]["scaffold_seed_42"]["mean_fold_macro_ap"]-values[0]}
            summaries["paired_xgb_minus_lgbm"]={s:summary["sensitivity"]["xgb"][s]["mean_fold_macro_ap"]-summary["sensitivity"]["lgbm"][s]["mean_fold_macro_ap"] for s in seeds}
            summary["sensitivity_descriptive_summary"]=summaries
        write(self.directory/"RESULTS_SUMMARY.json",summary)
        note="Status: "+summary["status"]+"\n\nSend this whole ZIP, not only model files. See RESULTS_SUMMARY.json, pipeline.json, experiment/*_trials.csv and experiment/{xgb,lgbm}/test_metrics.json.\nTemperature was not measured. 30 attempted trials / 4 active hours per learner were fixed before training. Missing/failed/truncated stages are explicitly reported. No global-optimum or statistical-significance claim.\n"
        (self.directory/"READ_RESULTS.txt").write_text(note,encoding="utf-8")
        files=[p for p in self.directory.rglob('*') if p.is_file() and not p.name.endswith('.lock')]
        write(self.directory/"result_checksums.json",{p.relative_to(self.directory).as_posix():sha(p) for p in files if p.name!='result_checksums.json'})
        target=ROOT/"campus-results"; target.mkdir(exist_ok=True)
        archive=target/(self.directory.name+"-"+datetime.now().strftime('%Y%m%d-%H%M%S')+".zip")
        with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED) as z:
            for p in self.directory.rglob('*'):
                if p.is_file() and not p.name.endswith('.lock'): z.write(p,p.relative_to(self.directory).as_posix())
        Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii')
        print("RESULT ZIP: "+str(archive),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--campus-no-temperature",action="store_true",required=True)
    args=parser.parse_args()
    amendment=read(HERE/"amendment.json")
    if os.name!='nt' or platform.node().casefold()==amendment["excluded_laptop_host"].casefold():
        raise ValueError("Only the campus Windows PC may run this pipeline")
    verifier=runpy.run_path(str(ROOT/"protocols/essenza_v1_20260908/verify_protocol.py"))
    verifier["verify"](); verifier["validate_files"](ROOT,amendment["files"])
    # Validate locked versions without fitting or loading feature matrices.
    import importlib.metadata
    protocol=read(ROOT/"protocols/essenza_v1_20260908/protocol.json")
    if platform.python_version()!=protocol["environment"]["python"]: raise ValueError("Python version differs")
    for name,version in protocol["environment"]["packages"].items():
        if importlib.metadata.version(name)!=version: raise ValueError("Package version differs: "+name)
    directory=args.run.resolve()
    if directory.parent!=ROOT/"runs" or not directory.name.startswith('campus-full-'):
        raise ValueError("Use runs/campus-full-* for a new or resumable run")
    directory.mkdir(parents=True,exist_ok=True)
    controller=ROOT/".local-tools/campus-full-controller"; controller.mkdir(parents=True,exist_ok=True)
    clean_stale_lock(controller)
    from src.runtime import exclusive_run
    with exclusive_run(controller):
        pipeline=Pipeline(directory)
        shutil.copyfile(HERE/"amendment.json",directory/"amendment.json")
        snapshot=directory/"snapshot"
        if not snapshot.exists():
            for relative in set(protocol["environment"]["source_files"])|set(amendment["files"]):
                dest=snapshot/relative; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/relative,dest)
            shutil.copytree(ROOT/"protocols",snapshot/"protocols")
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint]
        kernel.SetThreadExecutionState.restype=ctypes.c_uint
        if not kernel.SetThreadExecutionState(0x80000001):
            print("NOTE: automatic sleep prevention unavailable; adjust Windows sleep settings manually",flush=True)
        try: pipeline.execute()
        except BaseException as exc:
            pipeline.state.update(status="INTERRUPTED" if isinstance(exc,KeyboardInterrupt) else "FAILED",reason=str(exc))
            pipeline.save()
            raise
        finally:
            kernel.SetThreadExecutionState(0x80000000)
            pipeline.package()
    return 0 if pipeline.state["status"]=="COMPLETE" else 2


if __name__=='__main__': raise SystemExit(main())
