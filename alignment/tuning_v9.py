"""Equal active-time TPE budgets, with immutable 8 h / 16 h snapshots."""
from datetime import datetime, timezone
from pathlib import Path
import time

import joblib
import optuna

from . import experiments as core
from .metrics import summarize
from .tuning_v6 import BudgetLimits, atomic_dump, event, initialize, suggest


def best_by_budget(trials, seconds):
    eligible = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE
                and t.user_attrs.get("active_seconds_at_completion", float("inf")) <= seconds]
    return max(eligible, key=lambda t: (t.value, -t.number)) if eligible else None


def snapshots(folder, study, policy, charged):
    path = Path(folder) / "budget_snapshots.json"
    saved = core.read(path) if path.exists() else {}
    for seconds in policy["snapshot_seconds"]:
        key = str(seconds)
        if seconds > charged or key in saved:
            continue
        best = best_by_budget(study.trials, seconds)
        saved[key] = {
            "active_budget_seconds": seconds,
            "status": "available" if best else "no_completed_trial",
            "best_trial": best.number if best else None,
            "best_cv_ap": best.value if best else None,
            "parameters": best.user_attrs["resolved_parameters"] if best else None,
            "completed_trials": sum(t.state == optuna.trial.TrialState.COMPLETE
                and t.user_attrs.get("active_seconds_at_completion", float("inf")) <= seconds
                for t in study.trials),
            "interpretation": "Best fully completed trial by this active-time boundary; no test scores.",
        }
    core.write(path, saved)
    return saved


def tune(run, key, algorithm, x, y, meta, split, positions, protocol, initial_recipe):
    study = initialize(run, key, protocol, initial_recipe)
    try:
        return _tune(run, key, algorithm, x, y, meta, split, positions, protocol, study)
    finally:
        # Optuna 4.5 wraps RDBStorage; release Windows SQLite handles on pause/finish.
        storage = getattr(study._storage, "_backend", study._storage)
        storage.remove_session()
        storage.engine.dispose()


def _tune(run, key, algorithm, x, y, meta, split, positions, protocol, study):
    folder = Path(run) / key
    state_path = folder / "budget.json"
    state = core.read(state_path)
    policy = protocol["tuning"]
    # A hard crash can lose up to a heartbeat of recorded computation.
    if state["status"] == "running":
        allowance = policy["heartbeat_seconds"]
        state["charged_seconds"] = min(policy["seconds_per_study"], state["charged_seconds"] + allowance)
        state["recovery_allowance_seconds"] = state.get("recovery_allowance_seconds", 0) + allowance
        core.write(state_path, state)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    remaining = max(0., policy["seconds_per_study"] - state["charged_seconds"])
    snapshots(folder, study, policy, state["charged_seconds"])
    if remaining > 1 and len(study.trials) < policy["safety_max_trials"]:
        started, charged = time.perf_counter(), state["charged_seconds"]
        last_log = started

        def heartbeat(now):
            nonlocal last_log
            state.update(charged_seconds=min(policy["seconds_per_study"], charged + now - started),
                         last_heartbeat_at=datetime.now(timezone.utc).isoformat(), status="running")
            core.write(state_path, state)
            if now - last_log >= 60:
                event(run, "tuning", candidate=key, trial=state["active_trial"],
                      hours_used=round(state["charged_seconds"] / 3600, 3),
                      hours_budget=policy["seconds_per_study"] / 3600)
                snapshots(folder, study, policy, state["charged_seconds"])
                last_log = now

        limits = BudgetLimits(started + remaining, heartbeat, policy["heartbeat_seconds"])
        try:
            while time.perf_counter() < limits.deadline and len(study.trials) < policy["safety_max_trials"]:
                trial = study.ask()
                state["active_trial"] = trial.number
                heartbeat(time.perf_counter())
                try:
                    params = suggest(trial, algorithm, protocol)
                    trial.set_user_attr("resolved_parameters", params)
                    atomic_dump(study.sampler, folder / "sampler.joblib")
                    event(run, "trial_started", candidate=key, trial=trial.number)
                    values = core.cross_validate(algorithm, params, x, y, meta["labels"],
                        split["folds"], positions, limits, meta["summary_labels"])
                    score = summarize(values, meta["summary_labels"])["metrics"]["auprc"]
                    limits.check(force=True)
                    trial.set_user_attr("active_seconds_at_completion", charged + time.perf_counter() - started)
                    study.tell(trial, score)
                    event(run, "trial_completed", candidate=key, trial=trial.number,
                          macro_ap=score, best_macro_ap=study.best_value)
                except TimeoutError:
                    trial.set_user_attr("stop_reason", "budget_exhausted")
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    break
                except BaseException:
                    if study.trials[trial.number].state == optuna.trial.TrialState.RUNNING:
                        trial.set_user_attr("stop_reason", "interrupted_or_error")
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    raise
                finally:
                    state["active_trial"] = None
                    atomic_dump(study.sampler, folder / "sampler.joblib")
                    study.trials_dataframe().to_csv(folder / "trials.csv", index=False)
                    heartbeat(time.perf_counter())
                    snapshots(folder, study, policy, state["charged_seconds"])
        finally:
            state.update(charged_seconds=min(policy["seconds_per_study"], charged + time.perf_counter() - started),
                         active_trial=None)
            state["status"] = "complete" if state["charged_seconds"] >= policy["seconds_per_study"] - 1 else "paused"
            if state["status"] == "complete":
                state["charged_seconds"] = policy["seconds_per_study"]
            state["stop_reason"] = "time_budget_exhausted" if state["status"] == "complete" else "interrupted_or_safety_limit"
            core.write(state_path, state)
            atomic_dump(study.sampler, folder / "sampler.joblib")
            study.trials_dataframe().to_csv(folder / "trials.csv", index=False)
            snapshots(folder, study, policy, state["charged_seconds"])
    else:
        if remaining <= 1:
            state.update(status="complete", charged_seconds=policy["seconds_per_study"], active_trial=None,
                         stop_reason="time_budget_exhausted")
            core.write(state_path, state)
        study.trials_dataframe().to_csv(folder / "trials.csv", index=False)
        snapshots(folder, study, policy, state["charged_seconds"])
    if state["status"] != "complete":
        raise RuntimeError("Tuning belum lengkap; lanjutkan run yang sama. Jangan tandai selesai.")
    best = best_by_budget(study.trials, policy["seconds_per_study"])
    if best is None:
        raise RuntimeError(f"{key}: tidak ada trial selesai dalam budget; simpan hasil untuk diperiksa.")
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    core.write(folder / "tuning_summary.json", {
        "complete_trials": len(done),
        "failed_trials": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
        "best_trial": best.number, "best_macro_ap": best.value,
        "completed_trials_after_best": sum(t.number > best.number for t in done),
        "budget": state, "initial_recipe_recomputed": True,
        "snapshots": core.read(folder / "budget_snapshots.json"),
    })
    return best.user_attrs["resolved_parameters"]
