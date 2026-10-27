from pathlib import Path
import time

import optuna

from alignment import experiments_v5 as v5
from alignment.data import digest
from alignment.perfume_data import OUTPUT


def test_v5_protocol_and_inputs_are_locked():
    meta, protocol = v5.preflight(OUTPUT)
    assert protocol["id"] == v5.RUN_ID
    assert protocol["condition"] == "D"
    assert tuple(protocol["algorithms"]) == v5.ALGORITHMS
    assert len(meta["labels"]) == 109
    assert protocol["tuning"]["additional_seconds_total"] == 8 * 60 * 60
    assert protocol["tuning"]["heartbeat_seconds"] == 15


def test_v5_seed_database_states_are_exact():
    seed = v5.core.read(v5.SEED_MANIFEST)
    for key in v5.CANDIDATES:
        study = optuna.load_study(
            study_name=key,
            storage=f"sqlite:///{(v5.SEED_ROOT / key / 'optuna.sqlite3').as_posix()}",
        )
        states = v5.trial_state_counts(study)
        expected = seed["studies"][key]
        assert len(study.trials) == expected["database_trials"]
        assert states["COMPLETE"] == expected["completed_trials"]
        assert states["FAIL"] == expected["failed_trials"]


def test_initialization_creates_fresh_interruption_safe_budgets(tmp_path):
    _, protocol = v5.preflight(OUTPUT)
    run = Path(tmp_path) / "run"
    run.mkdir()
    v5.initialize_studies(run, protocol)
    marker = v5.core.read(run / "seed_initialized.json")
    assert marker["seed_manifest_sha256"] == digest(v5.SEED_MANIFEST)
    for key in v5.CANDIDATES:
        state = v5.core.read(run / key / "v5_budget.json")
        assert state["charged_seconds"] == 0
        assert "active_reservation" not in state
        assert state["budget_seconds"] == 4 * 60 * 60


def test_resume_does_not_overwrite_heartbeat_budget(tmp_path):
    _, protocol = v5.preflight(OUTPUT)
    run = Path(tmp_path) / "run"
    run.mkdir()
    v5.initialize_studies(run, protocol)
    state_path = run / "lgbm_D" / "v5_budget.json"
    state = v5.core.read(state_path)
    state["charged_seconds"] = 321.5
    state["status"] = "running"
    v5.core.write(state_path, state)
    v5.initialize_studies(run, protocol)
    assert v5.core.read(state_path)["charged_seconds"] == 321.5


def test_heartbeat_limits_persist_only_at_interval():
    calls = []
    limits = v5.HeartbeatLimits(
        time.perf_counter() + 60,
        lambda now: calls.append(now),
        interval=15,
    )
    limits.check(force=True)
    limits.check(force=True)
    assert len(calls) == 1


def test_selection_prefers_incumbent_on_tie_and_candidate_on_improvement(monkeypatch):
    robustness = {
        "xgb_D": {"mean_metrics": {"auprc": 0.31}},
        "lgbm_D": {"mean_metrics": {"auprc": 0.30}},
    }
    values = {"xgb_D_v3": 0.31, "lgbm_D_v3": 0.29}
    monkeypatch.setattr(
        v5.core,
        "read",
        lambda path: {"mean_metrics": {"auprc": values[Path(path).parent.name]}},
    )
    overall, selected, scores = v5.select_candidates(robustness)
    assert selected == {"xgb": "xgb_D_v3", "lgbm": "lgbm_D"}
    assert overall == "xgb_D_v3"
    assert scores["lgbm_D"] > scores["lgbm_D_v3"]
