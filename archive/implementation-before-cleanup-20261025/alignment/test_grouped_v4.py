from pathlib import Path

import optuna
import pandas as pd

from alignment import experiments_v4 as v4
from alignment.data import digest
from alignment.perfume_data import OUTPUT


def test_v4_protocol_and_inputs_are_locked():
    meta, protocol = v4.preflight(OUTPUT)
    assert protocol["id"] == v4.RUN_ID
    assert tuple(protocol["algorithms"]) == v4.ALGORITHMS
    assert len(meta["labels"]) == 109
    assert protocol["tuning"]["additional_seconds_total"] == 8 * 60 * 60
    assert protocol["tuning"]["seconds_per_algorithm"] == 8 * 60 * 60


def test_warm_start_trial_registries_are_exact():
    expected = {"xgb_C": 10, "xgb_D": 10, "lgbm_C": 43, "lgbm_D": 52}
    for key, count in expected.items():
        trials = pd.read_csv(v4.SEED_ROOT / key / "trials.csv")
        assert int((trials.state == "COMPLETE").sum()) == count
        study = optuna.load_study(
            study_name=key,
            storage=f"sqlite:///{(v4.SEED_ROOT / key / 'optuna.sqlite3').as_posix()}",
        )
        assert sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) == count


def test_seed_initialization_is_resumable_and_does_not_overwrite(tmp_path):
    run = Path(tmp_path) / "run"
    run.mkdir()
    v4.initialize_studies(run)
    state = run / "lgbm_C" / "tuning_budget.json"
    changed = state.read_bytes() + b"\n"
    state.write_bytes(changed)
    v4.initialize_studies(run)
    assert state.read_bytes() == changed
    assert v4.core.read(run / "seed_initialized.json")["seed_manifest_sha256"] == digest(
        v4.SEED_MANIFEST
    )


def test_selection_is_fair_and_prefers_each_incumbent_on_tie():
    robustness = {
        key: {"mean_metrics": {"auprc": value}}
        for key, value in {
            "xgb_C": 0.31, "xgb_D": 0.30, "lgbm_C": 0.29, "lgbm_D": 0.28
        }.items()
    }
    reference = {"incumbents": {
        "xgb_D_v3": {"repeated_cv_mean_auprc": 0.31},
        "lgbm_D_v3": {"repeated_cv_mean_auprc": 0.29},
    }}
    overall, selected, best, scores = v4.choose_candidates(robustness, reference)
    assert selected == {"xgb": "xgb_D_v3", "lgbm": "lgbm_D_v3"}
    assert best == {"xgb": "xgb_C", "lgbm": "lgbm_C"}
    assert overall == "xgb_D_v3"
    assert set(scores) == set(v4.CANDIDATES) | set(v4.INCUMBENTS.values())


def test_paired_comparisons_have_expected_direction():
    xgb = {"per_seed": {str(seed): {"metrics": {"auprc": 0.4 + seed / 1000}}
                         for seed in (7, 21, 42, 84, 100)}}
    lgbm = {"per_seed": {str(seed): {"metrics": {"auprc": 0.3 + seed / 1000}}
                          for seed in (7, 21, 42, 84, 100)}}
    paired = v4.paired_seed_comparison(xgb, lgbm)
    assert abs(paired["mean_difference"] - 0.1) < 1e-12
    labels = ["a", "b", "c"]
    xgb_labels = {name: {"auprc": 0.6} for name in labels}
    lgbm_labels = {name: {"auprc": 0.5} for name in labels}
    bootstrap = v4.paired_label_bootstrap(
        xgb_labels, lgbm_labels, labels, iterations=100, seed=42
    )
    assert bootstrap["percentile_95_ci"][0] > 0
    assert bootstrap["bootstrap_probability_xgb_higher"] == 1.0
