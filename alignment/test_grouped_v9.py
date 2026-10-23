"""Focused V9 invariants; synthetic integration never fits the research cohort."""
import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd

from . import experiments as core
from . import experiments_v9 as v9
from . import tuning_v9


class V9Tests(unittest.TestCase):
    def test_both_sizes_keep_exact_same_descriptors(self):
        rng = np.random.default_rng(4)
        morgan = rng.integers(0, 2, (7, 2048), dtype=np.uint8)
        desc = rng.normal(size=(7, 8)).astype(np.float32)
        small = v9.features("xgb_m1024_base", morgan, desc)
        large = v9.features("lgbm_m2048_tuned", morgan, desc)
        self.assertEqual(small.shape, (7, 1032))
        self.assertEqual(large.shape, (7, 2056))
        np.testing.assert_array_equal(small[:, :1024], np.maximum(morgan[:, :1024], morgan[:, 1024:]))
        np.testing.assert_array_equal(small[:, -8:], large[:, -8:])
        with self.assertRaises(ValueError):
            v9.features("xgb_m512_base", morgan, desc)

    def test_snapshot_excludes_trials_completed_after_boundary(self):
        study = optuna.create_study(direction="maximize")
        for score, completed in ((.3, 10), (.9, 21), (.4, 20)):
            study.add_trial(optuna.trial.create_trial(value=score, user_attrs={
                "active_seconds_at_completion": completed, "resolved_parameters": {"n_estimators": 100}}))
        self.assertEqual(tuning_v9.best_by_budget(study.trials, 20).number, 2)
        self.assertIsNone(tuning_v9.best_by_budget(study.trials, 9))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1], prefix="v9-test-") as tmp:
            saved = tuning_v9.snapshots(tmp, study, {"snapshot_seconds": [10, 20]}, 20)
            self.assertEqual(saved["10"]["best_trial"], 0)
            self.assertEqual(saved["20"]["best_trial"], 2)

    def test_selection_uses_validation_and_fixed_tie_order(self):
        rob = {k: {"mean_metrics": {"auprc": .3}} for k in v9.CANDIDATES}
        overall, _, _ = v9.select_candidates(rob)
        self.assertEqual(overall, v9.CANDIDATES[0])
        rob["lgbm_m1024_tuned"]["mean_metrics"]["auprc"] = .31
        self.assertEqual(v9.select_candidates(rob)[0], "lgbm_m1024_tuned")

    def test_time_budget_resume_and_snapshots(self):
        protocol = copy.deepcopy(core.read(v9.PROTOCOL))
        protocol["tuning"].update(seconds_per_study=2.3, snapshot_seconds=[1.0, 2.3], heartbeat_seconds=.1)
        calls = []
        def fake_cv(*args, **kwargs):
            calls.append(1)
            time.sleep(.65)
            args[7].check(force=True)
            return {"odor": {m: .3 for m in core.METRICS}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1], prefix="v9-test-") as tmp:
            run = Path(tmp)
            with patch.object(tuning_v9, "suggest", return_value={"n_estimators": 100}), patch.object(core, "cross_validate", side_effect=fake_cv):
                best = tuning_v9.tune(run, "tiny", "xgb", None, None,
                    {"labels": ["odor"], "summary_labels": ["odor"]}, {"folds": {}}, {}, protocol, {"n_estimators": 100})
                call_count = len(calls)
                again = tuning_v9.tune(run, "tiny", "xgb", None, None,
                    {"labels": ["odor"], "summary_labels": ["odor"]}, {"folds": {}}, {}, protocol, {"n_estimators": 100})
                self.assertEqual(len(calls), call_count)
                self.assertEqual(best, again)
                snapshots = core.read(run / "tiny/budget_snapshots.json")
                self.assertEqual(set(snapshots), {"1.0", "2.3"})
                self.assertEqual(core.read(run / "tiny/budget.json")["status"], "complete")

    def test_synthetic_full_run_freeze_inference_and_resume(self):
        # 48 synthetic training rows, two labels, two seeds/folds, one/two-tree models.
        rng = np.random.default_rng(9)
        fp = rng.integers(0, 2, (64, 2048), dtype=np.uint8)
        desc = rng.normal(size=(64, 8)).astype(np.float32)
        truth = np.column_stack((np.arange(64) % 2, (np.arange(64) // 2) % 2)).astype(np.int8)
        labels = ["label_a", "label_b"]
        protocol = copy.deepcopy(core.read(v9.PROTOCOL))
        protocol["baseline_parameters"] = {"n_estimators": 2, "max_depth": 2}
        protocol["robustness"].update(seeds=[7, 21], folds=2)
        meta = {"labels": labels, "summary_labels": labels,
                "feature_spec": {"radius": 2, "n_bits": 2048, "include_chirality": False}}
        original_make = core.make_model
        def tiny_model(algorithm, params):
            return original_make(algorithm, params).set_params(n_jobs=1)
        def fake_tune(run, key, *args):
            folder = run / key
            recipe = {"n_estimators": 2, "max_depth": 2}
            core.write(folder / "tuning_summary.json", {"budget": {"status": "complete"}})
            core.write(folder / "budget_snapshots.json", {
                "28800": {"status": "available", "best_trial": 0, "best_cv_ap": .3,
                          "parameters": {"n_estimators": 1, "max_depth": 2}, "completed_trials": 1},
                "57600": {"status": "available", "best_trial": 1, "best_cv_ap": .31,
                          "parameters": recipe, "completed_trials": 2}})
            return recipe
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1], prefix="v9-test-") as tmp:
            tmp = Path(tmp)
            dataset, extra, run = tmp / "data", tmp / "features", tmp / "run"
            dataset.mkdir(); extra.mkdir()
            core.write(dataset / "dataset_manifest.json", meta)
            core.write(extra / "manifest.json", {"test": "synthetic"})
            core.write(dataset / "splits.json", {"train": list(range(48)), "test": list(range(48, 64)), "folds": {}})
            pd.DataFrame({"group_id": np.arange(64)}).to_csv(dataset / "groups.csv", index=False)
            core.save_npz(dataset / "train.npz", truth=truth[:48], row_index=np.arange(48))
            core.save_npz(dataset / "test.npz", truth=truth[48:], row_index=np.arange(48, 64))
            with patch.object(core, "check_host"), patch.object(v9, "preflight", return_value=(meta, protocol)), \
                 patch.object(v9, "verify_features", return_value=(meta, fp, desc)), \
                 patch.object(v9, "tune", side_effect=fake_tune), patch.object(core, "make_model", side_effect=tiny_model):
                v9.run_all(dataset, run, extra)
                complete = core.read(run / "complete.json")
                self.assertEqual(len(complete["comparison"]), 8)
                self.assertFalse(complete["test_used_for_selection"])
                frozen = core.read(run / "selection_frozen.json")
                self.assertFalse(any("test_" in name for name in frozen["files"]))
                v9.verify_files(run, frozen["files"])
                bundle = core.read(run / "inference_bundle.json")
                self.assertEqual(len(bundle["feature_names"]), bundle["feature_spec"]["n_bits"] + 8)
                self.assertTrue((run / "budget_comparison.csv").exists())
                with patch.object(core, "fit", side_effect=AssertionError("Complete run must not refit")):
                    v9.run_all(dataset, run, extra)

    def test_interruption_preserves_charged_budget(self):
        protocol = copy.deepcopy(core.read(v9.PROTOCOL))
        protocol["tuning"].update(seconds_per_study=2.5, snapshot_seconds=[1.0, 2.5], heartbeat_seconds=.1)
        def interrupted(*args, **kwargs):
            time.sleep(.15)
            raise KeyboardInterrupt()
        def resumed(*args, **kwargs):
            time.sleep(.3)
            args[7].check(force=True)
            return {"odor": {m: .3 for m in core.METRICS}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1], prefix="v9-test-") as tmp:
            run = Path(tmp)
            arguments = (run, "pause", "xgb", None, None, {"labels": ["odor"], "summary_labels": ["odor"]},
                         {"folds": {}}, {}, protocol, {"n_estimators": 100})
            with patch.object(tuning_v9, "suggest", return_value={"n_estimators": 100}):
                with patch.object(core, "cross_validate", side_effect=interrupted):
                    with self.assertRaises(KeyboardInterrupt):
                        tuning_v9.tune(*arguments)
                paused = core.read(run / "pause/budget.json")
                self.assertEqual(paused["status"], "paused")
                self.assertGreater(paused["charged_seconds"], .1)
                with patch.object(core, "cross_validate", side_effect=resumed):
                    tuning_v9.tune(*arguments)
                finished = core.read(run / "pause/tuning_summary.json")
                self.assertEqual(finished["budget"]["charged_seconds"], 2.5)
                self.assertGreaterEqual(finished["failed_trials"], 1)


if __name__ == "__main__":
    unittest.main()
