"""Synthetic control-flow tests: no estimator fitting or project dataset loading."""

import copy
import importlib.util
import io
import json
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
spec = importlib.util.spec_from_file_location(
    "smoke_runner", DIRECTORY / "run_smoke_probe.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Guard:
    def __init__(self, clock):
        self.clock = clock

    def check(self, force=False):
        if self.clock.now >= 300:
            raise runner.ResourceLimit("Synthetic shared session deadline")


class FakeBackend:
    def __init__(self, protocol, clock):
        self.protocol = protocol
        self.clock = clock
        self.calls = []
        self.fit_seconds = 1.0
        self.invalid_predictions = False
        self.fail_lgbm = False
        self.wrong_manifest = False

    def load(self, path):
        self.clock.now += 1
        d = self.protocol["dataset"]
        return (
            np.zeros((8, 2)),
            np.tile(np.array([0, 1] * 4)[:, None], (1, 25)),
            {
                "dataset_id": "WRONG" if self.wrong_manifest else d["dataset_id"],
                "feature_schema_id": d["feature_schema_id"],
                "labels": d["label_order"],
            },
        )

    def resample(self, X, Y, strategy, **kwargs):
        self.clock.now += 0.1
        return X, Y

    def fit(self, algorithm, params, X, y, **kwargs):
        self.calls.append((algorithm, params, kwargs))
        if algorithm == "lgbm" and self.fail_lgbm:
            raise RuntimeError("Synthetic learner failure")
        self.clock.now += self.fit_seconds
        invalid = self.invalid_predictions

        class Model:
            def predict_proba(self, X):
                return np.full((len(X), 2), np.nan if invalid else 0.5)

        return Model(), 10

    def operations(self):
        return {
            "np": np,
            "pool": lambda **kwargs: nullcontext(),
            "load": self.load,
            "fit": self.fit,
            "resample": self.resample,
        }


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(
            (ROOT / "protocols/essenza_v1_20260908/protocol.json").read_text()
        )
        self.protocol = copy.deepcopy(self.protocol)
        self.protocol["dataset"].update(development_rows=8, features=2)
        self.splits = {
            "folds": [{"train": [0, 1, 2, 3], "stop": [4, 5], "score": [6, 7]}]
        }
        self.settings = yaml.safe_load((ROOT / "config.safe-smoke.yaml").read_text())[
            "resources"
        ]
        self.clock = Clock()
        self.guard = Guard(self.clock)
        self.fake = FakeBackend(self.protocol, self.clock)
        self.result = {"backend_import_seconds": 0.25, "trial_seconds_estimates": None}
        self.events = io.StringIO()

    def measure(self):
        runner.measure(
            self.protocol,
            self.splits,
            self.guard,
            self.fake.operations(),
            self.result,
            self.events,
            clock=self.clock,
        )

    def test_all_frozen_units_and_estimate(self):
        self.measure()
        self.assertEqual(len(self.fake.calls), 18)
        for algorithm in ("xgb", "lgbm"):
            measured = self.result["measurements"][algorithm]
            self.assertEqual(len(measured["units"]), 9)
            expected = 1.5 * (1.25 + 3 * (0.1 + 25))
            self.assertAlmostEqual(
                self.result["trial_seconds_estimates"][algorithm], expected
            )
        for algorithm, params, kwargs in self.fake.calls:
            self.assertEqual(
                params, self.protocol["smoke_measurement"]["probe_params"][algorithm]
            )
            self.assertEqual(kwargs["threads"], 1)
            self.assertEqual(kwargs["seed"], 42)
            self.assertIs(kwargs["guard"], self.guard)
            self.assertEqual(len(kwargs["stop"][1]), 2)
        completed = [
            json.loads(line)
            for line in self.events.getvalue().splitlines()
            if json.loads(line)["event"] == "unit_completed"
        ]
        self.assertEqual(
            [(r["algorithm"], r["strategy"], r["label"]) for r in completed],
            [
                (a, s, label)
                for a in ("xgb", "lgbm")
                for s in self.protocol["smoke_measurement"]["strategies"]
                for label in ("musty", "citrus", "fruity")
            ],
        )
        self.assertEqual(
            [call[2]["weighting"] for call in self.fake.calls],
            ([False] * 3 + [True] * 3 + [False] * 3) * 2,
        )

    def test_deadline_shared_across_learners(self):
        self.fake.fit_seconds = 20
        with self.assertRaises(runner.ResourceLimit):
            self.measure()
        self.assertEqual(len(self.result["measurements"]["xgb"]["units"]), 9)
        self.assertLess(len(self.result["measurements"]["lgbm"]["units"]), 9)
        self.assertIsNone(self.result["trial_seconds_estimates"])

    def test_invalid_predictions_cannot_complete_unit(self):
        self.fake.invalid_predictions = True
        with self.assertRaisesRegex(ValueError, "Invalid prediction"):
            self.measure()
        self.assertEqual(self.result["measurements"]["xgb"]["units"], [])
        self.assertIsNone(self.result["trial_seconds_estimates"])

    def test_wrong_dataset_rejected_before_fit(self):
        self.fake.wrong_manifest = True
        with self.assertRaisesRegex(ValueError, "development data differs"):
            self.measure()
        self.assertEqual(self.fake.calls, [])

    def execute_fixture(self, temperature, name, configure=None):
        base = ROOT / ".local-tools"
        with tempfile.TemporaryDirectory(
            prefix="smoke-runner-test-", dir=base
        ) as directory:
            temp = Path(directory).resolve()
            self.assertTrue(temp.is_relative_to(base.resolve()))
            (temp / "runs").mkdir()
            sensor = temp / "synthetic-test-sensor.json"
            sensor.write_text(
                json.dumps(
                    {
                        "timestamp": time.time(),
                        "cpu_c": temperature,
                        "fixture": "synthetic test only",
                    }
                )
            )
            run = temp / "runs" / name
            if configure:
                configure(run)
            with (
                patch.object(runner, "ROOT", temp),
                patch.object(
                    runner,
                    "frozen_inputs",
                    return_value=(self.protocol, self.settings, self.splits),
                ),
                patch.object(
                    runner, "backend", return_value=self.fake.operations()
                ) as factory,
            ):
                result = runner.execute(run, sensor)
                return result, factory.call_count

    def test_hot_preflight_never_loads_backend(self):
        result, calls = self.execute_fixture(85, "smoke-hot")
        self.assertEqual(result["status"], "STOP")
        self.assertEqual(calls, 0)
        self.assertIsNone(result["trial_seconds_estimates"])

    def test_failed_learner_keeps_partial_evidence_without_estimate(self):
        self.fake.fail_lgbm = True
        result, calls = self.execute_fixture(40, "smoke-failed")
        self.assertEqual(calls, 1)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(len(result["measurements"]["xgb"]["units"]), 9)
        self.assertIsNone(result["trial_seconds_estimates"])

    def test_existing_run_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            self.execute_fixture(40, "smoke-existing", configure=lambda p: p.mkdir())

    def test_non_smoke_run_is_rejected(self):
        with self.assertRaises(ValueError):
            self.execute_fixture(40, "final-run")


if __name__ == "__main__":
    unittest.main(verbosity=2)
