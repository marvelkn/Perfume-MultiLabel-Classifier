"""Watchdog integration tests use sleeping subprocesses and synthetic telemetry."""
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("supervised", DIRECTORY/"run_supervised_smoke.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakeSensor:
    def __init__(self, directory, values, stale=False, fail_after=None):
        self.output = directory/"SYNTHETIC_TEST_ONLY.json"
        self.values = iter(values)
        self.value = 40
        self.stale = stale
        self.fail_after = fail_after
        self.calls = 0

    def tick(self):
        self.calls += 1
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise ValueError("Synthetic sensor failure")
        self.value = next(self.values, self.value)
        self.output.write_text(json.dumps({"cpu_c": self.value,
            "timestamp": time.time()-(20 if self.stale else 0),
            "fixture": "SYNTHETIC TEST ONLY"}))
        return {"cpu_c": self.value, "observed_timestamp": time.time()}


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        base = runner.ROOT/".local-tools"
        self.temp = tempfile.TemporaryDirectory(prefix="supervisor-test-", dir=base)
        self.directory = Path(self.temp.name).resolve()
        self.assertTrue(self.directory.is_relative_to(base.resolve()))
        self.release = self.directory/"release"
        self.settings = {"profile": "synthetic", "threads": 1,
            "session_minutes": 1, "max_process_ram_gb": 3,
            "min_available_ram_gb": 4, "require_temperature": True,
            "temperature_max_age_seconds": 5, "start_temperature_c": 80,
            "stop_temperature_c": 80, "hard_temperature_c": 90,
            "hot_seconds": 5, "check_interval_seconds": 0.25}
        self.command = [sys.executable, "-c", "import time; time.sleep(60)"]

    def tearDown(self):
        self.temp.cleanup()

    def supervise(self, sensor, command=None):
        return runner.supervised_process(command or self.command, self.directory,
            sensor, self.settings, self.release)

    def test_hot_preflight_starts_no_process(self):
        with patch.object(runner.subprocess, "Popen") as launch:
            result = self.supervise(FakeSensor(self.directory, [80]))
        launch.assert_not_called()
        self.assertEqual(result["status"], "STOP")
        self.assertFalse(result["child_started"])

    def test_stale_sensor_starts_no_process(self):
        with patch.object(runner.subprocess, "Popen") as launch:
            result = self.supervise(FakeSensor(self.directory, [40], stale=True))
        launch.assert_not_called()
        self.assertEqual(result["status"], "STOP")

    def test_temperature_rise_kills_sleeping_worker(self):
        result = self.supervise(FakeSensor(self.directory, [40, 40, 81]))
        self.assertTrue(result["child_started"])
        self.assertEqual(result["status"], "STOP")
        self.assertIn("80 C", result["reason"])
        self.assertIsNotNone(result["child_exit_code"])
        self.assertLess(result["active_seconds"], 5)

    def test_sensor_failure_kills_sleeping_worker(self):
        result = self.supervise(FakeSensor(self.directory, [40], fail_after=3))
        self.assertTrue(result["child_started"])
        self.assertEqual(result["status"], "STOP")
        self.assertIn("sensor failure", result["reason"])
        self.assertIsNotNone(result["child_exit_code"])

    def test_timeout_kills_sleeping_worker(self):
        self.settings["session_minutes"] = 0.003
        result = self.supervise(FakeSensor(self.directory, [40]))
        self.assertTrue(result["child_started"])
        self.assertEqual(result["status"], "STOP")
        self.assertIn("time budget", result["reason"])
        self.assertLess(result["active_seconds"], 5)

    def test_successful_child_finishes_cleanly(self):
        result = self.supervise(FakeSensor(self.directory, [40]),
                                [sys.executable, "-c", "print('synthetic finished')"])
        self.assertEqual(result["status"], "WORKER_COMPLETED")
        self.assertEqual(result["child_exit_code"], 0)

    def test_assignment_failure_never_releases_worker(self):
        with patch.object(runner, "KillJob", side_effect=RuntimeError("job unavailable")):
            result = self.supervise(FakeSensor(self.directory, [40]))
        self.assertEqual(result["status"], "STOP")
        self.assertFalse(self.release.exists())
        self.assertIsNotNone(result["child_exit_code"])

    def test_direct_worker_rejects_unbounded_budget(self):
        with self.assertRaisesRegex(ValueError, "Invalid worker"):
            runner.worker(self.directory, "xgb", 301)



    def execute_synthetic_stages(self, times):
        import io
        from contextlib import redirect_stdout
        (self.directory/"runs").mkdir()
        run = self.directory/"runs/smoke-supervised-synthetic"
        received_budgets = []
        def fake_stage(command, directory, sensor, settings, release_file):
            algorithm = directory.name
            received_budgets.append(settings["session_minutes"]*60)
            (directory/"worker-result.json").write_text(json.dumps({
                "status": "STAGE_COMPLETED",
                "trial_seconds_estimates": {algorithm: 100}}))
            return {"status": "WORKER_COMPLETED", "child_started": True,
                    "active_seconds": times[algorithm]}
        with (patch.object(runner, "ROOT", self.directory),
              patch.object(runner, "inputs", return_value=(
                  {"protocol_id": "SYNTHETIC"}, self.settings, {})),
              patch.object(runner, "Sensor"),
              patch.object(runner, "supervised_process", side_effect=fake_stage),
              patch.object(runner, "cooldown") as cooling,
              redirect_stdout(io.StringIO())):
            code = runner.execute(run, self.directory/"SYNTHETIC.csv")
        return code, json.loads((run/"supervised-smoke-result.json").read_text()), received_budgets, cooling.call_count

    def test_cooldown_does_not_reset_combined_budget(self):
        code, result, budgets, _ = self.execute_synthetic_stages({"xgb": 180, "lgbm": 130})
        self.assertEqual(budgets, [300, 120])
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "STOP")
        self.assertIsNone(result["trial_seconds_estimates"])

    def test_both_stages_and_cooldowns_required_for_success(self):
        code, result, budgets, cools = self.execute_synthetic_stages({"xgb": 50, "lgbm": 70})
        self.assertEqual(len(budgets), 2)
        self.assertAlmostEqual(budgets[0], 300)
        self.assertAlmostEqual(budgets[1], 250)
        self.assertEqual(code, 0)
        self.assertEqual(cools, 2)
        self.assertEqual(result["active_seconds"], 120)
        self.assertEqual(result["trial_seconds_estimates"], {"xgb": 100, "lgbm": 100})


if __name__ == "__main__":
    unittest.main(verbosity=2)

