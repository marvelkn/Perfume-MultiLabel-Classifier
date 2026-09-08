import json
from types import SimpleNamespace

import pytest

from src import experiments, runtime
from src.config import PROJECT_ROOT, load_config
from src.runtime import ResourceGuard, ResourceLimit, validate_training_resources

SETTINGS = load_config(PROJECT_ROOT / "config.safe-smoke.yaml")["resources"]


@pytest.fixture
def machine(monkeypatch):
    now = [0.]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(runtime.time, "time", lambda: 1700000000 + now[0])
    monkeypatch.setattr(runtime.psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 1024**3))
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100),
                              children=lambda **kw: [])
    monkeypatch.setattr(runtime.psutil, "Process", lambda: process)
    return now


def sensor(tmp_path, temperature=40, timestamp=1700000000):
    path = tmp_path / "cpu.json"
    path.write_text(json.dumps({"timestamp": timestamp, "cpu_c": temperature}))
    return path


def test_missing_temperature_blocks_real_training(machine, monkeypatch):
    guard = ResourceGuard(SETTINGS)
    monkeypatch.setattr(guard, "temperature", lambda: None)
    with pytest.raises(ResourceLimit, match="unavailable"):
        guard.check()


def test_sustained_heat_and_cool_reset(machine, monkeypatch):
    guard = ResourceGuard(SETTINGS)
    temp = [40]
    monkeypatch.setattr(guard, "temperature", lambda: temp[0])
    guard.check()
    temp[0] = 81; machine[0] = 1; guard.check()
    machine[0] = 5; guard.check()
    temp[0] = 79; machine[0] = 6; guard.check()
    temp[0] = 81; machine[0] = 7; guard.check()
    machine[0] = 12
    with pytest.raises(ResourceLimit, match="Sustained"):
        guard.check()
    temp[0] = 40
    with pytest.raises(ResourceLimit, match="Sustained"):
        guard.check(force=True)


def test_memory_and_time_limits(machine, monkeypatch):
    guard = ResourceGuard({**SETTINGS, "require_temperature": False})
    monkeypatch.setattr(runtime.psutil, "virtual_memory", lambda: SimpleNamespace(available=100))
    with pytest.raises(ResourceLimit, match="RAM"):
        guard.check()
    fresh = ResourceGuard(SETTINGS)
    machine[0] = 300
    with pytest.raises(ResourceLimit, match="time budget"):
        fresh.check()


def test_stale_sensor_cannot_bypass_guard(machine, tmp_path):
    with pytest.raises(ResourceLimit, match="stale"):
        ResourceGuard(SETTINGS, sensor(tmp_path, timestamp=1)).check()


@pytest.mark.parametrize("payload", [
    None, [], {}, {"timestamp":1700000000, "cpu_c":None},
    {"timestamp":1700000000, "cpu_c":float("nan")},
    {"timestamp":1700000000, "cpu_c":True},
    {"timestamp":float("inf"), "cpu_c":40},
    {"timestamp":1700000003, "cpu_c":40},
    {"timestamp":1700000000, "cpu_c":40, "status":"STOP"},
])
def test_invalid_payload_is_resource_stop(machine, tmp_path, payload):
    path = tmp_path / "cpu.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ResourceLimit):
        ResourceGuard(SETTINGS, path).check()


@pytest.mark.parametrize("temperature,match", [(75, "too hot to start"), (90, "Hard")])
def test_hot_preflight_rejected_immediately(machine, tmp_path, temperature, match):
    with pytest.raises(ResourceLimit, match=match):
        ResourceGuard(SETTINGS, sensor(tmp_path, temperature)).check()


def test_process_tree_memory_is_counted(machine, monkeypatch, tmp_path):
    child = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=4 * 1024**3))
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100),
                              children=lambda **kw: [child])
    monkeypatch.setattr(runtime.psutil, "Process", lambda: process)
    with pytest.raises(ResourceLimit, match="memory budget"):
        ResourceGuard(SETTINGS, sensor(tmp_path)).check()


def test_session_logs_measurements_and_timeout(machine, tmp_path):
    log = tmp_path / "session.jsonl"
    with pytest.raises(ResourceLimit, match="time budget"), ResourceGuard(SETTINGS, sensor(tmp_path), log_path=log) as guard:
        machine[0] = 300
        guard.check()
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["event"] for r in events] == ["start", "sample", "stop", "end"]
    assert events[1]["cpu_c"] == 40 and events[1]["process_tree_rss_bytes"] == 100
    assert events[-1]["status"] == "stopped" and events[-1]["elapsed_seconds"] == 300


@pytest.mark.parametrize("exception,status", [(None, "completed"), (KeyboardInterrupt, "interrupted"),
                                               (ValueError, "failed")])
def test_session_end_records_normal_interrupt_and_error(machine, tmp_path, exception, status):
    log = tmp_path / "session.jsonl"
    try:
        with ResourceGuard(SETTINGS, sensor(tmp_path), log_path=log) as guard:
            if exception:
                raise exception("simulated")
    except (KeyboardInterrupt, ValueError):
        pass
    final = json.loads(log.read_text().splitlines()[-1])
    assert final["status"] == status
    with pytest.raises(ResourceLimit, match="closed"):
        guard.check()


def test_unwritable_log_fails_closed(machine, tmp_path):
    log = tmp_path / "existing.jsonl"
    log.write_text("existing evidence")
    with pytest.raises(ResourceLimit, match="session log"), ResourceGuard(SETTINGS, sensor(tmp_path), log_path=log):
        pytest.fail("Must not enter the workload")
    assert log.read_text() == "existing evidence"


@pytest.mark.parametrize("operation", ["baseline", "tune", "fit", "explain"])
def test_hot_session_stops_before_dataset_loading(machine, tmp_path, monkeypatch, operation):
    (tmp_path / "run.json").write_text(json.dumps({"resources": SETTINGS}))
    monkeypatch.setattr(experiments, "context", lambda run: pytest.fail("Dataset must not load"))
    with pytest.raises(ResourceLimit, match="too hot"), experiments.guarded_context(tmp_path, operation, "xgb", sensor(tmp_path, 80)):
        pytest.fail("Workload must not start")
    events = [json.loads(line) for line in next((tmp_path / "sessions").glob("*.jsonl")).read_text().splitlines()]
    assert events[-1]["event"] == "stop"


@pytest.mark.parametrize("profile", ["safe-smoke", "safe-training"])
def test_profiles_preserve_dataset_and_methodology(profile):
    base = load_config(PROJECT_ROOT / "config.yaml")
    selected = load_config(PROJECT_ROOT / f"config.{profile}.yaml")
    resources = selected.pop("resources")
    base.pop("resources")
    assert selected == base
    validate_training_resources(resources)


@pytest.mark.parametrize("key,value", [("require_temperature",False),("threads",3),
    ("session_minutes",float("nan")),("temperature_max_age_seconds",11),
    ("check_interval_seconds",2),("start_temperature_c",80),("max_process_ram_gb",5)])
def test_unsafe_operational_settings_are_rejected(key, value):
    with pytest.raises(ValueError):
        validate_training_resources({**SETTINGS, key:value})


def test_run_lock_blocks_second_writer_and_releases_on_error(tmp_path):
    from src.runtime import exclusive_run
    with pytest.raises(ValueError), exclusive_run(tmp_path):
        with pytest.raises(ResourceLimit, match="locked"), exclusive_run(tmp_path):
            pytest.fail("Second writer entered")
        assert (tmp_path / ".resource-session.lock").exists()
        raise ValueError("simulated interrupt")
    assert not (tmp_path / ".resource-session.lock").exists()


def test_invalid_log_write_stops_workload(machine, tmp_path):
    class BrokenLog:
        def write(self, text):
            raise OSError("simulated disk full")
    guard = ResourceGuard(SETTINGS, sensor(tmp_path))
    guard._log = BrokenLog()
    with pytest.raises(ResourceLimit, match="persist"):
        guard.check()
    assert guard.stop_reason is not None


def test_hard_limit_has_no_grace_period(machine, tmp_path, monkeypatch):
    guard = ResourceGuard(SETTINGS)
    temp = [40]
    monkeypatch.setattr(guard, "temperature", lambda: temp[0])
    guard.check()
    temp[0] = 90; machine[0] = 1
    with pytest.raises(ResourceLimit, match="Hard"):
        guard.check()


def test_legacy_default_cannot_initialize_an_operational_run(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "CONFIG", load_config(PROJECT_ROOT / "config.yaml"))
    monkeypatch.setattr(experiments, "load_dataset", lambda *a: pytest.fail("Must reject before loading data"))
    with pytest.raises(ValueError, match="explicit safe profile"):
        experiments.initialize(tmp_path / "dataset", tmp_path / "run")


def test_session_checks_after_last_workload_block(machine, tmp_path, monkeypatch):
    (tmp_path / "run.json").write_text(json.dumps({"resources": SETTINGS}))
    monkeypatch.setattr(experiments, "context", lambda run: (run, {}, None, None, {}))
    with pytest.raises(ResourceLimit, match="time budget"), experiments.guarded_context(
        tmp_path, "fit", "xgb", sensor(tmp_path)
    ):
        machine[0] = 300
    events = [json.loads(line) for line in next((tmp_path / "sessions").glob("*.jsonl")).read_text().splitlines()]
    assert events[-1]["status"] == "stopped"
    assert not (tmp_path / ".resource-session.lock").exists()


@pytest.mark.parametrize("algorithm", ["xgb", "lgbm"])
def test_boosting_callback_propagates_stop(algorithm):
    import numpy as np
    class StopDuringBoosting:
        def __init__(self):
            self.callback_checks = 0
        def check(self, force=False):
            if not force:
                self.callback_checks += 1
                raise ResourceLimit("synthetic callback stop")
    guard = StopDuringBoosting()
    X = np.random.default_rng(42).random((32, 4)).astype(np.float32)
    y = np.tile([0, 1], 16)
    with pytest.raises(ResourceLimit, match="synthetic callback stop"):
        experiments.fit_binary(algorithm, {"max_depth": 2}, X, y,
                               threads=1, guard=guard, rounds=4)
    assert guard.callback_checks == 1
