"""Synthetic fixtures test the adapter only; they never qualify a live sensor."""
import json
from datetime import datetime

import pytest

from src.cpu_telemetry import LhmCsvReader, TelemetryError, main, write_json_atomic
from src.runtime import ResourceGuard

CPU_ID = "/amdcpu/0/temperature/0"


def log_file(tmp_path, *, value="42.5", identifier=CPU_ID, name="Core (Tctl/Tdie)"):
    path = tmp_path / "fixture.csv"
    path.write_text(f',/gpu-amd/0/temperature/0,{identifier}\nTime,"GPU Core","{name}"\n'
                    f"09/07/2026 12:00:00,70,{value}\n", encoding="utf-8")
    return path


def test_cpu_selection_preserves_acquisition_time_for_guard(tmp_path, monkeypatch):
    timestamp = datetime(2026, 9, 7, 12).astimezone().timestamp()
    reading = LhmCsvReader(log_file(tmp_path), CPU_ID).read(now=timestamp + 2)
    assert reading.cpu_c == 42.5
    assert reading.timestamp == timestamp
    output = tmp_path / "cpu.json"
    write_json_atomic(output, vars(reading))
    monkeypatch.setattr("src.runtime.time.time", lambda: timestamp + 2)
    assert ResourceGuard({}, output).temperature() == 42.5
    assert json.loads(output.read_text())["timestamp"] != timestamp + 2


@pytest.mark.parametrize("offset", [6, -3])
def test_stale_and_future_samples_rejected(tmp_path, offset):
    with pytest.raises(TelemetryError, match="stale|future"):
        LhmCsvReader(log_file(tmp_path), CPU_ID).read(
            now=datetime(2026, 9, 7, 12).astimezone().timestamp() + offset)


@pytest.mark.parametrize("value", ["", "nan", "inf", "0", "150"])
def test_unavailable_or_invalid_temperature_rejected(tmp_path, value):
    with pytest.raises(TelemetryError, match="invalid|implausible"):
        LhmCsvReader(log_file(tmp_path, value=value), CPU_ID).read(
            now=datetime(2026, 9, 7, 12).astimezone().timestamp())


def test_gpu_and_non_package_sensor_rejected(tmp_path):
    path = log_file(tmp_path)
    with pytest.raises(TelemetryError, match="CPU temperature"):
        LhmCsvReader(path, "/gpu-amd/0/temperature/0")
    with pytest.raises(TelemetryError, match="package"):
        LhmCsvReader(log_file(tmp_path, name="Unverified sensor"), CPU_ID)


def test_partial_append_does_not_refresh_previous_sample(tmp_path):
    path = log_file(tmp_path)
    with path.open("a") as stream:
        stream.write("09/07/2026 12:00:02,70,")
    reader = LhmCsvReader(path, CPU_ID)
    timestamp = datetime(2026, 9, 7, 12).astimezone().timestamp()
    assert reader.read(now=timestamp + 2).timestamp == timestamp
    with pytest.raises(TelemetryError, match="stale"):
        reader.read(now=timestamp + 6)


def test_changed_sensor_headers_rejected(tmp_path):
    path = log_file(tmp_path)
    reader = LhmCsvReader(path, CPU_ID)
    log_file(tmp_path, name="Wrong sensor")
    with pytest.raises(TelemetryError, match="headers changed"):
        reader.read()


def test_missing_source_creates_no_fake_live_telemetry(tmp_path):
    output = tmp_path / "cpu.json"
    assert main(["--log", str(tmp_path / "missing.csv"), "--sensor-id", CPU_ID,
                 "--output", str(output), "--history", str(tmp_path / "attempt.jsonl")]) == 1
    assert not output.exists()


def test_collector_failure_invalidates_live_file(tmp_path):
    from src.cpu_telemetry import collect

    class FailedSensor:
        sensor_id = CPU_ID
        sensor_name = "Core (Tctl/Tdie)"
        path = tmp_path / "fixture.csv"

        def read(self):
            raise TelemetryError("Sensor disconnected")

    output = tmp_path / "cpu.json"
    history = tmp_path / "attempt.jsonl"
    with pytest.raises(TelemetryError, match="disconnected"):
        collect(FailedSensor(), output, history, 1)
    assert json.loads(output.read_text())["timestamp"] == 0
    from src.runtime import ResourceLimit
    with pytest.raises(ResourceLimit):
        ResourceGuard({}, output).temperature()
    assert json.loads(history.with_suffix(".summary.json").read_text())["status"] == "STOP"
