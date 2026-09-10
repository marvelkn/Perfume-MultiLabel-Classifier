"""Supervised CPU smoke with an explicit, versioned start-margin amendment."""
import argparse
import copy
import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DIRECTORY.parent))
import run_smoke_probe as original

from src.cpu_telemetry import LhmCsvReader, write_json_atomic
from src.runtime import ResourceGuard, ResourceLimit


class ImmediateGuard(ResourceGuard):
    def check(self, force=False):
        super().check(force=force)
        if self.snapshot.get("cpu_c") is not None and self.snapshot["cpu_c"] >= 80:
            self._stop("Supervisor: observed CPU temperature reached 80 C")


class KillJob:
    """Closing this Windows job kills its worker, even if the supervisor exits."""
    def __init__(self, process):
        if os.name != "nt":
            raise RuntimeError("This supervisor requires Windows Job Objects")
        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]
        class Io(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("ReadOperationCount", "WriteOperationCount",
                         "OtherOperationCount", "ReadTransferCount",
                         "WriteTransferCount", "OtherTransferCount")]
        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", Io),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            info = Extended()
            info.BasicLimitInformation.LimitFlags = 0x2000
            if not self.kernel.SetInformationJobObject(
                    self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.kernel.AssignProcessToJobObject(
                    self.handle, wintypes.HANDLE(int(process._handle))):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.handle:
            handle, self.handle = self.handle, None
            if not self.kernel.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())


class Sensor:
    def __init__(self, csv_path, run):
        self.reader = LhmCsvReader(csv_path, "/amdcpu/0/temperature/2")
        self.output = run / "cpu-live.json"
        self.stream = (run / "sensor-history.jsonl").open("x", encoding="utf-8")
        self.last_stamp = None
        self.last_value = None
        self.changed_at = time.monotonic()

    def tick(self):
        reading = self.reader.read()
        if self.last_stamp is not None and reading.timestamp < self.last_stamp:
            raise ResourceLimit("CPU acquisition timestamp moved backwards")
        if reading.cpu_c != self.last_value:
            self.changed_at = time.monotonic()
        elif time.monotonic() - self.changed_at >= 60:
            raise ResourceLimit("CPU value unchanged for 60 seconds")
        self.last_stamp, self.last_value = reading.timestamp, reading.cpu_c
        payload = vars(reading)
        write_json_atomic(self.output, payload)
        now = time.time()
        self.stream.write(json.dumps({**payload, "observed_timestamp": now,
                                      "age_seconds": now-reading.timestamp}) + "\n")
        self.stream.flush()
        return {**payload, "observed_timestamp": now}

    def close(self):
        try:
            write_json_atomic(self.output, {"timestamp": 0, "cpu_c": 0, "status": "STOP"})
        finally:
            self.stream.close()


def supervised_process(command, directory, sensor, settings, release_file):
    """Only release a worker after successful preflight and Job Object assignment."""
    result = {"status": "STARTING", "child_started": False}
    process = job = None
    started = time.perf_counter()
    guard = ImmediateGuard(settings, sensor.output,
                           log_path=directory / "supervisor-resources.jsonl",
                           metadata={"role": "independent parent", "poll_seconds": 0.25})
    try:
        sensor.tick()
        with guard, (directory / "worker-console.txt").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            result.update(child_started=True, child_pid=process.pid)
            job = KillJob(process)
            release_file.write_text("JOB_ATTACHED\n", encoding="utf-8")
            while process.poll() is None:
                sensor.tick()
                guard.check(force=True)
                time.sleep(0.25)
            sensor.tick()
            guard.check(force=True)
            if process.returncode != 0:
                raise RuntimeError(f"Worker exited with code {process.returncode}")
        result["status"] = "WORKER_COMPLETED"
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- Stop and retain evidence on every failure.
        result.update(status="STOP", reason=str(exc) or "User interrupted",
                      stop_observed_timestamp=time.time(), stop_snapshot=guard.snapshot)
    finally:
        # Child fitting cannot survive supervisor failure or cancellation.
        try:
            if job:
                job.close()
        finally:
            if process:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                result["child_exit_code"] = process.returncode
                result["child_exited_timestamp"] = time.time()
        result["active_seconds"] = time.perf_counter()-started
        write_json_atomic(directory / "supervisor-result.json", result)
    return result


def inputs():
    protocol, settings, splits = original.frozen_inputs()
    amendment = json.loads((DIRECTORY / "amendment.json").read_text())
    import runpy
    checker = runpy.run_path(str(original.PROTOCOL / "verify_protocol.py"))
    checker["validate_files"](ROOT, amendment["files"])
    if checker["digest"](original.PROTOCOL / "freeze_manifest.json") != amendment["parent_freeze_sha256"]:
        raise ValueError("Parent protocol hash differs")
    if amendment["status"] != "FROZEN_BEFORE_EXECUTION":
        raise ValueError("Amendment is not frozen")
    # An explicit profile, not a silent edit of safe-smoke or its validator.
    settings = {**settings, "profile": "supervised-smoke-v2",
                "start_temperature_c": 80, "temperature_max_age_seconds": 5}
    return protocol, settings, splits


def worker(run, algorithm, seconds):
    run = Path(run).resolve()
    if (run.parent.parent != ROOT/"runs" or not run.parent.name.startswith("smoke-supervised-")
            or run.name != algorithm or seconds is None or not 0 < seconds <= 300):
        raise ValueError("Invalid worker stage or time budget")
    release = run / "worker-release"
    deadline = time.monotonic()+10
    while not release.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("Supervisor did not attach the worker job")
        time.sleep(0.05)
    protocol, settings, splits = inputs()
    settings["session_minutes"] = seconds/60
    stage_protocol = copy.deepcopy(protocol)
    stage_protocol["smoke_measurement"]["learner_order"] = [algorithm]
    result = {"status": "STARTING", "algorithm": algorithm,
              "trial_seconds_estimates": None, "active_unit": None}
    guard = ImmediateGuard(settings, run.parent/"cpu-live.json",
                           log_path=run/"worker-resources.jsonl",
                           metadata={"role": "cooperative worker", "algorithm": algorithm})
    try:
        with guard:
            started = time.perf_counter()
            operations = original.backend()
            guard.check(force=True)
            result["backend_import_seconds"] = time.perf_counter()-started
            with operations["pool"](limits=1), (run/"units.jsonl").open("x", encoding="utf-8") as events:
                original.measure(stage_protocol, splits, guard, operations, result, events)
            guard.check(force=True)
        result["status"] = "STAGE_COMPLETED"
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- Stop and retain evidence on every failure.
        result.update(status="STOP", reason=str(exc) or "Interrupted",
                      trial_seconds_estimates=None)
    finally:
        write_json_atomic(run/"worker-result.json", result)
    return 0 if result["status"] == "STAGE_COMPLETED" else 2


def cooldown(sensor, path, seconds=610):
    started = time.monotonic()
    observations = []
    while time.monotonic()-started < seconds:
        observations.append(sensor.tick())
        time.sleep(1)
    last = observations[-1]["observed_timestamp"]
    window = [r for r in observations if r["observed_timestamp"] >= last-300]
    result = {"duration_seconds": time.monotonic()-started,
              "end_temperature_c": observations[-1]["cpu_c"],
              "last_300_seconds_max_c": max(r["cpu_c"] for r in window),
              "status": "GO" if all(r["cpu_c"] < 75 for r in window) else "STOP"}
    write_json_atomic(path, result)
    if result["status"] != "GO":
        raise ResourceLimit("Cooldown did not reach stable temperature below 75 C")


def execute(run, csv_path):
    protocol, settings, _ = inputs()
    run = Path(run).resolve()
    if run.parent != ROOT/"runs" or not run.name.startswith("smoke-supervised-"):
        raise ValueError("Use a new direct child runs/smoke-supervised-*")
    run.mkdir(exist_ok=False)
    result = {"status": "STARTING", "protocol_id": protocol["protocol_id"],
              "resource_amendment": "essenza-smoke-supervised-v2",
              "started_at": datetime.now().astimezone().isoformat(),
              "active_seconds": 0, "stages": {}, "trial_seconds_estimates": None,
              "model_metrics_computed": False, "test_arrays_loaded": False,
              "final_trial_count": None}
    sensor = None
    try:
        sensor = Sensor(csv_path, run)
        for algorithm in ("xgb", "lgbm"):
            remaining = 300-result["active_seconds"]
            if remaining <= 0:
                raise ResourceLimit("Combined 300-second smoke budget exhausted")
            stage = run/algorithm
            stage.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()), "worker",
                       "--run", str(stage), "--algorithm", algorithm,
                       "--seconds", str(remaining)]
            stage_settings = {**settings, "session_minutes": remaining/60}
            observed = supervised_process(command, stage, sensor, stage_settings,
                                          stage/"worker-release")
            result["active_seconds"] += observed["active_seconds"]
            result["stages"][algorithm] = observed
            write_json_atomic(run/"supervised-smoke-result.json", result)
            if observed["status"] != "WORKER_COMPLETED":
                raise ResourceLimit(observed.get("reason", "Worker incomplete"))
            stage_result = json.loads((stage/"worker-result.json").read_text())
            if stage_result["status"] != "STAGE_COMPLETED":
                raise ResourceLimit("Worker measurements incomplete")
            result["stages"][algorithm]["measurement"] = stage_result
            print(f"{algorithm}: complete; cooldown started", flush=True)
            cooldown(sensor, stage/"cooldown.json")
        if result["active_seconds"] > 300:
            raise ResourceLimit("Combined active budget exceeded")
        result["trial_seconds_estimates"] = {
            a: result["stages"][a]["measurement"]["trial_seconds_estimates"][a]
            for a in ("xgb", "lgbm")}
        result["status"] = "SMOKE_COMPLETE_AND_COOLED_PENDING_REVIEW"
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- Stop and retain evidence on every failure.
        result.update(status="STOP", reason=str(exc) or "User interrupted",
                      trial_seconds_estimates=None)
    finally:
        if sensor:
            try:
                if (result["status"] == "STOP" and
                        any(stage.get("child_started") for stage in result["stages"].values())):
                    print("Worker stopped; observing post-stop cooldown", flush=True)
                    cooldown(sensor, run/"post-stop-cooldown.json")
            except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- Stop and retain evidence on every failure.
                result["cooldown_issue"] = str(exc) or "Interrupted"
            finally:
                sensor.close()
        result["ended_at"] = datetime.now().astimezone().isoformat()
        write_json_atomic(run/"supervised-smoke-result.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "SMOKE_COMPLETE_AND_COOLED_PENDING_REVIEW" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "run", "worker"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--algorithm", choices=["xgb", "lgbm"])
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    if args.command == "validate":
        inputs()
        print("VALIDATED_ONLY: no worker or model started")
        return 0
    if args.command == "worker":
        return worker(args.run, args.algorithm, args.seconds)
    return execute(args.run, args.csv)


if __name__ == "__main__":
    raise SystemExit(main())

