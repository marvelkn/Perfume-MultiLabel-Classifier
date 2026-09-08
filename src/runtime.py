"""Cooperative resource limits and append-only session evidence."""
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

PROFILE_LIMITS = {
    "safe-smoke": {"threads": 1, "session_minutes": 5, "max_process_ram_gb": 4,
                   "stop_temperature_c": 80, "hot_seconds": 5},
    "safe-training": {"threads": 2, "session_minutes": 20, "max_process_ram_gb": 4,
                      "stop_temperature_c": 85, "hot_seconds": 10},
}


class ResourceLimit(RuntimeError):
    pass


def _number(settings, key, default=None):
    value = settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid positive resource setting: {key}")
    return value


def validate_training_resources(settings):
    """Operational commands require an explicit, bounded profile."""
    profile = settings.get("profile")
    if profile not in PROFILE_LIMITS or settings.get("require_temperature") is not True:
        raise ValueError("Choose an explicit safe profile with require_temperature=true for a new run")
    for key, ceiling in PROFILE_LIMITS[profile].items():
        if _number(settings, key) > ceiling:
            raise ValueError(f"{key} exceeds the {profile} limit")
    if not isinstance(settings["threads"], int):
        raise TypeError("threads must be an integer")
    if _number(settings, "min_available_ram_gb") < 4:
        raise ValueError("Keep at least 4 GiB system RAM available")
    if _number(settings, "temperature_max_age_seconds", 10) > 10:
        raise ValueError("Telemetry age limit cannot exceed 10 seconds")
    if _number(settings, "check_interval_seconds", 1) > 1:
        raise ValueError("Guard check interval cannot exceed 1 second")
    if _number(settings, "start_temperature_c") > settings["stop_temperature_c"] - 5:
        raise ValueError("Start temperature must leave at least 5 C below the stop threshold")
    hard = _number(settings, "hard_temperature_c", 90)
    if not settings["stop_temperature_c"] <= hard <= 90:
        raise ValueError("Hard temperature limit must lie between the profile stop and 90 C")


@contextmanager
def exclusive_run(run):
    """A hard-killed session leaves a lock for explicit, inspected recovery."""
    lock = Path(run) / ".resource-session.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ResourceLimit("Run session is locked; inspect its process before recovering a stale lock") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created_timestamp": time.time()}, stream)
        yield
    finally:
        lock.unlink()


class ResourceGuard:
    def __init__(self, settings, temperature_file=None, *, log_path=None, metadata=None):
        self.settings = dict(settings)
        self.temperature_file = Path(temperature_file) if temperature_file else None
        self.started = time.monotonic()
        self.hot_since = None
        self.last_check = -float("inf")
        self.stop_reason = None
        self.closed = False
        self.preflight_passed = False
        self.sensor_age_seconds = None
        self.snapshot = {}
        self.log_path = Path(log_path) if log_path else None
        self.metadata = dict(metadata or {})
        self._log = None

    def _emit(self, event, **details):
        if self._log is None:
            return
        record = {**self.snapshot, "timestamp": time.time(),
                  "elapsed_seconds": time.monotonic() - self.started,
                  "event": event, **details}
        try:
            self._log.write(json.dumps(record, allow_nan=False) + "\n")
            self._log.flush()
        except (OSError, ValueError) as exc:
            self.stop_reason = "Cannot persist resource session log"
            raise ResourceLimit(self.stop_reason) from exc

    def _stop(self, reason):
        self.stop_reason = reason
        self._emit("stop", reason=reason)
        raise ResourceLimit(reason)

    def __enter__(self):
        try:
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    self._log = self.log_path.open("x", encoding="utf-8")
                except OSError as exc:
                    raise ResourceLimit("Cannot open resource session log") from exc
            self._emit("start", resources=self.settings, metadata=self.metadata,
                       temperature_file=str(self.temperature_file) if self.temperature_file else None)
            self.check(force=True)
            return self
        except BaseException:
            if self._log:
                self._log.close()
            self.closed = True
            raise

    def __exit__(self, exc_type, exc, traceback):
        try:
            status = "completed" if exc is None else "stopped" if isinstance(exc, ResourceLimit) else "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            self._emit("end", status=status, reason=str(exc) if exc is not None else None)
        finally:
            self.closed = True
            if self._log:
                self._log.close()
        return False

    def temperature(self):
        self.sensor_age_seconds = None
        if self.temperature_file:
            try:
                data = json.loads(self.temperature_file.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or data.get("status", "LIVE") != "LIVE":
                    raise ValueError("Telemetry is not live")
                stamp, value = data["timestamp"], data["cpu_c"]
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (stamp, value)):
                    raise ValueError("Telemetry values must be numeric")
                age = time.time() - stamp
                age_limit = min(_number(self.settings, "temperature_max_age_seconds", 10), 10)
                if not math.isfinite(age) or not -2 <= age <= age_limit or not math.isfinite(value) or not 0 < value < 150:
                    raise ValueError("Telemetry stale or invalid")
                self.sensor_age_seconds = age
                return float(value)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise ResourceLimit("CPU sensor telemetry is stale, invalid, or unavailable") from exc
        reader = getattr(psutil, "sensors_temperatures", None)
        if reader:
            values = [entry.current for name, entries in reader().items()
                      if name.lower() in {"coretemp", "k10temp", "cpu_thermal"}
                      for entry in entries if entry.current is not None]
            if values:
                if not all(math.isfinite(v) and 0 < v < 150 for v in values):
                    raise ResourceLimit("CPU sensor telemetry is invalid")
                return max(values)
        return None

    def check(self, force=False):
        if self.closed or self.stop_reason:
            raise ResourceLimit(self.stop_reason or "Resource session is closed")
        now = time.monotonic()
        try:
            interval = min(_number(self.settings, "check_interval_seconds", 1), 1)
            minutes = _number(self.settings, "session_minutes")
            max_rss = _number(self.settings, "max_process_ram_gb")
            min_available = _number(self.settings, "min_available_ram_gb")
            stop_c = _number(self.settings, "stop_temperature_c")
            hot_seconds = _number(self.settings, "hot_seconds", 30)
            start_c = _number(self.settings, "start_temperature_c", stop_c)
            hard_c = _number(self.settings, "hard_temperature_c", 90)
            if start_c > stop_c or hard_c < stop_c:
                raise ValueError("Inconsistent temperature thresholds")
        except ValueError as exc:
            self._stop(str(exc))
        if now - self.started >= minutes * 60:
            self._stop("Session time budget reached; completed work remains saved")
        if not force and now - self.last_check < interval:
            return
        self.last_check = now
        try:
            available = psutil.virtual_memory().available
            process = psutil.Process()
            rss = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
            self.snapshot = {"measured_timestamp": time.time(), "cpu_c": None,
                             "sensor_age_seconds": None, "process_tree_rss_bytes": rss,
                             "system_available_ram_bytes": available}
            temperature = self.temperature()
            self.snapshot.update(cpu_c=temperature, sensor_age_seconds=self.sensor_age_seconds)
        except (psutil.Error, OSError, ValueError, TypeError, ResourceLimit) as exc:
            self._stop(f"Cannot monitor resources: {exc}")
        self._emit("sample")
        if available / 1024**3 < min_available:
            self._stop("Available RAM is below the configured minimum")
        if rss / 1024**3 > max_rss:
            self._stop("Training process memory budget exceeded")
        if temperature is None:
            if self.settings.get("require_temperature", True):
                self._stop("CPU temperature unavailable. Connect a live sensor before training.")
        else:
            if temperature >= hard_c:
                self._stop("Hard CPU temperature limit reached")
            if not self.preflight_passed and temperature >= start_c:
                self._stop("CPU too hot to start; wait for cooling below the start threshold")
            if temperature >= stop_c:
                self.hot_since = now if self.hot_since is None else self.hot_since
                if now - self.hot_since >= hot_seconds:
                    self._stop("Sustained CPU temperature exceeded the configured operating limit")
            else:
                self.hot_since = None
        self.preflight_passed = True
