"""Conservative training budget; never changes OS power or firmware settings."""
import json
import math
import time
from pathlib import Path
import psutil

class ResourceLimit(RuntimeError):
    pass

class ResourceGuard:
    def __init__(self, settings, temperature_file=None):
        self.settings = settings
        self.temperature_file = Path(temperature_file) if temperature_file else None
        self.started = time.monotonic()
        self.hot_since = None
        self.last_check = -float("inf")

    def temperature(self):
        if self.temperature_file:
            data = json.loads(self.temperature_file.read_text(encoding="utf-8"))
            age = time.time() - float(data["timestamp"])
            value = float(data["cpu_c"])
            if not math.isfinite(age) or age < -2 or age > self.settings.get("temperature_max_age_seconds", 10) or not math.isfinite(value) or not 0 < value < 150:
                raise ResourceLimit("CPU sensor telemetry is stale or invalid")
            return value
        reader = getattr(psutil, "sensors_temperatures", None)
        if reader:
            values = [entry.current for name, entries in reader().items()
                      if name.lower() in {"coretemp", "k10temp", "cpu_thermal"}
                      for entry in entries if entry.current is not None]
            if values:
                return max(values)
        return None

    def check(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_check < 1:
            return
        self.last_check = now
        if now - self.started >= self.settings["session_minutes"] * 60:
            raise ResourceLimit("Session time budget reached; completed trials are saved")
        available = psutil.virtual_memory().available / 1024**3
        process = psutil.Process()
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if available < self.settings["min_available_ram_gb"]:
            raise ResourceLimit(f"Available RAM is only {available:.2f} GiB")
        if rss / 1024**3 > self.settings["max_process_ram_gb"]:
            raise ResourceLimit("Training process memory budget exceeded")
        try:
            temperature = self.temperature()
        except (OSError, ValueError, KeyError) as exc:
            raise ResourceLimit("Cannot read valid CPU sensor telemetry") from exc
        if temperature is None and self.settings.get("require_temperature", True):
            raise ResourceLimit("CPU temperature unavailable. Connect a live sensor before real-data training.")
        if temperature is not None:
            if temperature >= self.settings["stop_temperature_c"]:
                self.hot_since = now if self.hot_since is None else self.hot_since
                if now - self.hot_since >= self.settings.get("hot_seconds", 30):
                    raise ResourceLimit("Sustained CPU temperature exceeded the configured operating limit")
            else:
                self.hot_since = None
