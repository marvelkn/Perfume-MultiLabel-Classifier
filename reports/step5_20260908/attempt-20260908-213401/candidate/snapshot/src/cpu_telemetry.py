"""Bridge a verified LibreHardwareMonitor CPU log to ResourceGuard telemetry."""
import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import psutil


class TelemetryError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reading:
    timestamp: float
    cpu_c: float
    sensor_id: str
    sensor_name: str


def _row(line):
    try:
        return next(csv.reader([line.decode("utf-8-sig").strip()], strict=True))
    except (UnicodeError, csv.Error, StopIteration) as exc:
        raise TelemetryError("Invalid monitor CSV row") from exc


def sensor_columns(path):
    with Path(path).open("rb") as stream:
        identifiers = _row(stream.readline(65536))
        names = _row(stream.readline(65536))
    if len(identifiers) != len(names) or not names or names[0] != "Time":
        raise TelemetryError("Expected LibreHardwareMonitor identifier and name headers")
    return identifiers, names


class LhmCsvReader:
    def __init__(self, path, sensor_id):
        self.path = Path(path)
        self.identifiers, self.names = sensor_columns(self.path)
        if not re.fullmatch(r"/(?:amdcpu|intelcpu)/\d+/temperature/\d+", sensor_id):
            raise TelemetryError("Select a CPU temperature identifier, never GPU or disk")
        if self.identifiers.count(sensor_id) != 1:
            raise TelemetryError("Sensor identifier missing or ambiguous")
        self.index = self.identifiers.index(sensor_id)
        self.sensor_id = sensor_id
        self.sensor_name = self.names[self.index]
        if self.sensor_name not in {"Core (Tctl/Tdie)", "CPU Package", "Core (Tdie)"}:
            raise TelemetryError("Selected sensor is not a recognized CPU package/die sensor")

    def read(self, now=None):
        with self.path.open("rb") as stream:
            identifiers = _row(stream.readline(65536))
            names = _row(stream.readline(65536))
            if identifiers != self.identifiers or names != self.names:
                raise TelemetryError("Monitor log sensor headers changed; revalidate the source")
            header_end = stream.tell()
            size = os.fstat(stream.fileno()).st_size
            start = max(header_end, size - 65536)
            stream.seek(start)
            if start > header_end:
                stream.readline()
            lines = stream.read(65536).splitlines(keepends=True)
        # Ignore an in-progress append; its predecessor retains its original age.
        complete = [line for line in lines if line.endswith(b"\n")]
        if not complete:
            raise TelemetryError("No complete sensor sample")
        row = _row(complete[-1])
        if len(row) != len(self.identifiers):
            raise TelemetryError("Sensor sample column count changed")
        try:
            # LHM v0.9.6 uses invariant-culture local time, without a UTC offset.
            timestamp = datetime.strptime(row[0], "%m/%d/%Y %H:%M:%S").astimezone().timestamp()
            value = float(row[self.index])
        except (ValueError, OverflowError) as exc:
            raise TelemetryError("CPU timestamp or value missing/invalid") from exc
        age = (time.time() if now is None else now) - timestamp
        if not math.isfinite(value) or not 0 < value < 150:
            raise TelemetryError("CPU temperature missing or implausible")
        if not math.isfinite(age) or not -2 <= age <= 5:
            raise TelemetryError("CPU sample stale or timestamp in the future")
        return Reading(timestamp, value, self.sensor_id, self.sensor_name)


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def collect(reader, output, history, duration_seconds):
    started = time.monotonic()
    process = psutil.Process()
    summary = {
        "status": "STOP", "sensor_id": reader.sensor_id,
        "sensor_name": reader.sensor_name, "source_log": str(reader.path.resolve()),
        "started_at": datetime.now().astimezone().isoformat(),
        "observations": 0, "cpu_min_c": None, "cpu_max_c": None,
        "max_source_age_seconds": 0, "peak_collector_rss_bytes": 0,
        "live_sensor_verification_required": True,
    }
    last_value = None
    changed_at = started
    last_timestamp = None
    psutil.cpu_percent()
    try:
        # A new history path prevents silently mixing separate baseline attempts.
        with Path(history).open("x", encoding="utf-8") as stream:
            while time.monotonic() - started < duration_seconds:
                reading = reader.read()
                observed = time.time()
                if last_timestamp is not None and reading.timestamp < last_timestamp:
                    raise TelemetryError("Source timestamp moved backwards")
                if reading.cpu_c != last_value:
                    changed_at = time.monotonic()
                    last_value = reading.cpu_c
                elif time.monotonic() - changed_at >= 60:
                    raise TelemetryError("CPU value unchanged for 60 seconds; inspect the sensor")
                last_timestamp = reading.timestamp
                # Preserve acquisition time, not collector time, for stale detection.
                write_json_atomic(output, asdict(reading))
                rss = process.memory_info().rss
                record = {
                    **asdict(reading), "observed_timestamp": observed,
                    "age_seconds": observed - reading.timestamp,
                    "collector_rss_bytes": rss,
                    "system_available_ram_bytes": psutil.virtual_memory().available,
                    "system_cpu_percent": psutil.cpu_percent(),
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                summary["observations"] += 1
                summary["cpu_min_c"] = min(summary["cpu_min_c"] or reading.cpu_c, reading.cpu_c)
                summary["cpu_max_c"] = max(summary["cpu_max_c"] or reading.cpu_c, reading.cpu_c)
                summary["cpu_end_c"] = reading.cpu_c
                summary["max_source_age_seconds"] = max(summary["max_source_age_seconds"], record["age_seconds"])
                summary["peak_collector_rss_bytes"] = max(summary["peak_collector_rss_bytes"], rss)
                time.sleep(min(1, max(0, duration_seconds - (time.monotonic() - started))))
        summary["status"] = "RECORDED"
        return summary
    except (OSError, ValueError, TelemetryError, KeyboardInterrupt) as exc:
        summary["reason"] = str(exc) or "Interrupted"
        raise
    finally:
        summary["duration_seconds"] = time.monotonic() - started
        summary["ended_at"] = datetime.now().astimezone().isoformat()
        # Terminated collectors invalidate the live file immediately.
        write_json_atomic(output, {"timestamp": 0, "cpu_c": 0, "status": "STOP"})
        write_json_atomic(Path(history).with_suffix(".summary.json"), summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--list-sensors", action="store_true")
    parser.add_argument("--sensor-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--duration-seconds", type=int, default=610)
    args = parser.parse_args(argv)
    if not args.list_sensors and not all((args.sensor_id, args.output, args.history)):
        parser.error("--sensor-id, --output and a new --history path are required")
    if not 1 <= args.duration_seconds <= 7200:
        parser.error("--duration-seconds must be 1..7200")
    try:
        if args.list_sensors:
            identifiers, names = sensor_columns(args.log)
            print(json.dumps([{"id": key, "name": name} for key, name in zip(identifiers, names)
                              if "/temperature/" in key], indent=2))
        else:
            paths = [args.log, args.output, args.history, args.history.with_suffix(".summary.json")]
            if len({path.resolve() for path in paths}) != len(paths):
                raise TelemetryError("Source, output, history, and summary paths must differ")
            if args.history.exists() or args.history.with_suffix(".summary.json").exists():
                raise TelemetryError("Use a new history path for each attempt")
            reader = LhmCsvReader(args.log, args.sensor_id)
            print(json.dumps(collect(reader, args.output, args.history, args.duration_seconds), indent=2))
        return 0
    except (OSError, ValueError, TelemetryError, KeyboardInterrupt) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
