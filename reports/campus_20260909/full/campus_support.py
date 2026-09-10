"""Campus pipeline policy, atomic ledger and independent Windows process ownership."""
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ALGORITHMS = ("xgb", "lgbm")
PROFILE = "campus-full-no-temperature-v1"
TARGET_ATTEMPTS = 30
TUNING_SECONDS = 14400
SESSION_SECONDS = 1200


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def charge(entries, algorithm):
    # An interrupted controller conservatively charges its full reservation.
    return sum(e.get("elapsed_seconds", e["reserved_seconds"]) for e in entries
               if e["operation"] == "tune" and e["algorithm"] == algorithm)


def admit(entries, algorithm, attempts):
    return attempts < TARGET_ATTEMPTS and TUNING_SECONDS - charge(entries, algorithm) >= 1


def best_complete(trials):
    values = [t for t in trials if t.state.name == "COMPLETE" and t.value is not None and math.isfinite(t.value)]
    return min(values, key=lambda t: (-t.value, t.number)) if values else None


def install_policy(seconds):
    import sys
    sys.path.insert(0, str(ROOT))
    os.environ["ESSENZA_CONFIG"] = str(ROOT / "config.safe-training.yaml")
    for name in ("ESSENZA_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    from src import runtime
    from src.config import CONFIG
    original_validate = runtime.validate_training_resources
    def validate(settings):
        if settings.get("profile") != PROFILE or settings.get("require_temperature") is not False:
            raise ValueError("Use the explicit campus full profile")
        original_validate(dict(settings, profile="safe-training", require_temperature=True))
    class CampusGuard(runtime.ResourceGuard):
        def __init__(self, settings, temperature_file=None, **kwargs):
            validate(settings)
            if temperature_file is not None:
                raise ValueError("Campus mode has no temperature source")
            bounded = dict(settings)
            bounded["session_minutes"] = min(settings["session_minutes"], max(0.01, seconds-5)/60)
            super().__init__(bounded, None, **kwargs)
        def temperature(self):
            self.sensor_age_seconds = None
            return None
    CONFIG["resources"].update(profile=PROFILE, require_temperature=False,
                               thermal_monitoring="disabled_by_user", amendment_id=PROFILE)
    runtime.validate_training_resources = validate
    runtime.ResourceGuard = CampusGuard
    return CONFIG


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
