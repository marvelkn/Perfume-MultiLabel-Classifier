"""File, lingkungan, dan batas eksekusi; tidak menjalankan ML saat diimpor."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
COMPUTE_PROFILE = ROOT / "research/config/compute.json"
WEIGHTING_POLICY = "balanced-sample-weight-from-training-partition-v1"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def environment():
    return {name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "rdkit", "xgboost", "lightgbm", "optuna", "joblib")}


def compute_profile():
    profile = read(COMPUTE_PROFILE)
    threads = profile.get("threads_per_model")
    if not isinstance(threads, int) or not 1 <= threads <= 64:
        raise ValueError("Invalid threads_per_model in campus compute profile.")
    if profile.get("execution_device") != "cpu":
        raise ValueError("This reproducible pipeline currently supports the CPU profile only.")
    return profile


def check_host():
    amendment = read(ROOT / "reports/campus_20260909/full/amendment.json")
    if platform.system() not in {"Windows", "Linux"}:
        raise RuntimeError("Campus execution supports Windows or Linux only.")
    if platform.node().casefold() == amendment["excluded_laptop_host"].casefold():
        raise RuntimeError("Campus execution is blocked on the original laptop.")


@contextmanager
def run_lock(run):
    path = run / ".alignment.lock"
    if path.exists():
        previous = read(path)
        same_host = previous.get("host", "").casefold() == platform.node().casefold()
        active = same_host and psutil.pid_exists(int(previous.get("pid", -1)))
        if active:
            raise RuntimeError("Another training process is still active for this run.")
        path.unlink()
    with path.open("x", encoding="utf-8") as f:
        json.dump({"host": platform.node(), "pid": os.getpid(), "created": time.time()}, f)
    try:
        yield
    finally:
        path.unlink()


class Limits:
    def __init__(self, deadline=None):
        self.deadline = deadline
        self.checked = 0

    def check(self, force=False):
        now = time.perf_counter()
        if self.deadline is not None and now >= self.deadline:
            raise TimeoutError("Tuning budget exhausted.")
        if not force and now - self.checked < 1:
            return
        self.checked = now
        if psutil.virtual_memory().available < 4 * 1024**3:
            raise RuntimeError("Less than 4 GiB system memory available.")
        if psutil.Process().memory_info().rss > 4 * 1024**3:
            raise RuntimeError("Training process exceeds 4 GiB RSS.")


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
