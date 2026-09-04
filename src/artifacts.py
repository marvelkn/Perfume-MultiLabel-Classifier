"""Versioned, explicit artifact writes and reproducibility metadata."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from datetime import datetime, timezone

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")
    temporary.replace(target)

def environment():
    packages = {}
    for name in ("numpy", "pandas", "rdkit", "scikit-learn", "xgboost", "lightgbm", "optuna", "onnxmltools", "onnxruntime", "onnx", "protobuf", "joblib", "scipy", "iterative-stratification"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True, encoding="utf-8", stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    source_files = sorted(set(root.glob("*.py")) | set(root.glob("requirements*.txt")) |
                          set(root.glob("*.yaml")) | set((root/"src").rglob("*.py")))
    source_hashes = {p.relative_to(root).as_posix(): digest(p) for p in source_files}
    return {"python": platform.python_version(), "packages": packages, "commit": git("rev-parse", "HEAD"),
            "source_sha256": json_hash(source_hashes), "source_files": source_hashes,
            "working_tree_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat()}
