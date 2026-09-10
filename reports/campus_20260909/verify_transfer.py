"""Verify the campus overlay and pinned environment without loading dataset arrays."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
manifest = json.loads((ROOT / "reports/campus_20260909/transfer_manifest.json").read_text())
for relative, expected in manifest["files"].items():
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise SystemExit("STOP: path outside repository")
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"STOP: transfer hash differs: {relative}")
protocol = json.loads((ROOT / "protocols/essenza_v1_20260908/protocol.json").read_text())
if platform.python_version() != protocol["environment"]["python"]:
    raise SystemExit(f"STOP: Python {platform.python_version()} differs from frozen Python")
for name, expected in protocol["environment"]["packages"].items():
    actual = metadata.version(name)
    if actual != expected:
        raise SystemExit(f"STOP: {name}: {actual} != {expected}")
print(json.dumps({"status": "PASS", "files": len(manifest["files"]),
                  "python": platform.python_version(), "executable": sys.executable,
                  "platform": platform.platform(), "training_started": False,
                  "test_arrays_loaded": False}, indent=2))
