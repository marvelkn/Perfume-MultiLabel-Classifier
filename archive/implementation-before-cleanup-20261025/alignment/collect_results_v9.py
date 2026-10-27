"""Bundle V9 code, inputs and results once the training process has stopped."""
from datetime import datetime
from pathlib import Path
import json

from .data import ROOT
from . import experiments as core
from .experiments_v9 import RUN_ID, verify_files
from .package_campus_v9 import paths
from .package_campus_v7 import write_archive


def collect(root=ROOT):
    root = Path(root).resolve()
    run = root / "runs" / RUN_ID
    if not run.is_dir():
        raise FileNotFoundError("Belum ada hasil V9.")
    if (run / ".alignment.lock").exists():
        raise RuntimeError("Hentikan training dahulu. Setelah crash, jalankan launcher untuk memulihkan lock.")
    complete = (run / "complete.json").exists() and (run / "completion_manifest.json").exists()
    if complete:
        verify_files(run, core.read(run / "completion_manifest.json")["files"])
    files = paths(root) | {p for p in run.rglob("*") if p.is_file() and not p.name.endswith(".tmp")}
    name = "perfume-grouped-v9-bits-d8-results-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".zip"
    result = write_archive(root, files, root / "campus-results" / name,
                           {"protocol": RUN_ID, "complete": complete}, "RESULTS_MANIFEST.json")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    collect()
