"""Collect completed or interrupted v7 results after the training process stops."""
from datetime import datetime
from pathlib import Path
import json

from .data import ROOT
from . import experiments as core
from .experiments_v7 import verify_files
from .package_campus_v7 import paths, write_archive


def collect(root=ROOT):
    root = Path(root).resolve()
    run = root/"runs/perfume-five-grouped-v7-m2048-d8"
    if not run.is_dir():
        raise FileNotFoundError("Belum ada hasil v7.")
    if (run/".alignment.lock").exists():
        raise RuntimeError("Training masih aktif atau lock belum dipulihkan. Jalankan launcher lagi setelah proses berhenti.")
    complete = (run/"complete.json").exists() and (run/"completion_manifest.json").exists()
    if complete:
        verify_files(run,core.read(run/"completion_manifest.json")["files"])
    files = paths(root) | {p for p in run.rglob("*") if p.is_file() and p.suffix!=".tmp"}
    target = root/"campus-results"/("perfume-grouped-v7-m2048-d8-results-"+datetime.now().strftime("%Y%m%d-%H%M%S-%f")+".zip")
    result = write_archive(root,files,target,{"protocol":"perfume-five-grouped-v7-m2048-d8","complete":complete},"RESULTS_MANIFEST.json")
    print(json.dumps(result,indent=2))
    return result


if __name__ == "__main__":
    collect()
