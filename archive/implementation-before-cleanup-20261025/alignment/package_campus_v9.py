"""Build a ready-to-copy V9 folder and a verified, self-contained ZIP."""
from pathlib import Path
import json
import shutil

from .data import ROOT, digest
from .package_campus_v7 import write_archive

PACKAGE = "perfume-campus-grouped-v9-bits-d8"
FOLDERS = ("alignment", "src", "data/builds/perfume-five-grouped-v2",
           "data/builds/perfume-m2048-d8-v7", "data/snapshots/perfume-five-v1")
FILES = ("config.yaml", "requirements_campus_v9.txt", "V9_EXPERIMENT_PLAN.md", "START_HERE_V9.md",
         "WINDOWS_GUIDE_V9.md", "LINUX_GUIDE_V9.md", "RUN_GROUPED_V9.cmd", "RUN_GROUPED_V9.sh",
         "COLLECT_GROUPED_V9_RESULTS.cmd", "COLLECT_GROUPED_V9_RESULTS.sh",
         "reports/campus_20260909/full/amendment.json")


def paths(root):
    root = Path(root)
    found = set()
    for name in FOLDERS:
        folder = root / name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        found.update(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                     and p.suffix not in (".pyc", ".tmp"))
    found.update(root / name for name in FILES)
    if any(not p.is_file() for p in found):
        raise FileNotFoundError("Package input incomplete.")
    return found


def build(root=ROOT):
    root = Path(root).resolve()
    files = paths(root)
    metadata = {"protocol": "perfume-five-grouped-v9-bits-d8", "dataset_included": True,
                "five_raw_sources_included": True, "labels": 109, "molecules": 6686,
                "candidates": 8, "morgan_bits": [1024, 2048], "descriptor_count": 8,
                "feature_dimensions": [1032, 2056], "fresh_optuna_studies": 4,
                "hours_per_study": 16, "total_tuning_hours": 64, "snapshot_hours": [8, 16],
                "execution_device": "cpu", "threads_per_model": 6, "training_started": False,
                "research_status": "V7 locked; V8 pre-ML exploration; V9 separate exploratory extension."}
    parent = root / ".local-tools/campus-transfer-full-20260909"
    folder = parent / PACKAGE
    if folder.exists():
        raise FileExistsError(f"Refuse to overwrite an existing campus folder: {folder}")
    folder.mkdir(parents=True)
    hashes = {}
    for source in sorted(files):
        relative = source.relative_to(root)
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative.as_posix()] = digest(source)
        if digest(target) != hashes[relative.as_posix()]:
            raise ValueError(f"Copy differs: {relative}")
    (folder / "CAMPUS_PACKAGE_MANIFEST.json").write_text(json.dumps({**metadata, "files": hashes}, indent=2) + "\n", encoding="utf-8")
    result = write_archive(root, files, parent / f"{PACKAGE}.zip", metadata, "CAMPUS_PACKAGE_MANIFEST.json")
    result["folder"] = str(folder)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    build()
