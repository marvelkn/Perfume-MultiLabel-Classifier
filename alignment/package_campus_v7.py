"""Self-contained campus v7 ZIP with data and read-only historical summaries."""
from pathlib import Path
import hashlib
import json
import zipfile

from .data import ROOT

FOLDERS = ("alignment", "src", "data/builds/perfume-five-grouped-v2",
           "data/builds/perfume-m2048-d8-v7", "data/snapshots/perfume-five-v1", "v7_reference")
FILES = ("config.yaml", "requirements_campus_v7.txt", "DESCRIPTOR_COMPARISON_V7_PLAN.md",
         "WINDOWS_GUIDE_V7.md", "LINUX_GUIDE_V7.md", "AFTER_V7_REPORT_UPDATE.md", "RUN_GROUPED_V7.cmd", "RUN_GROUPED_V7.sh",
         "COLLECT_GROUPED_V7_RESULTS.cmd", "COLLECT_GROUPED_V7_RESULTS.sh",
         "reports/campus_20260909/full/amendment.json")


def paths(root):
    root = Path(root)
    found = set()
    for name in FOLDERS:
        folder = root/name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        found.update(p for p in folder.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts
                     and p.suffix not in (".pyc",".tmp"))
    found.update(root/name for name in FILES)
    for path in found:
        if not path.is_file():
            raise FileNotFoundError(path)
    return found


def write_archive(root, files, target, metadata, manifest_name):
    root, target = Path(root), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".zip.tmp")
    hashes = {}
    with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(files):
            name = path.relative_to(root).as_posix()
            data = path.read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            z.writestr(name,data)
        z.writestr(manifest_name,json.dumps({**metadata,"files":hashes},indent=2))
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise ValueError("Corrupt ZIP.")
        for name, expected in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError(f"ZIP checksum mismatch: {name}")
    temporary.replace(target)
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".zip.sha256").write_text(f"{checksum}  {target.name}\n",encoding="ascii")
    return {"archive":str(target),"sha256":checksum,"bytes":target.stat().st_size,"files":len(hashes)}


def build(root=ROOT):
    root = Path(root).resolve()
    result = write_archive(root,paths(root),
                          root/".local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v7-m2048-d8.zip",
                          {"protocol":"perfume-five-grouped-v7-m2048-d8","dataset_included":True,
                           "five_raw_sources_included":True,"candidates":8,
                           "protocol_revision":2,"morgan_bits":2048,
                           "descriptor_count":8,"feature_dimensions":[2048,2056],
                           "fresh_optuna_studies":4,"tuning_hours_total":32,
                           "training_started":False},"CAMPUS_PACKAGE_MANIFEST.json")
    print(json.dumps(result,indent=2))
    return result


if __name__ == "__main__":
    build()
