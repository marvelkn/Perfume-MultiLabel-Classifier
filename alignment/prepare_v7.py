"""Prepare portable inputs once; never fit research models."""
import argparse
from pathlib import Path
import shutil

import pandas as pd

from . import experiments as core
from .data import ROOT, digest
from .descriptors_v7 import prepare, EXTRA
from .perfume_data import OUTPUT

REFERENCE = ROOT / "v7_reference"
DEFAULT_V6 = ROOT / ".local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v6/perfume-campus-grouped-v6/runs/perfume-five-grouped-v6"


def prepare_reference(v6):
    v6 = Path(v6)
    identity = core.read(v6 / "identity.json")
    if identity["dataset"] != digest(OUTPUT / "dataset_manifest.json"):
        raise ValueError("Historical v6 dataset differs.")
    manifest = core.read(v6 / "completion_manifest.json")
    names = ["complete.json", "comparison_all.csv", "selection_frozen.json", "identity.json"]
    names += [f"robustness/{a}_{c}/robustness_summary.json"
              for a in ("xgb", "lgbm") for c in "ABCD"]
    for name in names:
        if digest(v6 / name) != manifest["files"][name]:
            raise ValueError(f"Historical v6 evidence changed: {name}")
    table = pd.read_csv(v6 / "comparison_all.csv")
    if set(table.candidate) != {f"{a}_{c}" for a in ("xgb","lgbm") for c in "ABCD"}:
        raise ValueError("Incomplete v6 comparison.")
    if REFERENCE.exists() and any(REFERENCE.iterdir()):
        saved = core.read(REFERENCE / "manifest.json")
        for name, expected in saved["files"].items():
            if digest(REFERENCE / name) != expected:
                raise ValueError(f"Existing reference changed: {name}")
        if saved["source_completion_manifest_sha256"] != digest(v6/"completion_manifest.json"):
            raise ValueError("Cannot replace frozen historical evidence.")
        return saved
    REFERENCE.mkdir(parents=True, exist_ok=True)
    mapping = {name: name for name in names}
    mapping["comparison_all.csv"] = "comparison_all_v6.csv"
    for source, dest in mapping.items():
        target = REFERENCE / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(v6 / source, target)
    saved = {
        "source_run":"perfume-five-grouped-v6", "dataset_manifest_sha256":identity["dataset"],
        "source_completion_manifest_sha256":digest(v6/"completion_manifest.json"),
        "source_files":{name:manifest["files"][name] for name in names},
        "files":{dest:digest(REFERENCE/dest) for dest in mapping.values()},
        "usage":"Historical context only; not used for v7 hyperparameters or selection.",
    }
    core.write(REFERENCE/"manifest.json", saved)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-run", type=Path, default=DEFAULT_V6)
    args = parser.parse_args()
    prepare(OUTPUT, EXTRA)
    prepare_reference(args.v6_run)
    path = ROOT/"alignment/protocol_v7.json"
    p = core.read(path)
    for key, expected in {
        "feature_manifest_sha256":digest(EXTRA/"manifest.json"),
        "reference_manifest_sha256":digest(REFERENCE/"manifest.json"),
    }.items():
        if p[key] not in ("PREPARATION_PENDING", expected):
            raise ValueError("Cannot replace protocol inputs.")
        p[key] = expected
    core.write(path,p)
    from .experiments_v7 import preflight
    meta, _ = preflight(OUTPUT)
    print({"status":"V7_M2048_D8_PREPARED", "molecules":6686, "labels":len(meta["labels"]),
           "training_started":False})


if __name__ == "__main__":
    main()
