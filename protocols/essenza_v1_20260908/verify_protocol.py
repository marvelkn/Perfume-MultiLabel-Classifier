"""Verify the local preregistration without loading features, labels, or models."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def validate_files(root, files):
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Path escapes project: {name}")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Frozen file mismatch: {name}")


def planned_trials(xgb_seconds, lgbm_seconds, budget):
    durations = (xgb_seconds, lgbm_seconds)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("Both smoke estimates must be finite and positive")
    attempts = min(budget["maximum_planned_attempts"],
                   math.floor((1 - budget["time_reserve_fraction"]) *
                              budget["active_seconds_per_algorithm"] / max(durations)))
    if attempts < budget["minimum_planned_attempts"] or max(durations) > 960:
        raise ValueError("Resource-only budget gate STOP; no tuning may start")
    return attempts


def verify():
    frozen = json.loads((DIRECTORY / "freeze_manifest.json").read_text(encoding="utf-8"))
    validate_files(ROOT, frozen["files"])
    protocol = json.loads((DIRECTORY / "protocol.json").read_text(encoding="utf-8"))
    current_source = sorted(set(ROOT.glob("*.py")) | set(ROOT.glob("requirements*.txt")) |
                            set(ROOT.glob("*.yaml")) | set((ROOT / "src").rglob("*.py")))
    if {path.relative_to(ROOT).as_posix() for path in current_source} != set(protocol["environment"]["source_files"]):
        raise ValueError("Production source inventory changed; record an amendment")
    if protocol["status"] != "FROZEN_PRETRAINING_RULES":
        raise ValueError("Protocol is not frozen")
    if protocol["budget"]["trial_count_final"] is not None:
        raise ValueError("Step 3 does not supply a measured final trial count")
    if protocol["imbalance"]["mlsmote_enabled"]:
        raise ValueError("MLSMOTE was not authorized for this protocol")
    primary = json.loads((ROOT / protocol["partitions"]["primary"]).read_text())
    variants = json.loads((ROOT / protocol["sensitivity"]["partition_file"]).read_text())
    fit = set(primary["fit"])
    threshold = set(primary["threshold"])
    if len(fit) != 4575 or len(threshold) != 808 or fit & threshold:
        raise ValueError("Invalid primary partition")
    if fit | threshold != set(range(5383)):
        raise ValueError("Primary partitions do not cover development")
    groups = json.loads((ROOT / "reports/step3_20260908/development_scaffold_groups.json").read_text())
    if set(variants) != set(protocol["sensitivity"]["variant_order"]):
        raise ValueError("Sensitivity variants differ")
    for value in variants.values():
        scores = []
        for fold in value["folds"]:
            train, score = set(fold["train"]), set(fold["score"])
            if train & score or train | score != fit or (train | score) & threshold:
                raise ValueError("Sensitivity role leakage")
            if value["grouped"] and {groups[i] for i in train} & {groups[i] for i in score}:
                raise ValueError("Scaffold leakage")
            scores.extend(fold["score"])
        if sorted(scores) != sorted(fit):
            raise ValueError("Sensitivity score coverage differs")
    return {"status":"PASS", "protocol_id":protocol["protocol_id"],
            "verified_files":len(frozen["files"]), "test_arrays_deserialized":False,
            "training_started":False, "final_trial_count":"pending Step 5 smoke-derived freeze"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
