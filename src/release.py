"""Package and verify a completed v6 LightGBM release without fitting models."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from .artifacts import digest, write_json
from .featurize import DESCRIPTORS, FeatureSpec

DEFAULT_RELEASE = Path(__file__).resolve().parents[1] / "releases/grouped-v6-lgbm-d"
RUNTIME_PACKAGES = ("numpy", "pandas", "scikit-learn", "rdkit", "lightgbm", "joblib")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contained(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Artifact path escapes package: {name}")
    return path


def runtime_spec(bundle):
    feature = bundle["feature_spec"]
    if tuple(feature["descriptors"]) != DESCRIPTORS:
        raise ValueError("Descriptor order differs from inference implementation")
    spec = FeatureSpec(radius=feature["radius"], n_bits=feature["n_bits"],
                       include_chirality=feature["include_chirality"],
                       use_descriptors=bundle["use_descriptors"],
                       fragment_policy=feature["fragment_policy"],
                       rdkit_version=feature["rdkit_version"])
    key = "B_D" if spec.use_descriptors else "A_C"
    if feature["dimensions"][key] != spec.n_features:
        raise ValueError("Training feature dimension mismatch")
    return spec


def verify_release(directory=DEFAULT_RELEASE):
    directory = Path(directory).resolve()
    manifest = read(directory / "release.json")
    if manifest["format_version"] != 1 or manifest["selected_candidate"] != "lgbm_D":
        raise ValueError("Unsupported release")
    for name, expected in manifest["files"].items():
        if digest(contained(directory, name)) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    bundle_path = "evidence/inference_bundle.json"
    if bundle_path not in manifest["files"] or "feature_spec.json" not in manifest["files"]:
        raise ValueError("Missing verified feature/label contract")
    bundle = read(directory / bundle_path)
    spec = runtime_spec(bundle)
    if read(directory / "feature_spec.json") != spec.to_dict():
        raise ValueError("Runtime features differ from training")
    if manifest["feature_schema_id"] != spec.schema_id:
        raise ValueError("Feature schema mismatch")
    labels = bundle["labels"]
    if len(labels) != 109 or len(set(labels)) != len(labels) or manifest["labels"] != labels:
        raise ValueError("Invalid label order")
    if bundle["selected_candidate"] != manifest["selected_candidate"] or bundle["primary_threshold"] != 0.5:
        raise ValueError("Selection or primary threshold changed")
    if set(manifest["models"]) != set(labels):
        raise ValueError("Missing or extra models")
    for label in labels:
        name = manifest["models"][label]
        original = bundle["models"][label]
        if name != "models/" + original["file"] or manifest["files"].get(name) != original["sha256"]:
            raise ValueError(f"Model/label binding mismatch: {label}")
    if manifest["environment"] != {key: bundle["environment"][key] for key in RUNTIME_PACKAGES}:
        raise ValueError("Runtime versions differ from source bundle")
    return manifest, spec


def build_release(run, output=DEFAULT_RELEASE):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Release already exists; verify it or use a new output directory")
    if output.is_relative_to(run):
        raise ValueError("Do not write a release inside frozen research results")
    completion = read(run / "completion_manifest.json")["files"]
    bundle = read(run / "inference_bundle.json")
    selected = bundle["selected_candidate"]
    selection = read(run / "selection_frozen.json")
    complete = read(run / "complete.json")
    if selected != "lgbm_D" or bundle["algorithm"] != "lgbm" or bundle["models_directory"] != selected:
        raise ValueError("Expected the frozen LightGBM D candidate")
    if any(item["overall_selected_from_repeated_grouped_cv"] != selected for item in (selection, complete)):
        raise ValueError("Frozen selection differs")
    if digest(run / "selection_frozen.json") != bundle["selection_sha256"]:
        raise ValueError("Selection checksum mismatch")
    spec = runtime_spec(bundle)
    if spec.n_features != 1029 or len(bundle["labels"]) != 109:
        raise ValueError("Unexpected v6 dimensions")
    copies = {}
    evidence = ["inference_bundle.json", "selection_frozen.json", "complete.json", "identity.json",
                "RESULTS_SUMMARY.md", "comparison_all.csv", "ablation_deltas.csv", "comparison_history.csv",
                f"{selected}/test_summary.json", f"{selected}/test_summary_validation_thresholds.json",
                f"{selected}/models.json", f"{selected}/recipe.json", f"{selected}/training_policy.json",
                bundle["secondary_thresholds_path"]]
    for name in evidence:
        copies["evidence/" + name] = name
    for label in bundle["labels"]:
        info = bundle["models"][label]
        name = selected + "/" + info["file"]
        if completion.get(name) != info["sha256"]:
            raise ValueError(f"Model not bound to completion: {label}")
        copies["models/" + info["file"]] = name
    # Check each copied artifact against the completed run before deserializing anything.
    for name in copies.values():
        if name not in completion or digest(contained(run, name)) != completion[name]:
            raise ValueError(f"Source completion checksum mismatch: {name}")
    output.mkdir(parents=True)
    files = {}
    for target, source in copies.items():
        destination = contained(output, target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(contained(run, source), destination)
        files[target] = digest(destination)
        if files[target] != completion[source]:
            raise ValueError(f"Copy checksum mismatch: {target}")
    write_json(output / "feature_spec.json", spec.to_dict())
    files["feature_spec.json"] = digest(output / "feature_spec.json")
    manifest = {"format_version": 1, "release_id": "grouped-v6-lgbm-d",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "selected_candidate": selected, "labels": bundle["labels"],
                "feature_schema_id": spec.schema_id,
                "models": {label: "models/" + bundle["models"][label]["file"] for label in bundle["labels"]},
                "environment": {key: bundle["environment"][key] for key in RUNTIME_PACKAGES},
                "score_interpretation": bundle["score_interpretation"],
                "source_run": str(run), "source_completion_sha256": digest(run / "completion_manifest.json"),
                "primary_threshold": 0.5, "secondary_thresholds": "evidence/" + bundle["secondary_thresholds_path"],
                "status": {"training": "completed", "evaluation": "exploratory", "onnx": "deferred", "android": "deferred"},
                "files": files}
    write_json(output / "release.json", manifest)
    verify_release(output)
    return {"release": str(output), "models": len(manifest["models"]), "features": spec.n_features,
            "verified_files": len(files), "manifest_sha256": digest(output / "release.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Completed local v6 run to package")
    parser.add_argument("--output", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        manifest, spec = verify_release(args.output)
        result = {"release_id": manifest["release_id"], "verified_files": len(manifest["files"]),
                  "features": spec.n_features, "models": len(manifest["models"])}
    elif args.run:
        result = build_release(args.run, args.output)
    else:
        parser.error("Use --run or --verify")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
