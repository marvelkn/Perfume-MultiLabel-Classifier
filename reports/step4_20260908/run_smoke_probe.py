"""Execute only the frozen Step 4 resource probe; never score or tune models."""

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocols/essenza_v1_20260908"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cpu_telemetry import write_json_atomic
from src.runtime import (
    ResourceGuard,
    ResourceLimit,
    exclusive_run,
    validate_training_resources,
)


def frozen_inputs():
    verification = runpy.run_path(str(PROTOCOL / "verify_protocol.py"))
    verification["verify"]()
    extra = json.loads(
        (DIRECTORY / "smoke_runner_freeze.json").read_text(encoding="utf-8")
    )
    verification["validate_files"](ROOT, extra["files"])
    if (
        verification["digest"](PROTOCOL / "freeze_manifest.json")
        != extra["protocol_freeze_sha256"]
    ):
        raise ValueError("Protocol differs from the runner preparation record")
    import yaml

    protocol = json.loads((PROTOCOL / "protocol.json").read_text(encoding="utf-8"))
    settings = yaml.safe_load((ROOT / "config.safe-smoke.yaml").read_text())[
        "resources"
    ]
    validate_training_resources(settings)
    probe = protocol["smoke_measurement"]
    if (
        settings["session_minutes"] * 60 != probe["total_active_limit_seconds"]
        or settings["threads"] != 1
    ):
        raise ValueError("Smoke resource settings differ")
    splits = json.loads((ROOT / protocol["partitions"]["primary"]).read_text())
    return protocol, settings, splits


def backend():
    # Configure libraries before importing the training modules in this process.
    os.environ["ESSENZA_CONFIG"] = str(ROOT / "config.safe-smoke.yaml")
    for name in (
        "ESSENZA_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    import numpy as np
    from threadpoolctl import threadpool_limits

    from src.experiments import fit_binary, load_dataset
    from src.resampling import resample

    return {
        "np": np,
        "pool": threadpool_limits,
        "load": load_dataset,
        "fit": fit_binary,
        "resample": resample,
    }


def measure(protocol, splits, guard, operations, result, events, clock=time.perf_counter):
    np = operations["np"]
    probe = protocol["smoke_measurement"]
    fold = splits["folds"][probe["fold"]]
    expected = protocol["dataset"]
    labels = expected["label_order"]
    result["measurements"] = {}
    for algorithm in probe["learner_order"]:
        guard.check(force=True)
        started = clock()
        X, Y, manifest = operations["load"](ROOT / expected["path"])
        guard.check(force=True)
        load_seconds = clock() - started
        if (
            manifest["dataset_id"] != expected["dataset_id"]
            or manifest["feature_schema_id"] != expected["feature_schema_id"]
            or manifest["labels"] != labels
            or X.shape != (expected["development_rows"], expected["features"])
            or Y.shape != (expected["development_rows"], expected["labels"])
        ):
            raise ValueError("Loaded development data differs from frozen protocol")
        record = {
            "dataset_load_seconds": load_seconds,
            "load_seconds": load_seconds + result["backend_import_seconds"],
            "resample_seconds": [],
            "units": [],
        }
        result["measurements"][algorithm] = record
        train, stop, score = (fold[role] for role in ("train", "stop", "score"))
        for strategy in probe["strategies"]:
            guard.check(force=True)
            started = clock()
            xr, yr = operations["resample"](
                X[train],
                Y[train],
                strategy,
                seed=expected["outer_split_seed"] + probe["fold"],
                ratio=0.5,
                n_bits=expected["feature_spec"]["n_bits"],
            )
            guard.check(force=True)
            record["resample_seconds"].append(clock() - started)
            for label in probe["labels"]:
                guard.check(force=True)
                j = labels.index(label)
                unit = {"algorithm": algorithm, "strategy": strategy, "label": label}
                result["active_unit"] = unit
                events.write(json.dumps({"event": "unit_started", **unit}) + "\n")
                events.flush()
                model = None
                started = clock()
                try:
                    model, rounds = operations["fit"](
                        algorithm,
                        dict(probe["probe_params"][algorithm]),
                        xr,
                        yr[:, j],
                        stop=(X[stop], Y[stop, j]),
                        threads=1,
                        seed=expected["outer_split_seed"],
                        weighting=strategy == "class_weight",
                        guard=guard,
                    )
                    probabilities = np.asarray(model.predict_proba(X[score]))
                    guard.check(force=True)
                    elapsed = clock() - started
                    if (
                        probabilities.shape != (len(score), 2)
                        or not np.isfinite(probabilities).all()
                        or np.any((probabilities < 0) | (probabilities > 1))
                        or not np.isfinite(elapsed)
                        or elapsed <= 0
                    ):
                        raise ValueError("Invalid prediction or timing from smoke unit")
                    if not 1 <= rounds <= (2000 if algorithm == "xgb" else 3000):
                        raise ValueError("Invalid boosting rounds")
                finally:
                    del model
                completed = {
                    **unit,
                    "fit_predict_seconds": elapsed,
                    "rounds": int(rounds),
                }
                record["units"].append(completed)
                result["active_unit"] = None
                events.write(
                    json.dumps({"event": "unit_completed", **completed}) + "\n"
                )
                events.flush()
        del X, Y, xr, yr
    guard.check(force=True)
    estimates = {}
    for algorithm in probe["learner_order"]:
        m = result["measurements"][algorithm]
        if len(m["units"]) != 9 or len(m["resample_seconds"]) != 3:
            raise ValueError("All nine completed label fits per learner are required")
        estimates[algorithm] = protocol["budget"]["estimation_factor"] * (
            m["load_seconds"]
            + 3
            * (
                max(m["resample_seconds"])
                + expected["labels"]
                * max(unit["fit_predict_seconds"] for unit in m["units"])
            )
        )
    result["trial_seconds_estimates"] = estimates


def execute(run, telemetry):
    protocol, settings, splits = frozen_inputs()
    run = Path(run).resolve()
    if run.parent != ROOT / "runs" or not run.name.startswith("smoke-"):
        raise ValueError(
            "Use a new direct child runs/smoke-* in the original repository"
        )
    run.mkdir(exist_ok=False)
    result = {
        "status": "STARTING",
        "protocol_id": protocol["protocol_id"],
        "purpose": "Step 4 resource probe only",
        "active_unit": None,
        "trial_seconds_estimates": None,
        "metrics_computed": False,
        "test_arrays_loaded": False,
        "final_trial_count": None,
    }
    guard = ResourceGuard(
        settings,
        telemetry,
        log_path=run / "resources.jsonl",
        metadata={"operation": "smoke", "shared_limit_seconds": 300},
    )
    try:
        with exclusive_run(run), guard:
            guard.check(force=True)
            started = time.perf_counter()
            operations = backend()
            guard.check(force=True)
            result["backend_import_seconds"] = time.perf_counter() - started
            with (
                operations["pool"](limits=1),
                (run / "units.jsonl").open("x", encoding="utf-8") as events,
            ):
                measure(protocol, splits, guard, operations, result, events)
            guard.check(force=True)
        result["status"] = "PROBE_COMPLETED_PENDING_COOLDOWN_REVIEW"
    except ResourceLimit as exc:
        result.update(status="STOP", reason=str(exc), trial_seconds_estimates=None)
    except KeyboardInterrupt:
        result.update(
            status="INTERRUPTED",
            reason="User interrupted",
            trial_seconds_estimates=None,
        )
    except Exception as exc:  # noqa: BLE001 -- Persist failures and return a non-success CLI exit.
        result.update(
            status="FAILED",
            reason=f"{type(exc).__name__}: {exc}",
            trial_seconds_estimates=None,
        )
    finally:
        result["active_seconds"] = time.monotonic() - guard.started
        write_json_atomic(run / "smoke_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    launch = commands.add_parser("run")
    launch.add_argument("--run", type=Path, required=True)
    launch.add_argument("--temperature-file", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        protocol, _, _ = frozen_inputs()
        print(
            json.dumps(
                {
                    "status": "VALIDATED_ONLY",
                    "protocol_id": protocol["protocol_id"],
                    "model_fit_started": False,
                    "dataset_loaded": False,
                }
            )
        )
        return 0
    result = execute(args.run, args.temperature_file)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PROBE_COMPLETED_PENDING_COOLDOWN_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
