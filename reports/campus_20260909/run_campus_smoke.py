"""Campus-only resource amendment: real smoke fitting without CPU telemetry."""
import argparse
import copy
import hashlib
import json
import platform
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.runtime import ResourceGuard, validate_training_resources

PROFILE = "campus-smoke-no-temperature-v1"


def campus_settings(original):
    validate_training_resources(original)
    settings = copy.deepcopy(original)
    settings.update(profile=PROFILE, require_temperature=False,
                    temperature_monitoring="disabled_by_user_for_campus")
    return settings


class CampusGuard(ResourceGuard):
    def __init__(self, settings, temperature_file=None, **kwargs):
        if settings.get("profile") != PROFILE or settings.get("require_temperature") is not False:
            raise ValueError("Campus guard requires its explicit resource profile")
        if temperature_file is not None:
            raise ValueError("No temperature file is accepted in no-telemetry mode")
        bounded = dict(settings, profile="safe-smoke", require_temperature=True)
        validate_training_resources(bounded)
        super().__init__(settings, None, **kwargs)

    def temperature(self):
        # Missing temperature is explicit evidence, never a fabricated safe reading.
        self.sensor_age_seconds = None
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--campus-no-temperature", required=True, action="store_true")
    args = parser.parse_args()
    amendment_path = Path(__file__).with_name("campus_no_temperature_amendment.json")
    amendment = json.loads(amendment_path.read_text())
    if platform.system() != "Windows" or platform.node().casefold() == amendment["excluded_laptop_host"].casefold():
        raise ValueError("This launcher is only for the campus Windows PC, not the original laptop")
    verifier = runpy.run_path(str(ROOT / "protocols/essenza_v1_20260908/verify_protocol.py"))
    verifier["verify"]()
    verifier["validate_files"](ROOT, amendment["files"])
    # Verify the original probe before substituting only the resource policy.
    module = runpy.run_path(str(ROOT / "reports/step4_20260908/run_smoke_probe.py"))
    execute = module["execute"]
    namespace = execute.__globals__
    original_inputs = namespace["frozen_inputs"]
    def amended_inputs():
        protocol, settings, splits = original_inputs()
        return protocol, campus_settings(settings), splits
    namespace["frozen_inputs"] = amended_inputs
    namespace["ResourceGuard"] = CampusGuard
    target = args.run.resolve()
    if target.parent != ROOT / "runs" or not target.name.startswith("smoke-campus-") or target.exists():
        raise ValueError("Use a NEW direct child runs/smoke-campus-*")
    result = execute(target, None)
    result.update(resource_amendment=amendment["id"],
                  resource_amendment_sha256=hashlib.sha256(amendment_path.read_bytes()).hexdigest(),
                  temperature_monitoring=False, thermal_safety_assessed=False,
                  host=platform.node(), platform=platform.platform())
    if result["status"] == "PROBE_COMPLETED_PENDING_COOLDOWN_REVIEW":
        result["status"] = "PROBE_COMPLETED_CAMPUS_NO_TEMPERATURE"
    namespace["write_json_atomic"](target / "smoke_result.json", result)
    namespace["write_json_atomic"](target / "campus_resource_amendment.json", amendment)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PROBE_COMPLETED_CAMPUS_NO_TEMPERATURE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
