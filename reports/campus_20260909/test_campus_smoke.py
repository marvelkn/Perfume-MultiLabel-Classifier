"""Synthetic guard tests; no dataset or real learner is used."""
import runpy
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml
from src import runtime

ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(str(Path(__file__).with_name("run_campus_smoke.py")))
SETTINGS = yaml.safe_load((ROOT / "config.safe-smoke.yaml").read_text())["resources"]

@pytest.fixture
def machine(monkeypatch):
    monkeypatch.setattr(runtime.psutil, "virtual_memory", lambda: SimpleNamespace(available=8*1024**3))
    monkeypatch.setattr(runtime.psutil, "Process", lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100), children=lambda **kw: []))


def guard():
    return MODULE["CampusGuard"](MODULE["campus_settings"](SETTINGS))


def test_no_sensor_is_read_and_null_is_logged(machine, monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.ResourceGuard, "temperature", lambda self: pytest.fail("Sensor was read"))
    target = tmp_path / "resources.jsonl"
    with MODULE["CampusGuard"](MODULE["campus_settings"](SETTINGS), log_path=target) as g:
        g.check(force=True)
        assert g.snapshot["cpu_c"] is None
    assert '"require_temperature": false' in target.read_text()


def test_ram_limit_still_stops(machine, monkeypatch):
    monkeypatch.setattr(runtime.psutil, "virtual_memory", lambda: SimpleNamespace(available=1*1024**3))
    with pytest.raises(runtime.ResourceLimit, match="Available RAM"):
        guard().check(force=True)


def test_process_ram_limit_still_stops(machine, monkeypatch):
    monkeypatch.setattr(runtime.psutil, "Process", lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=5*1024**3), children=lambda **kw: []))
    with pytest.raises(runtime.ResourceLimit, match="memory budget"):
        guard().check(force=True)


def test_session_deadline_still_stops(machine):
    g = guard()
    g.started -= 301
    with pytest.raises(runtime.ResourceLimit, match="time budget"):
        g.check(force=True)


def test_default_profile_still_requires_temperature():
    settings = dict(SETTINGS, require_temperature=False)
    with pytest.raises(ValueError):
        runtime.validate_training_resources(settings)


def test_resource_ceiling_cannot_be_relaxed():
    settings = MODULE["campus_settings"](SETTINGS)
    settings["threads"] = 8
    with pytest.raises(ValueError):
        MODULE["CampusGuard"](settings)


def test_temperature_file_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        MODULE["CampusGuard"](MODULE["campus_settings"](SETTINGS), tmp_path/'fake.json')
