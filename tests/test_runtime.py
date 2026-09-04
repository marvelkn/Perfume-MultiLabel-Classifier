import json
from types import SimpleNamespace
import pytest
from src.runtime import ResourceGuard, ResourceLimit
import src.runtime as runtime

SETTINGS={"session_minutes":30,"min_available_ram_gb":1,"max_process_ram_gb":1,
          "require_temperature":True,"stop_temperature_c":90,"hot_seconds":30}
@pytest.fixture
def machine(monkeypatch):
    now=[0.]
    monkeypatch.setattr(runtime.time,"monotonic",lambda:now[0])
    monkeypatch.setattr(runtime.psutil,"virtual_memory",lambda:SimpleNamespace(available=2*1024**3))
    process=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=100),children=lambda **kw:[])
    monkeypatch.setattr(runtime.psutil,"Process",lambda:process)
    return now

def test_missing_temperature_blocks_real_training(machine,monkeypatch):
    guard=ResourceGuard(SETTINGS)
    monkeypatch.setattr(guard,"temperature",lambda:None)
    with pytest.raises(ResourceLimit,match="unavailable"): guard.check()

def test_sustained_heat_and_cool_reset(machine,monkeypatch):
    guard=ResourceGuard(SETTINGS)
    temp=[91]
    monkeypatch.setattr(guard,"temperature",lambda:temp[0])
    guard.check();machine[0]=29;guard.check()
    temp[0]=80;machine[0]=30;guard.check()
    temp[0]=91;machine[0]=31;guard.check()
    machine[0]=62
    with pytest.raises(ResourceLimit,match="Sustained"): guard.check()

def test_memory_and_time_limits(machine,monkeypatch):
    guard=ResourceGuard({**SETTINGS,"require_temperature":False})
    monkeypatch.setattr(runtime.psutil,"virtual_memory",lambda:SimpleNamespace(available=100))
    with pytest.raises(ResourceLimit,match="RAM"):guard.check()
    machine[0]=1801
    with pytest.raises(ResourceLimit,match="time budget"):guard.check()

def test_stale_sensor_cannot_bypass_guard(machine,tmp_path):
    p=tmp_path/"sensor.json";p.write_text(json.dumps({"timestamp":1,"cpu_c":40}))
    with pytest.raises(ResourceLimit,match="stale"):ResourceGuard(SETTINGS,p).check()
