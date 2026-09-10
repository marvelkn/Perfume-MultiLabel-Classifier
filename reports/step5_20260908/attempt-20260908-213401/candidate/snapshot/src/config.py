"""Configuration reads have no filesystem side effects."""
from pathlib import Path
import os
import yaml
PROJECT_ROOT = Path(__file__).resolve().parents[1]
def load_config(filename=None):
    target = Path(filename or os.environ.get("ESSENZA_CONFIG", PROJECT_ROOT / "config.yaml"))
    cfg = yaml.safe_load(target.read_text(encoding="utf-8"))
    cfg["paths"] = {k: (PROJECT_ROOT / v).resolve() for k, v in cfg["paths"].items()}
    return cfg
CONFIG = load_config()
def path(key, *, create=False):
    target = CONFIG["paths"][key]
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target
