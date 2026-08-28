"""Load central configuration from config.yaml and expose resolved, absolute paths.

Every other module imports CONFIG from here so there is exactly one source of truth
for the seed, fingerprint settings, split ratio, and directory layout.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Read config.yaml once and resolve all `paths` entries to absolute Paths."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {key: (PROJECT_ROOT / rel) for key, rel in cfg["paths"].items()}
    return cfg


CONFIG = load_config()


def path(key: str) -> Path:
    """Return the absolute Path for a configured directory key, creating it if missing."""
    p: Path = CONFIG["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
