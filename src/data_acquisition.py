"""Read a pinned Pyrfume revision; retain byte-identical source snapshots."""
import io
import re
from pathlib import Path
import pandas as pd
import requests
from .config import CONFIG, path
from .artifacts import digest, write_json

def load_all(save_raw=True, *, config=None):
    cfg = config or CONFIG
    revision = cfg["data_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("data_revision must be a full Git commit SHA")
    snapshot = Path(cfg["paths"]["data_raw"]) / revision
    out, manifest = {}, {}
    for source, files in cfg["sources"].items():
        for kind, rel in files.items():
            target = snapshot / rel
            url = f"https://raw.githubusercontent.com/pyrfume/pyrfume-data/{revision}/{rel}"
            if target.exists():
                content = target.read_bytes()
            else:
                response = requests.get(url, timeout=(10, 60))
                response.raise_for_status()
                content = response.content
                if save_raw:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
            import hashlib
            manifest[rel] = {"sha256": hashlib.sha256(content).hexdigest(), "url": url}
            # Identifiers stay strings, never implicit RangeIndex or floats.
            out[f"{source}_{kind}"] = pd.read_csv(io.BytesIO(content), dtype=str)
    if save_raw:
        manifest_path = snapshot / "manifest.json"
        if manifest_path.exists():
            import json
            old = json.loads(manifest_path.read_text())
            if old != manifest:
                raise ValueError("Source snapshot checksum mismatch")
        else:
            write_json(manifest_path, manifest)
    return out, manifest
