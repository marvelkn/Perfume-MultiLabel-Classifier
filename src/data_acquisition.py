"""Load GoodScents, Leffingwell, Arctander, Sigma, and Flavornet archives.

Priority order for loading each file:
  1. Local dataset folder (dataset/pyrfume-data-main/pyrfume-data-main/)
  2. pyrfume library
  3. Raw GitHub fallback

Saves a flat copy of each file into data/raw/ for inspection.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CONFIG, path

_RAW_BASE = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main"

# Map config source keys to the actual local subfolder names
_LOCAL_FOLDER_MAP = {
    "arctander_1960/molecules.csv":         "arctander_1960/molecules.csv",
    "arctander_1960/behavior_1_sparse.csv": "arctander_1960/behavior_1_sparse.csv",
    "arctander_1960/stimuli.csv":           "arctander_1960/stimuli.csv",
    "sigma_2014/molecules.csv":             "sigma_2014/molecules.csv",
    "sigma_2014/behavior.csv":              "sigma_2014/behavior.csv",
    "sigma_2014/stimuli.csv":               "sigma_2014/stimuli.csv",
    "flavornet/molecules.csv":              "flavornet/molecules.csv",
    "flavornet/behavior.csv":               "flavornet/behavior.csv",
    "flavornet/stimuli.csv":                "flavornet/stimuli.csv",
}

_LOCAL_BASE = Path("dataset/pyrfume-data-main/pyrfume-data-main")


def _load_one(rel: str) -> pd.DataFrame:
    """Load a single archive file. Try local folder first, then pyrfume, then GitHub."""
    df = None
    
    # 1. Try local dataset folder (already downloaded)
    local_path = _LOCAL_BASE / rel
    if local_path.exists():
        df = pd.read_csv(local_path)
    else:
        # 2. Try pyrfume library
        try:
            import pyrfume
            df = pyrfume.load_data(rel)
        except Exception as exc:  # noqa: BLE001
            print(f"      pyrfume.load_data({rel!r}) failed ({exc!r}); using raw GitHub")
            # 3. Fallback: raw GitHub
            df = pd.read_csv(f"{_RAW_BASE}/{rel}", index_col=0)
            
    # Pandas 3.0 compatibility: if index has a name, it must be a column to be accessed as df["name"]
    if df is not None and df.index.name is not None:
        df = df.reset_index()
        
    return df


def load_all(save_raw: bool = True) -> dict[str, pd.DataFrame]:
    """Load all configured source files. Keys look like 'goodscents_molecules'."""
    out: dict[str, pd.DataFrame] = {}
    for source, files in CONFIG["sources"].items():
        for kind, rel in files.items():
            df = _load_one(rel)
            out[f"{source}_{kind}"] = df
            if save_raw:
                df.to_csv(path("data_raw") / rel.replace("/", "__"))
            print(f"      {rel:<40} rows={len(df):>6}  cols={list(df.columns)[:6]}")
    return out


if __name__ == "__main__":
    print("Loading all sources ...")
    load_all(save_raw=True)
    print("Done. Raw copies in", path("data_raw"))

