"""Lokasi artefak penelitian. Tidak mengubah data atau hasil eksperimen."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "builds" / "perfume-five-grouped-v2"
FITUR = ROOT / "data" / "builds" / "perfume-m2048-d8-v7"
PROTOKOL = ROOT / "research" / "config" / "experiment.json"
RUN_ID = "perfume-five-grouped-v7-m2048-d8"
RUN_KAMPUS = (
    ROOT
    / ".local-tools"
    / "campus-transfer-full-20260909"
    / "perfume-campus-grouped-v7-m2048-d8"
    / "perfume-campus-grouped-v7-m2048-d8"
    / "runs"
    / RUN_ID
)


def baca_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def hasil_penelitian() -> Path:
    """Pilih hasil yang ada; laporan dan data tidak diubah."""
    for path in (ROOT / "runs" / RUN_ID, RUN_KAMPUS):
        if (path / "complete.json").exists():
            return path
    raise FileNotFoundError(
        "Hasil eksperimen utama tidak ditemukan. Periksa folder runs/ atau hasil kampus; "
        "jangan mengulang training hanya untuk membaca kodenya."
    )
