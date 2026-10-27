"""Alamat hasil dikunci, bukan pencarian versi terbaru."""
from .runtime import ROOT
ARCHIVED_RUN_ID = "perfume-five-grouped-v7-m2048-d8"

def main_run():
    candidates = (
        ROOT / "runs" / ARCHIVED_RUN_ID,
        ROOT / ".local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v7-m2048-d8/perfume-campus-grouped-v7-m2048-d8/runs" / ARCHIVED_RUN_ID,
    )
    for path in candidates:
        if (path / "complete.json").is_file():
            return path
    raise FileNotFoundError("Hasil utama tidak ditemukan. Jangan training ulang hanya untuk membaca hasil.")
