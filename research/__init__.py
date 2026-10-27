"""Compatibility import: source utama berada di KODE_ML/research/."""
from pathlib import Path

# Pertahankan `python -m research.experiment` tanpa menduplikasi logika ML.
__path__ = [str(Path(__file__).resolve().parents[1] / "KODE_ML" / "research")]
