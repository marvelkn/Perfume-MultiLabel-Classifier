"""Baca pengaturan dan jejak Optuna; tidak membuat study atau trial baru."""

import sys

import pandas as pd

from _lokasi import PROTOKOL, ROOT, baca_json, hasil_penelitian

sys.path.insert(0, str(ROOT))
from research.tuning import suggest  # noqa: E402


def ringkas_optuna() -> None:
    protokol = baca_json(PROTOKOL)
    run = hasil_penelitian()
    print(f"Pencarian: {protokol['tuning']['sampler']}; objective: {protokol['tuning']['objective']}")
    print(f"Budget per studi: {protokol['tuning']['seconds_per_study'] / 3600:g} jam")
    print("Pencarian hanya untuk C dan D pada XGBoost dan LightGBM.")
    print("Fungsi parameter asli:", f"{suggest.__module__}.{suggest.__name__}")
    for kandidat in ("xgb_C", "xgb_D", "lgbm_C", "lgbm_D"):
        trials = pd.read_csv(run / kandidat / "trials.csv")
        selesai = trials.loc[trials["state"] == "COMPLETE"]
        if selesai.empty:
            raise ValueError(f"Tidak ada trial selesai untuk {kandidat}.")
        terbaik = selesai.loc[selesai["value"].idxmax()]
        print(
            f"{kandidat:<7} trial selesai={len(selesai):>3}, "
            f"trial terbaik={int(terbaik['number']):>3}, "
            f"macro AP CV={terbaik['value']:.6f}"
        )
    print("Detail ruang pencarian ada di research/config/experiment.json.")


if __name__ == "__main__":
    ringkas_optuna()
