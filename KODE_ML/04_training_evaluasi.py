"""Baca delapan kandidat dan hitung ulang metrik test kandidat terpilih."""

import sys

import numpy as np
import pandas as pd

from _lokasi import ROOT, baca_json, hasil_penelitian

sys.path.insert(0, str(ROOT))
from alignment.metrics import binary_metrics, summarize  # noqa: E402
from alignment.experiments import prediction_diagnostics  # noqa: E402
from alignment.experiments_v7 import run_all as training_asli  # noqa: E402


def ringkas_hasil() -> None:
    run = hasil_penelitian()
    tabel = pd.read_csv(run / "comparison_all.csv")
    kolom = ["candidate", "cv_ap_mean", "auprc", "auroc", "macro_f1_0_5"]
    print(tabel[kolom].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    freeze = baca_json(run / "selection_frozen.json")
    kandidat = freeze["overall_selected_from_repeated_grouped_cv"]
    with np.load(run / kandidat / "test_predictions.npz", allow_pickle=False) as saved:
        y = saved["truth"]
        skor = saved["probabilities"]
        keputusan = saved["decisions"]
        labels = saved["labels"].tolist()
    assert np.array_equal(keputusan, (skor >= 0.5).astype(np.int8))
    per_label = {
        nama: binary_metrics(y[:, j], skor[:, j], keputusan[:, j])
        for j, nama in enumerate(labels)
    }
    macro_ap = summarize(per_label, labels)["metrics"]["auprc"]
    macro_f1 = prediction_diagnostics(y, keputusan, labels)["macro_f1"]
    tersimpan = baca_json(run / kandidat / "test_summary.json")
    assert np.isclose(macro_ap, tersimpan["summary"]["metrics"]["auprc"])
    assert np.isclose(macro_f1, tersimpan["diagnostics"]["macro_f1"])
    print(f"\nDipilih lewat validasi: {kandidat}")
    print(f"Test macro AP (hitung ulang): {macro_ap:.6f}")
    print(f"Test macro F1 @ 0,5 (hitung ulang): {macro_f1:.6f}")
    print(f"Kode training asli: {training_asli.__module__}.{training_asli.__name__}")
    print("File ini hanya membaca hasil; tidak menjalankan training ulang.")


if __name__ == "__main__":
    ringkas_hasil()
