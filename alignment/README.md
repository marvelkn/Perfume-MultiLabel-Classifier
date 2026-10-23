# Peta pipeline penelitian utama

**Mulai dari [MULAI_BACA_KODE.md](../MULAI_BACA_KODE.md)** untuk urutan belajar dan pembedaan notebook lama.

Status 23 Oktober 2026: **V7 dikunci sebagai penelitian utama**, Morgan 2048 + delapan deskriptor.
**V9 selesai sebagai eksplorasi tambahan** 1024/2048 + delapan deskriptor.
Data lima sumber: 6.686 molekul, 109 label, 5.353 train / 1.333 test.

| Tahap | File |
|---|---|
| Preprocessing lima sumber | `perfume_data.py`, `perfume_config.json` |
| Pengelompokan struktur/fitur dan split | `grouped_splits.py` |
| Fitur aktual V7 2048+8 | `descriptors_v7.py` |
| Protokol utama | `protocol_v7.json` |
| Optuna V7 | `tuning_v6.py` |
| Mesin training, CV, OOF dan ambang | `experiments.py` |
| Pengatur V7 | `experiments_v7.py` |
| Fitting final yang dipakai V7/V9 | `experiments_v5.py:fit_final_models` |
| Metrik | `metrics.py` |
| Prediksi research bundle V7 | `predict_v7.py` |
| Pengatur dan Optuna V9 | `experiments_v9.py`, `tuning_v9.py`, `protocol_v9.json` |
| Paket/hasil kampus | `package_campus_v7.py`, `collect_results_v7.py` dan pasangan `v9` |

Notebook [02_perfume_five_preprocessing.ipynb](../notebooks/02_perfume_five_preprocessing.ipynb)
menjelaskan kohor dan grup lima sumber, tetapi bagian fiturnya masih dasar 1024+5.
Ekstensi penelitian utama berada pada `descriptors_v7.py`; belum ada notebook
Optuna/training V7 yang mutakhir.

Kode versi lama masih dipanggil sebagai helper. Jangan menghapusnya berdasarkan nama versi.
Grup yang sama tidak tersebar antarpartisi; kebijakan ini bukan scaffold split.
Ambang utama 0,5; ambang tambahan dipilih dari OOF pada train, bukan dari test.
Hasil historis tetap eksploratori karena test pernah diperiksa.

Data dasar: `../data/builds/perfume-five-grouped-v2/`.
Fitur V7: `../data/builds/perfume-m2048-d8-v7/`.
Analisis: [V7](../reports/grouped_v7_analysis_20260928/HASIL_ANALISIS_V7.md)
dan [V9](../reports/grouped_v9_analysis_20261021/HASIL_ANALISIS_V9.md).

