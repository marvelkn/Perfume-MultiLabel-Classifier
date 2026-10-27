# Penelitian ML aroma molekul — Essenza

**Mulai belajar:** [KODE_ML/README.md](KODE_ML/README.md) — delapan notebook bertahap.
**Kode utama:** [KODE_ML/research/](KODE_ML/research/). **Peta file:** [KODE_ML/README.md](KODE_ML/README.md).

Penelitian membandingkan XGBoost dan LightGBM untuk memprediksi 109 label aroma dari satu molekul. Lima sumber Pyrfume: GoodScents, IFRA 2019, Leffingwell, Arctander 1960, Sigma-Aldrich 2014. Sumber ini relevan untuk konteks parfum, tetapi tidak menjamin setiap molekul hanya digunakan dalam parfum.

| Skenario | Fitur | Parameter |
|---|---|---|
| A | Morgan 2048 | Baseline 100 putaran |
| B | Morgan 2048 + 8 deskriptor | Baseline 100 putaran |
| C | Morgan 2048 | Optuna |
| D | Morgan 2048 + 8 deskriptor | Optuna |

Setiap algoritma menjalankan A–D. Satu kandidat berisi 109 classifier biner. Morgan memakai radius 2, tanpa chirality. Delapan deskriptor: MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA, NumRotatableBonds, HeavyAtomCount, RingCount.

## Status

Hasil utama **sudah selesai dan dikunci sebagai V7**. V9 adalah eksperimen tambahan, bukan pengganti otomatis. Kandidat utama terpilih LightGBM D: mean CV AP 0,299731; test AP 0,310531; macro F1 pada ambang 0,5 sekitar 0,293156. AP bukan accuracy. Hasil masih eksploratori karena test sudah pernah dilihat dalam pengembangan dan CV bukan nested CV.

Source dirapikan pada 25 Oktober dan dipusatkan di KODE_ML pada 26 Oktober 2026. Hasil lama tidak ditulis ulang dan tidak diklaim dibuat oleh source baru. ONNX dan Android model baru ditunda. API lama dan folder src tetap untuk kompatibilitas, di luar jalur penelitian utama; kontraknya belum diubah.

## Menjalankan contoh ringan

Dari root repo, gunakan Python environment penelitian:

```powershell
.\.venv\Scripts\python.exe -B -m research.experiment --check
```

Linux: ganti `.\.venv\Scripts\python.exe` dengan `.venv/bin/python`.
Notebook membutuhkan kernel Python penelitian dengan ipykernel, nbformat, pandas dan Matplotlib. Paket ML dikunci dalam [requirements_research.txt](requirements_research.txt); runtime notebook terpisah dari definisi metode ML.

Semua perintah di atas hanya pemeriksaan, membaca hasil, atau inferensi. CLI `research.experiment` tanpa `--train` juga hanya memeriksa input. Training memerlukan tugas terpisah dan folder run baru; source ini menolak melanjutkan arsip lama.

## Data dan bukti tetap

- Kohor, target, indeks, grup dan split: `data/builds/perfume-five-grouped-v2/`.
- **X aktual:** `data/builds/perfume-m2048-d8-v7/features.npz`.
- Hasil utama ditemukan oleh [KODE_ML/research/results.py](KODE_ML/research/results.py); alamat lengkap ada di notebook.
- Audit hasil utama: [HASIL_ANALISIS_V7.md](reports/grouped_v7_analysis_20260928/HASIL_ANALISIS_V7.md).
- Audit perapian: [REFACTOR_REPORT.md](reports/research_cleanup_20261025/REFACTOR_REPORT.md).
- Source lama: [archive/README.md](archive/README.md).

Grup historis berasal dari struktur tanpa stereo atau fingerprint Morgan 1024 yang identik. Loader aktif membaca grup/split tersebut, tidak menghitung fitur 1024+5. Ia memeriksa checksum, identitas baris, seluruh fold, serta fitur 2048+8. Mengganti aturan grup membutuhkan rancangan dan eksperimen baru.

`config.yaml`, `feature_spec.json`, `src/` dan API di root adalah jalur kompatibilitas aplikasi lama, bukan konfigurasi penelitian utama. Tes ringan utama: `python -B -m pytest KODE_ML/research/tests/test_research_readonly.py -q -p no:cacheprovider`. Sebagian tes lama melakukan fitting; jangan jalankan seluruh suite untuk audit baca-saja.

Audit label dan hasil perapian terbaru: [KODE_ML/HASIL_VERIFIKASI.md](KODE_ML/HASIL_VERIFIKASI.md). `research/__init__.py` di root hanya penghubung import ke source dalam KODE_ML.
