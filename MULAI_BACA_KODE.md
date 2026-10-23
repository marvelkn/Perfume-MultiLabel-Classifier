# Peta kode ML — mulai baca dari sini

**Mulai dari folder rapi:** [KODE_ML/README.md](KODE_ML/README.md). Di sana ada notebook preprocessing/EDA dan file Python per tahap yang menunjuk implementasi eksperimen asli.

Terakhir diperbarui: **23 Oktober 2026**.

Folder proyek yang dipakai:
`C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier`

**Penelitian utama: V7, Morgan 2048 bit + 8 deskriptor RDKit.**
V9 adalah eksperimen tambahan 1024/2048 + 8 deskriptor dan sudah selesai.
Lima sumber data: GoodScents, IFRA 2019, Leffingwell, Arctander 1960, Sigma-Aldrich 2014.
Dataset utama: **6.686 molekul, 109 label, 5.353 train / 1.333 test**.

## 1. Urutan belajar yang disarankan

| Urutan | Tahap | File yang dibaca | Yang perlu dipahami |
|---|---|---|---|
| 1 | Lihat preprocessing | [notebooks/02_perfume_five_preprocessing.ipynb](notebooks/02_perfume_five_preprocessing.ipynb) | Lima sumber, pencocokan identitas, data kosong, penggabungan label, pengelompokan, pembagian data. **Bagian fitur notebook masih 1024+5; lanjut ke nomor 4 untuk V7.** |
| 2 | Kode preprocessing asli | [alignment/perfume_data.py](alignment/perfume_data.py) | Bagaimana sumber dibaca, identitas molekul disatukan, label dikodekan dan dataset dasar disimpan. |
| 3 | Grup dan split | [alignment/grouped_splits.py](alignment/grouped_splits.py) | Menjaga anggota grup struktur/fitur yang sama tetap bersama ketika split train/test dan train/validasi. |
| 4 | Fitur aktual V7 | [alignment/descriptors_v7.py](alignment/descriptors_v7.py) | SMILES → Morgan **2048** + **8 angka deskriptor** → 2.056 fitur. |
| 5 | Pengaturan eksperimen | [alignment/protocol_v7.json](alignment/protocol_v7.json) | Skenario A–D, ruang tuning, budget, seed, fold, ambang dan aturan pemilihan model. |
| 6 | Pencarian Optuna | [alignment/tuning_v6.py](alignment/tuning_v6.py) | Optuna TPE, usulan hyperparameter, objective AP validasi, batas waktu, SQLite dan resume. **File ini dipakai V7 walaupun namanya V6.** |
| 7 | Proses training dan validasi | [alignment/experiments.py](alignment/experiments.py) | Membuat XGBoost/LightGBM, bobot kelas, fitting, CV, prediksi OOF dan pemilihan ambang. |
| 8 | Pengatur eksperimen utama | [alignment/experiments_v7.py](alignment/experiments_v7.py) | Menjalankan kandidat, validasi lima seed, memilih model, membekukan pilihan, fitting final dan evaluasi test. |
| 9 | Fitting final | [alignment/experiments_v5.py](alignment/experiments_v5.py) | Fungsi `fit_final_models`: training satu pengklasifikasi untuk tiap label dan menyimpan model. **Dipakai V7 dan V9.** |
| 10 | Rumus evaluasi | [alignment/metrics.py](alignment/metrics.py) | Accuracy biner, AUROC, Average Precision, specificity, precision dan recall; penggabungan macro. F1/diagnostik tambahan ada di core eksperimen. |
| 11 | Memakai model V7 | [alignment/predict_v7.py](alignment/predict_v7.py) | SMILES baru → fitur 2048+8 → skor/keputusan untuk 109 label memakai research bundle V7. |
| 12 | Membaca hasil | [analisis V7](reports/grouped_v7_analysis_20260928/HASIL_ANALISIS_V7.md), [analisis V9](reports/grouped_v9_analysis_20261021/HASIL_ANALISIS_V9.md) | Perbandingan kandidat, metrik, ambang dan batas kesimpulan. |

Tidak perlu membaca semua file sekaligus. Mulai **1 → 2 → 3 → 4**, kemudian bahas tuning dan training.

## 2. Fungsi utama di setiap tahap

| File | Fungsi yang dicari |
|---|---|
| `alignment/perfume_data.py` | `load_sources`, `merge_sources`, `encode_and_split`, `extract_features`, `prepare` |
| `src/build_dataset.py` | `_attach_smiles`, `_attach_sigma`, `_cas_bridge`; helper pencocokan identitas yang dipanggil preprocessing utama |
| `alignment/grouped_splits.py` | `structure_feature_groups`, `outer_split`, `inner_folds`, `validate_grouped_split` |
| `alignment/descriptors_v7.py` | `calculate`, `calculate_morgan`, `prepare`, `verify` |
| `alignment/tuning_v6.py` | `suggest`, `initialize`, `tune`, `BudgetLimits` |
| `alignment/experiments.py` | `make_model`, `balanced_sample_weights`, `fit`, `cross_validate`, `select_validation_thresholds`, `repeated_grouped_cv`, `prediction_diagnostics` |
| `alignment/experiments_v7.py` | `select_candidates`, `evaluate_test`, `summaries`, `run_all` |
| `alignment/experiments_v5.py` | `fit_final_models` |
| `alignment/metrics.py` | `binary_metrics`, `summarize` |
| `alignment/predict_v7.py` | `predict` |

Delapan deskriptor tetap: **MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA, NumRotatableBonds, HeavyAtomCount, RingCount**.

Alur utama:
```text
Lima sumber mentah
  → perfume_data.py: gabung/bersihkan identitas dan label
  → grouped_splits.py: grup + train/test + fold
  → descriptors_v7.py: fitur Morgan 2048 + 8
  → tuning_v6.py: cari hyperparameter lewat CV pada train
  → experiments.py: repeated grouped CV dan prediksi OOF
  → experiments_v7.py: pilih kandidat berdasarkan rata-rata CV AP
  → experiments_v5.py: fit final dengan seluruh train
  → metrics.py: evaluasi pada test
  → predict_v7.py: prediksi SMILES baru
```

Di V7, A/C memakai Morgan saja; B/D menambahkan delapan deskriptor.
A/B baseline, C/D tuned. Setiap algoritma mempunyai empat kandidat.
Pilihan konfigurasi memakai AP validasi; ambang utama 0,5, ambang OOF analisis tambahan.
Lima seed adalah pengulangan pembagian validasi, bukan lima pencarian Optuna independen.

## 3. File khusus V9

| File | Kegunaan |
|---|---|
| [alignment/experiments_v9.py](alignment/experiments_v9.py) | Pengatur delapan kandidat: dua algoritma × dua panjang Morgan × baseline/tuned. |
| [alignment/tuning_v9.py](alignment/tuning_v9.py) | Tuning 16 jam/studi, snapshot resep terbaik pada 8/16 jam. Memakai helper `tuning_v6.py`. |
| [alignment/protocol_v9.json](alignment/protocol_v9.json) | Search space, empat budget, seed/fold, fitur dan aturan pemilihan. |
| [alignment/predict_v9.py](alignment/predict_v9.py) | Prediksi dari research bundle V9 sesuai panjang bit terpilih. |
| [alignment/package_campus_v9.py](alignment/package_campus_v9.py) | Membuat folder dan ZIP input PC kampus. |
| [alignment/collect_results_v9.py](alignment/collect_results_v9.py) | Mengumpulkan artefak run menjadi ZIP hasil. |
| [reports/grouped_v9_analysis_20261021/audit_v9.py](reports/grouped_v9_analysis_20261021/audit_v9.py) | Audit hasil, checksum dan hitung ulang metrik; bukan training. |

V9 tetap tambahan, tidak otomatis mengganti V7 atau rilis aplikasi.

## 4. Notebook yang tersedia dan statusnya

| Notebook dalam `notebooks/` | Status |
|---|---|
| `02_perfume_five_preprocessing.ipynb` | **Paling relevan untuk preprocessing lima sumber.** Bagian fitur masih build dasar 1024+5. Fitur aktual V7 ada di `descriptors_v7.py`. |
| `00_preprocessing_audit.ipynb` | Audit versi awal 25 label, 2048+5 dan split lama. Riwayat. |
| `01_eda.ipynb` | EDA lama; sengaja diblokir dari eksekusi agar tidak dianggap pipeline terkini. |
| `02_modeling.ipynb` | Modeling lama; sengaja diblokir. Bukan tempat tuning/training V7 atau V9. |
| `01_suh_alignment_audit.ipynb` | Audit alignment benchmark Suh sebelumnya; bukan dataset utama lima sumber. |
| `03_perfume_data_quality_audit.ipynb` | Audit masalah sebelum perbaikan grouped split. Angka split di outputnya adalah riwayat. |
| `04_suh_v8_preml.ipynb` | Audit sepuluh sumber V8, berhenti sebelum ML. Eksplorasi. |

**Belum ada notebook tuning/training V7 yang mutakhir.** Implementasi eksperimen terkini adalah file Python di `alignment/`. Jangan menjalankan notebook modeling lama untuk mengulang V7.

## 5. Konfigurasi, dataset dan lokasi hasil

| Lokasi | Isi |
|---|---|
| `alignment/perfume_config.json` | Konfigurasi kohor lima sumber/build dasar 1024+5. |
| `alignment/protocol_v7.json` | Pengaturan eksperimen utama V7, termasuk fitur aktual 2048+8. |
| `alignment/campus_compute_profile.json` | Pengaturan CPU dan thread untuk eksperimen kampus. |
| `data/snapshots/perfume-five-v1/` | Salinan sumber mentah yang dibekukan. |
| `data/builds/perfume-five-grouped-v2/` | Records, target, grup, split dan dataset dasar yang sama untuk V7/V9. |
| `data/builds/perfume-m2048-d8-v7/` | `features.npz`, `descriptors.csv`, manifest fitur aktual 2048+8. |
| `v7_reference/` | Ringkasan historis untuk pembanding dan audit; bukan sumber label baru. |
| `reports/grouped_v7_analysis_20260928/` | Analisis utama V7. |
| `reports/grouped_v9_analysis_20261021/` | Analisis tambahan V9 dan perhitungan ulang. |
| `.local-tools/campus-transfer-full-20260909/` | Paket dan hasil yang dibawa kembali dari kampus. |

Hasil V7 yang dibaca saat audit:
```text
.local-tools/campus-transfer-full-20260909/
  perfume-campus-grouped-v7-m2048-d8/
    perfume-campus-grouped-v7-m2048-d8/
      runs/perfume-five-grouped-v7-m2048-d8/
```

Hasil V9:
```text
.local-tools/campus-transfer-full-20260909/
  perfume-campus-grouped-v9-bits-d8/
    perfume-campus-grouped-v9-bits-d8/
      runs/perfume-five-grouped-v9-bits-d8/
```

Di dalam run: `comparison_all.csv` untuk tabel kandidat; `robustness/` untuk CV/OOF;
`selection_frozen.json` untuk keputusan sebelum test; folder kandidat berisi
`recipe.json`, model `.joblib`, prediksi test dan ringkasannya. Kandidat tuned
juga menyimpan SQLite Optuna, trial, sampler dan budget.

## 6. Packaging, pengujian dan aplikasi

- `alignment/prepare_v7.py`: menyiapkan fitur dan referensi beku sebelum paket V7.
- `alignment/package_campus_v7.py`, `collect_results_v7.py`: kemas input/hasil V7.
- `RUN_GROUPED_V7.cmd/.sh`, `RUN_GROUPED_V9.cmd/.sh`: pintasan terminal.
- `alignment/test_grouped_v7.py`, `test_grouped_v9.py`, `test_grouped_splits.py`, `test_perfume.py`: pengujian komponen, bukan training penelitian.
- `v8/prepare.py`: audit sepuluh sumber; bukan alur utama V7.
- `src/inference.py` dan `src/release.py`: helper rilis Python lama V6 **1024+5**. Berbeda dari research bundle V7.
- `api.py` → `api_light.py`: layanan **ekstraksi fitur** `/fingerprint`, belum layanan prediksi model V7.
- `app.py`: antarmuka Gradio ekstraksi fitur.
- ONNX model baru, Android dan deployment masih ditunda.

## 7. File lama yang sengaja dipertahankan

`experiments_v5.py`, `experiments_v4.py`, `tuning_v6.py`, `alignment/data.py`,
beberapa helper `src/`, dan `config.yaml` masih berada dalam rantai import V7/V9.
**Nama versi lama tidak berarti file itu sudah tidak dipakai.**

`config.yaml` berisi konfigurasi historis 25 label dan **bukan protokol eksperimen V7**,
tetapi masih dimuat oleh helper konfigurasi ketika import. Jangan menghapusnya tanpa refactor.

Kode, data, reference, model, Optuna DB, hasil audit dan arsip penelitian dipertahankan.
Pembersihan 23 Oktober difokuskan pada cache, hasil tes sintetis dan salinan environment kampus.
Lihat [catatan cleanup](reports/cleanup_20261023/CLEANUP_SUMMARY.md).

