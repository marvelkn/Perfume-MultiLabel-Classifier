# Peta kode penelitian dan lokasi hasil

**Buka [BELAJAR_KODE_ML_LENGKAP.ipynb](BELAJAR_KODE_ML_LENGKAP.ipynb) lebih dulu.** Ada penjelasan library, baris kode, input/output, contoh hasil, dan latihan sidang. Training berat tampil sebagai Markdown; Run All hanya membaca artefak, memeriksa data, menghitung fitur/metrik, dan inferensi tersimpan.

## Alur yang dipelajari

Lima sumber → cocokkan identitas → bersihkan/petakan label → gabungkan molekul identik → baca 109 target dan split beku → Morgan 2048 + 8 → XGBoost/LightGBM A–D → Optuna → repeated grouped CV/OOF → bekukan pilihan → final fit → test → prediksi satu molekul.

## File belajar

| File | Kerjanya | Output |
|---|---|---|
| [BELAJAR_KODE_ML_LENGKAP.ipynb](BELAJAR_KODE_ML_LENGKAP.ipynb) | Panduan utama lengkap | Tabel, grafik, dan prediksi di notebook |
| [_lokasi.py](_lokasi.py) | Menentukan alamat data/fitur/protokol/hasil | Dipakai skrip lain |
| [01_fitur.py](01_fitur.py) | Contoh SMILES ke fitur | Terminal |
| [02_split.py](02_split.py) | Cek train/test dan contoh fold satu label | Terminal; bukan audit semua seed |
| [03_optuna.py](03_optuna.py) | Baca parameter dan trials.csv tersimpan | Terminal; tidak membuat trial |
| [04_training_evaluasi.py](04_training_evaluasi.py) | Baca delapan kandidat, hitung ulang metrik kandidat terpilih | Terminal; tidak fitting |
| [05_prediksi.py](05_prediksi.py) | Gunakan 109 model tersimpan | Label, skor, keputusan |
| [PANDUAN_SIDANG_KODE.md](PANDUAN_SIDANG_KODE.md) | Latihan menjelaskan metode | Tanya-jawab |

Notebook preprocessing terpisah yang isinya duplikat sudah diarsipkan. Bagian 3 notebook utama memuat pembersihan dan distribusi label.

## Source aktif yang sebenarnya

| Tahap | File | Fungsi penting |
|---|---|---|
| Identitas/normalisasi | [source_identity.py](../research/source_identity.py) | `_attach_smiles`, `_attach_sigma`, `_cas_bridge`, `normalize` |
| Cleaning | [preprocessing.py](../research/preprocessing.py) | `load_sources`, `map_labels`, `merge_sources`, `review_frozen_preprocessing` |
| Data beku | [dataset.py](../research/dataset.py) | `verify`, `load_partition`, `encode_frozen_records` |
| Fitur | [features.py](../research/features.py) | `calculate_morgan`, `calculate`, `features` |
| Fold | [splits.py](../research/splits.py) | `inner_folds`, `validate_grouped_split` |
| Training | [training.py](../research/training.py) | `make_model`, `balanced_sample_weights`, `fit`, `fit_final_models` |
| Tuning | [tuning.py](../research/tuning.py) | `suggest`, `tune` |
| CV/OOF | [validation.py](../research/validation.py) | `cross_validate`, `repeated_grouped_cv`, `select_validation_thresholds` |
| Urutan eksperimen | [experiment.py](../research/experiment.py) | `preflight`, `select_candidates`, `run_all` |
| Evaluasi | [metrics.py](../research/metrics.py) | `binary_metrics`, `summarize` |
| Prediksi | [inference.py](../research/inference.py) | `predict` |
| Bukti/lokasi | [runtime.py](../research/runtime.py), [results.py](../research/results.py) | JSON/checksum/versi, `main_run` |

Tidak ada import dari alignment/experiments_v5, tuning_v6, atau src pada jalur aktif. Fungsi yang masih diperlukan telah dipindah; algoritma intinya dipertahankan.

## Konfigurasi dan data

| File/folder | Isinya |
|---|---|
| [sources.json](../research/config/sources.json) | Lima sumber, commit Pyrfume, aturan label, keterangan split historis |
| [dataset.json](../research/config/dataset.json) | SHA-256 file beku, ukuran dan alamat dataset/fitur |
| [experiment.json](../research/config/experiment.json) | 2048+8, A–D, search space, 8 jam/studi, 5 seed × 5 fold, evaluasi |
| [compute.json](../research/config/compute.json) | Profil CPU penelitian |
| [Snapshot](../data/snapshots/perfume-five-v1/8054ea98ed675005ec10e67359902f500e4911b0/) | CSV sumber mentah; tidak diunduh ulang |
| [Kohor beku](../data/builds/perfume-five-grouped-v2/) | records.csv, target, grup, split, audit dan provenance |
| [Fitur aktual](../data/builds/perfume-m2048-d8-v7/features.npz) | 6.686 × 2048 Morgan dan 6.686 × 8 deskriptor |

`load_partition` mengambil `truth`, `row_index`, `source_row` dari NPZ kohor. Ia **tidak mengambil** `morgan`/`descriptors` historis di NPZ itu. X diambil dari bundle aktual pada indeks yang sama. Tidak ada split baru saat membaca dataset.

## Lokasi hasil utama

Hasil saat ini berada di:

```text
.local-tools/campus-transfer-full-20260909/
  perfume-campus-grouped-v7-m2048-d8/
    perfume-campus-grouped-v7-m2048-d8/
      runs/perfume-five-grouped-v7-m2048-d8/
```

Ini alamat arsip nyata, bukan versi metode yang perlu ditulis dalam judul skripsi. `hasil_penelitian()` juga menerima salinan lengkap di `runs/perfume-five-grouped-v7-m2048-d8` bila tersedia; bukan memilih run terbaru.

| Artefak di folder hasil | Untuk apa |
|---|---|
| `comparison_all.csv` | Delapan kandidat, CV/test dan F1 |
| `selection_frozen.json` | Pilihan dari validasi dan checksum sebelum test |
| `inference_bundle.json` | Urutan fitur/label, model terpilih, ambang utama |
| `lgbm_D/models.json`, `label_*.joblib` | Peta 109 aroma ke classifier tersimpan |
| `xgb_C`, `xgb_D`, `lgbm_C`, `lgbm_D` / `trials.csv` | Jejak Optuna |
| `*/recipe.json` | Parameter tiap kandidat |
| `*/test_predictions.npz` | Truth, skor dan keputusan test |
| `robustness/*/seed_*/` | Fold dan hasil CV setiap seed |
| `robustness/*/pooled_oof_predictions.npz` | Rata-rata skor OOF lima seed |
| `robustness/*/thresholds.json` | Ambang tambahan pilihan OOF |

AP trial terbaik, mean AP lima seed, dan AP test adalah tiga hasil berbeda. Ambang utama 0,5; ambang OOF hanya tambahan. `--top 10` pada prediksi membatasi tampilan, bukan memaksa sepuluh label positif.

## Batas kesimpulan

Nol pada target berarti tidak tercatat, bukan pasti negatif sensoris. Ada ketimpangan kelas. Lima sumber tidak menjamin seluruh molekul eksklusif parfum. Grup historis memakai kesamaan struktur tanpa stereo atau Morgan 1024 identik; belum berarti semua molekul mirip dikelompokkan. Test pernah dilihat; repeated CV bukan nested CV. Tidak ada klaim signifikan hanya dari selisih kecil atau lima seed.

Backup/arsip source: [archive/README.md](../archive/README.md). Hasil V7 dan V9 tetap utuh. ONNX/Android ditunda.
