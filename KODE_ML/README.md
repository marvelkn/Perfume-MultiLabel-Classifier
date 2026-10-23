# Kode ML penelitian — mulai dari sini

Folder ini memisahkan **kode yang enak dibaca** menurut urutan penelitian: **lima sumber Pyrfume, Morgan 2048 bit, delapan deskriptor, XGBoost vs LightGBM**. Nama versi pada beberapa file asli hanyalah riwayat teknis pengembangan, bukan nama metode di laporan.

**Pilihan format:** `.ipynb` untuk *preprocessing dan EDA* karena tabel/grafiknya bisa dilihat per langkah. `.py` untuk fitur, split, Optuna, training/evaluasi, dan prediksi karena lebih mudah ditelusuri, diuji, serta dijalankan ulang. Notebook ini membaca artefak dataset yang sudah dibekukan; **tidak membuat dataset atau mengulang training**.

| Urutan | Baca | Isi |
|---|---|---|
| 1 | [00_preprocessing_dan_eda.ipynb](00_preprocessing_dan_eda.ipynb) | Lima sumber, alasan baris ditolak, duplikat molekul, distribusi label, fitur dan pemeriksaan split. Sudah menyimpan output dan grafik. |
| 2 | [01_fitur.py](01_fitur.py) | Contoh SMILES → 2048 bit + 8 deskriptor. |
| 3 | [02_split.py](02_split.py) | Train/test dan contoh lima fold tanpa kebocoran grup. |
| 4 | [03_optuna.py](03_optuna.py) | Ruang/budget tuning dan trial yang benar-benar selesai. |
| 5 | [04_training_evaluasi.py](04_training_evaluasi.py) | Delapan kandidat, pemilihan lewat CV, hitung ulang AP dan F1 test. |
| 6 | [05_prediksi.py](05_prediksi.py) | Contoh prediksi SMILES tunggal dari model penelitian yang tersimpan. |

File `.py` di sini adalah **pintu masuk baca dan inspeksi**. Logika yang benar-benar digunakan saat eksperimen tetap ada di `alignment/`; skrip folder ini memanggil fungsi aslinya supaya tidak muncul dua versi implementasi yang berbeda. Tidak ada data/model yang disalin atau diubah.

## Kode asli per tahap

| Tahap | Implementasi eksperimen asli |
|---|---|
| Ambil lima sumber, cocokkan SMILES/CAS, bersihkan, gabung label, encode target | [perfume_data.py](../alignment/perfume_data.py), [build_dataset.py](../src/build_dataset.py), [label_harmonization.py](../src/label_harmonization.py) |
| Grup struktur/fitur dan split | [grouped_splits.py](../alignment/grouped_splits.py) |
| Bangun fitur penelitian | [descriptors_v7.py](../alignment/descriptors_v7.py) |
| Ruang parameter, aturan perbandingan dan budget | [protocol_v7.json](../alignment/protocol_v7.json) |
| Optuna TPE dan objective CV | [tuning_v6.py](../alignment/tuning_v6.py) — nama file historis; ini fungsi yang dipakai eksperimen utama |
| Bobot kelas, fit per label, lima seed × lima fold, OOF | [experiments.py](../alignment/experiments.py) |
| Jalankan A/B/C/D × dua algoritma, freeze pilihan, evaluasi test | [experiments_v7.py](../alignment/experiments_v7.py), [experiments_v5.py](../alignment/experiments_v5.py) |
| Metrik | [metrics.py](../alignment/metrics.py) |
| Prediksi dari research bundle | [predict_v7.py](../alignment/predict_v7.py) |

**A/C** memakai Morgan 2048 saja; **B/D** memakai Morgan 2048 + 8 deskriptor. **A/B** parameter awal; **C/D** Optuna. Pemilihan model memakai rata-rata macro Average Precision (AP) pada validasi berkelompok. Hasil test dibaca setelah pilihan dibekukan. Ambang utama 0,5; ambang OOF hanya analisis tambahan. Kode training asli bukan aplikasi Android/ONNX.

## Cara membaca hasil tanpa melatih ulang

Dari root repositori pada PowerShell:

```powershell
.\.venv\Scripts\python.exe KODE_ML\01_fitur.py
.\.venv\Scripts\python.exe KODE_ML\02_split.py
.\.venv\Scripts\python.exe KODE_ML\03_optuna.py
.\.venv\Scripts\python.exe KODE_ML\04_training_evaluasi.py
.\.venv\Scripts\python.exe KODE_ML\05_prediksi.py --smiles "COc1cc(C=O)ccc1O" --top 10
```

Notebook bisa langsung **dibuka untuk melihat output yang tersimpan**. Jika ingin menjalankan ulang di VS Code/Jupyter, pilih kernel Python dengan dependensi proyek (`.venv`); bila kernel belum tersedia, pasang `ipykernel` pada environment tersebut. Jalankan notebook dari folder ini atau root repo. Semua langkah hanya membaca `data/builds/`.

Hasil run utama dicari di `runs/perfume-five-grouped-v7-m2048-d8` atau folder hasil kampus yang sudah disalin ke `.local-tools/`. Ringkasan interpretasi ada di [HASIL_ANALISIS_V7.md](../reports/grouped_v7_analysis_20260928/HASIL_ANALISIS_V7.md). **Accuracy tinggi saja tidak cukup** di data yang label positifnya langka; baca AP, F1, precision, recall, dan dukungan per label bersama-sama.
