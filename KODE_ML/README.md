# Kode ML penelitian — penjelasan file dan lokasi hasil

**Mulai belajar dari [BELAJAR_KODE_ML_LENGKAP.ipynb](BELAJAR_KODE_ML_LENGKAP.ipynb).** Notebook panduan ini menggabungkan penjelasan per bagian, kode asli, contoh output, lokasi hasil, dan latihan sidang. Terdapat 13 bagian dan 30 sel contoh yang sudah dijalankan; output tabel, grafik, dan prediksi tersimpan. Kode training berat ditampilkan sebagai Markdown untuk dipelajari, sehingga Run All tidak memulai training ulang.

Folder ini berada di `Perfume-MultiLabel-Classifier/KODE_ML`. Gunakan sebagai urutan belajar metode penelitian: lima sumber Pyrfume, Morgan 2048 bit, delapan deskriptor RDKit, XGBoost vs LightGBM, dan Optuna.

**Penting:** file di sini mendemonstrasikan fitur, membaca pembagian data, membaca hasil tuning, menghitung ulang evaluasi, dan memakai model tersimpan. **Tidak ada skrip dalam folder ini yang menjalankan training lengkap.** Implementasi preprocessing/training asli tetap di `alignment/`, ditautkan di bagian akhir.

Untuk persiapan tanya jawab, baca juga [PANDUAN_SIDANG_KODE.md](PANDUAN_SIDANG_KODE.md). Nama versi pada beberapa file sumber/hasil hanyalah identitas arsip, bukan nama metode di laporan.

## 1. Semua file di folder ini

| File | Fungsinya | Hasil terlihat di mana? |
|---|---|---|
| [BELAJAR_KODE_ML_LENGKAP.ipynb](BELAJAR_KODE_ML_LENGKAP.ipynb) | Panduan lengkap seluruh alur, kode asli, contoh dan latihan | Penjelasan serta output tersimpan dalam notebook |
| [README.md](README.md) | Penjelasan file, urutan belajar, lokasi data/hasil dan perintah menjalankan | Dokumen ini |
| [_lokasi.py](_lokasi.py) | Menentukan root proyek, folder data, fitur, protokol, dan hasil penelitian | Tidak menghasilkan output sendiri; dipakai skrip lain |
| [00_preprocessing_dan_eda.ipynb](00_preprocessing_dan_eda.ipynb) | Menelusuri hasil preprocessing dan melihat statistik/grafik dataset | Output sel dan grafik di notebook |
| [01_fitur.py](01_fitur.py) | Mengubah satu SMILES menjadi Morgan dan delapan deskriptor | Terminal |
| [02_split.py](02_split.py) | Memeriksa train/test dan contoh fold dari satu label | Terminal |
| [03_optuna.py](03_optuna.py) | Membaca pengaturan dan riwayat trial yang telah selesai | Terminal |
| [04_training_evaluasi.py](04_training_evaluasi.py) | Membandingkan hasil delapan kandidat dan menghitung ulang AP/F1 kandidat terpilih | Terminal |
| [05_prediksi.py](05_prediksi.py) | Memuat model tersimpan dan memprediksi satu SMILES | Terminal |
| [PANDUAN_SIDANG_KODE.md](PANDUAN_SIDANG_KODE.md) | Audit kerumitan, konsep inti, contoh penyederhanaan dan 20 latihan pertanyaan | Dokumen |

Semua skrip Python tersebut tidak menyimpan hasil prediksi/CSV baru. Jika ingin menyimpan perubahan output notebook setelah menjalankannya kembali, simpan notebook dari editor.

## 2. _lokasi.py — alamat yang dipakai skrip

`ROOT = Path(__file__).resolve().parents[1]` mengambil root repositori dari letak file. Jadi alamat root tidak bergantung pada nama akun Windows tertentu.

| Variabel/fungsi | Makna |
|---|---|
| `ROOT` | Folder utama `Perfume-MultiLabel-Classifier` |
| `DATA` | Dataset beku: [perfume-five-grouped-v2](../data/builds/perfume-five-grouped-v2/) |
| `FITUR` | Fitur aktual Morgan 2048 + 8: [perfume-m2048-d8-v7](../data/builds/perfume-m2048-d8-v7/) |
| `PROTOKOL` | [protocol_v7.json](../alignment/protocol_v7.json): aturan eksperimen |
| `RUN_ID` | Nama folder hasil yang digunakan penelitian utama |
| `RUN_KAMPUS` | Lokasi hasil yang dikembalikan dari PC kampus |
| `baca_json(path)` | Membaca file JSON menjadi struktur data Python, biasanya dictionary/list |
| `hasil_penelitian()` | Mencari folder run yang memiliki `complete.json`, kemudian mengembalikan alamatnya |

Urutan pencariannya: `ROOT/runs/RUN_ID` dahulu, kemudian `RUN_KAMPUS`. Ini bukan pencarian otomatis versi terbaru. Saat diperiksa 23 Oktober 2026, lokasi pertama tidak berisi run selesai, sehingga hasil yang dipakai adalah folder kampus di bagian 9.

Tidak perlu menjalankan `_lokasi.py` sendiri. Ketika skrip lain melakukan `from _lokasi import ...`, mereka mengambil alamat/fungsi dari sini.

## 3. Notebook — membaca hasil preprocessing dan EDA

Notebook memiliki 22 sel: 11 penjelasan Markdown dan 11 sel kode. Sel kode sebaiknya dibaca/dijalankan berurutan karena sel berikutnya menggunakan variabel dari sel sebelumnya.

| Bagian | Apa yang dilakukan kode? | File yang dibaca |
|---|---|---|
| Setup | Mencari root repo, mengimpor NumPy/pandas/Matplotlib dan menentukan folder | Protokol dipakai sebagai penanda root |
| Jejak seleksi sumber | Membaca jumlah baris diterima/ditolak dan membuat grafik batang | `audit.json` |
| Cleaning dan duplikat | Memeriksa SMILES kosong, duplikat, label, dan jumlah sumber per molekul | `records.csv` dan audit |
| Target dan distribusi | Memuat target train/test dan menampilkan label sering/jarang | `dataset_manifest.json`, `label_selection.csv`, `train.npz`, `test.npz` |
| Fitur aktual | Memeriksa bentuk Morgan/deskriptor, urutan nama fitur, dan satu contoh hasil RDKit | `features.npz` di folder `FITUR` |
| Grup/split | Memastikan tidak ada group ID yang muncul pada train dan test | `groups.csv` serta indeks baris dari NPZ |
| Checks | Memeriksa ukuran data, target, fitur, duplikat dan overlap yang diharapkan | Variabel yang sudah dibaca |

Hasil yang tersimpan menunjukkan:

- 13.337 rekaman sumber; 11.777 diterima sebelum penggabungan identitas; 1.560 ditolak.
- Setelah penggabungan identitas: 6.686 molekul. Sebanyak 2.808 molekul tercatat pada lebih dari satu sumber.
- 5.353 train dan 1.333 test; 109 label dari 184 istilah IFRA.
- Morgan berbentuk `(6686, 2048)`; deskriptor `(6686, 8)`.
- Dua grafik: baris diterima/ditolak per sumber dan 20 label paling sering pada train.

Notebook **tidak membersihkan data mentah ulang**. Ia menjelaskan dan memeriksa hasil cleaning yang sudah tersimpan. Kode cleaning asli ada di [perfume_data.py](../alignment/perfume_data.py).

Catatan: `records.csv` menyimpan label sebelum pemilihan 109 target. Karena itu “tidak ada baris tanpa label” pada records tidak bertentangan dengan 11 baris train yang menjadi seluruhnya nol setelah label yang tidak dipilih dikeluarkan dari target.

## 4. 01_fitur.py — satu SMILES menjadi fitur

Fungsi utama: `periksa_fitur(smiles)`.

```python
morgan = calculate_morgan([smiles])
deskriptor = calculate([smiles])
```

Tanda `[smiles]` membungkus satu string menjadi daftar berisi satu molekul karena fungsi aslinya menerima banyak molekul. Hasilnya `(1, 2048)` dan `(1, 8)`.

Kode kemudian:

1. Menampilkan SMILES dan jumlah bit aktif. Bit aktif berarti posisi fingerprint bernilai 1, bukan jumlah label aroma.
2. Memasangkan nama deskriptor dan nilainya memakai `zip`.
3. Memanggil `features` untuk menunjukkan A/C mempunyai 2048 fitur dan B/D 2056 fitur. Nama `xgb_A` pada panggilan ini hanya memilih susunan fitur; **tidak membuat atau melatih XGBoost**.
4. Memastikan Morgan hanya berisi 0 dan 1.

Default SMILES adalah vanilin: `COc1cc(C=O)ccc1O`. Contoh hasil yang sebelumnya dijalankan: 25 bit aktif, `MolWt≈152.149`, `MolLogP≈1.2133`, `TPSA≈46.53`. Delapan deskriptor dihitung dengan fungsi yang sudah ditetapkan, bukan dipilih otomatis oleh RDKit.

| Deskriptor | Informasi yang diringkas |
|---|---|
| `MolWt` | Berat molekul |
| `MolLogP` | Perkiraan kecenderungan larut di lingkungan lemak dibanding air |
| `NumHDonors` | Jumlah donor ikatan hidrogen |
| `NumHAcceptors` | Jumlah akseptor ikatan hidrogen |
| `TPSA` | Luas permukaan polar berdasarkan struktur |
| `NumRotatableBonds` | Jumlah ikatan yang dapat berotasi |
| `HeavyAtomCount` | Jumlah atom selain hidrogen |
| `RingCount` | Jumlah cincin |

Morgan adalah pola bit struktur, bukan konversi karakter SMILES ke kode biner. Delapan nilai tambahannya berupa angka, bukan delapan bit biner.

Output hanya tercetak di terminal. File dataset dan fitur seluruh molekul tidak ditimpa.

## 5. 02_split.py — memeriksa pembagian yang sudah ada

Fungsi utama: `periksa_split()`.

Kode membaca `splits.json`, mengambil indeks train/test, lalu memeriksa:

```python
bocor = set(kelompok[train]) & set(kelompok[test])
```

`kelompok[train]` mengambil group ID milik baris train. `set` mengambil anggota unik; `&` mencari irisan. Jika irisan tidak kosong, ada grup yang muncul di kedua sisi dan kode berhenti dengan error.

Setelah itu skrip mengambil **label pertama** dari daftar label, saat ini `acidic`, lalu menampilkan ukuran lima fold dan memeriksa overlap grupnya. Output contoh fold pertama: 4.266 baris latih dan 1.087 validasi; seluruh data test tetap terpisah.

**Cakupan pemeriksaan:** semua train/test, tetapi fold yang diperiksa hanya contoh satu label dari split awal. Skrip hanya menampilkan daftar lima seed dari protokol; tidak membuat ulang fold untuk seluruh seed atau mengaudit seluruh 109 label. Pemeriksaan lengkap ada pada pipeline/audit penelitian.

Tidak ada split baru atau training yang dibuat.

## 6. 03_optuna.py — membaca hasil pencarian parameter

Fungsi utama: `ringkas_optuna()`.

Kode membaca protokol untuk menunjukkan TPE, objective AP, dan budget delapan jam per studi. Lalu membaca `trials.csv` untuk `xgb_C`, `xgb_D`, `lgbm_C`, dan `lgbm_D`.

```python
selesai = trials.loc[trials["state"] == "COMPLETE"]
terbaik = selesai.loc[selesai["value"].idxmax()]
```

Baris pertama mempertahankan trial yang selesai. Baris kedua mencari baris dengan skor tertinggi. Trial gagal/terputus tidak dipilih. Nomor trial mulai dari nol.

| Studi | Trial selesai | Nomor trial terbaik |
|---|---:|---:|
| XGBoost C | 12 | 2 |
| XGBoost D | 12 | 2 |
| LightGBM C | 104 | 56 |
| LightGBM D | 74 | 73 |

Import `suggest` di skrip ini hanya dipakai untuk menampilkan lokasi fungsi asli. **Tidak ada `study.optimize`, `study.ask`, atau training yang dijalankan oleh skrip pembaca ini.** Pencarian sebenarnya ada di [tuning_v6.py](../alignment/tuning_v6.py).

Jangan tertukar tiga skor LightGBM D:

| Angka | Maknanya |
|---|---|
| AP sekitar 0,301979 | Objective trial Optuna terbaik pada split tuning |
| AP sekitar 0,299731 | Rata-rata AP validasi berulang lima seed |
| AP sekitar 0,310531 | AP pada test |

Skornya berbeda karena tahap evaluasi dan pembagiannya berbeda; bukan otomatis kesalahan hitung.

## 7. 04_training_evaluasi.py — membaca tabel dan memeriksa metrik

Fungsi utama: `ringkas_hasil()`. Meskipun namanya memuat “training”, **skrip ini tidak melatih model**. Import `training_asli` hanya dipakai untuk menampilkan lokasi runner sebenarnya.

Delapan kandidat berasal dari dua algoritma, masing-masing dengan empat skenario:

| Skenario | Fitur | Parameter |
|---|---|---|
| A | Morgan 2048 | Baseline, tanpa pencarian Optuna |
| B | Morgan 2048 + 8 deskriptor | Baseline, tanpa pencarian Optuna |
| C | Morgan 2048 | Hasil pencarian Optuna |
| D | Morgan 2048 + 8 deskriptor | Hasil pencarian Optuna |

Contohnya `lgbm_D` berarti LightGBM, fitur 2056 kolom, dengan parameter hasil tuning. Baseline menggunakan `n_estimators=100` dan pengaturan implementasi proyek; bukan berarti seluruh opsi library dibiarkan tanpa pengaturan. Penanganan ketimpangan menggunakan bobot yang dihitung dari partisi latih, termasuk pada baseline.

Urutan kerja:

1. Baca `comparison_all.csv` dan tampilkan delapan kandidat.
2. Baca `selection_frozen.json` untuk mengetahui kandidat yang dipilih dari validasi, saat ini `lgbm_D`.
3. Muat `test_predictions.npz` kandidat tersebut. `truth` adalah target, `probabilities` skor, `decisions` keputusan 0/1, dan `labels` urutan nama label.
4. Pastikan keputusan tersimpan sama dengan `skor >= 0.5`.
5. Hitung metrik setiap label menggunakan `binary_metrics`, lalu hitung macro AP dan macro F1.
6. Bandingkan AP/F1 dengan `test_summary.json` memakai `np.isclose`, lalu tampilkan hasil.

```python
nama: binary_metrics(y[:, j], skor[:, j], keputusan[:, j])
```

`[:, j]` berarti semua baris untuk kolom label ke-j. Misalnya kolom sweet: bandingkan anotasi sweet, skor sweet, dan keputusan sweet untuk seluruh 1.333 molekul test.

| Kolom yang dicetak | Arti |
|---|---|
| `candidate` | Algoritma dan skenario A/B/C/D |
| `cv_ap_mean` | Rata-rata AP validasi berulang |
| `auprc` | AP test; key internal ini memakai Average Precision |
| `auroc` | AUROC test |
| `macro_f1_0_5` | Rata-rata F1 per label pada ambang 0,5 |

**Cakupan hitung ulang:** tabel menampilkan semua kandidat, tetapi penghitungan ulang AP/F1 di skrip ini hanya kandidat terpilih dan ambang utama 0,5. Hasil ambang OOF juga tersedia di folder run, tetapi tidak ditampilkan oleh skrip ini.

## 8. 05_prediksi.py — memakai model yang sudah belajar

Fungsi utama: `contoh_prediksi(smiles, jumlah)`.

Ia memanggil `predict` dari [predict_v7.py](../alignment/predict_v7.py). Fungsi asli memuat informasi bundle, memeriksa skema/versi/model, mengubah satu SMILES valid menjadi fitur yang urutannya sama, lalu memuat classifier per label untuk menghasilkan 109 skor.

Output per label berisi:

- `label`: nama aroma.
- `score`: skor dari `predict_proba`, belum dikalibrasi dan bukan intensitas aroma.
- `predicted`: benar jika skor mencapai ambang utama 0,5.

Hasil diurutkan dari skor terbesar. `--top 10` hanya menampilkan sepuluh baris pertama dari 109 hasil; **tidak memaksa sepuluh label positif**, tidak mengubah threshold, dan tidak mengubah model. Baris yang ditampilkan bisa memiliki `predicted=False`.

Skrip membutuhkan `--smiles`; default `--top` adalah 10 dan harus minimal 1. Ia tidak menyimpan prediksi ke file atau menambah molekul ke dataset. Mencoba SMILES di sini adalah demonstrasi fungsi; bukan otomatis evaluasi independen model.

## 9. Lokasi data dan hasil yang benar-benar dipakai

Semua tautan berikut relatif terhadap repositori ini dan membuka folder/file yang sudah ada.

| Isi | Lokasi |
|---|---|
| Lima sumber mentah yang dibekukan | [Snapshot Pyrfume](../data/snapshots/perfume-five-v1/8054ea98ed675005ec10e67359902f500e4911b0/) |
| Molekul, target, grup, split, jejak cleaning | [Dataset beku](../data/builds/perfume-five-grouped-v2/) |
| Morgan 2048 dan delapan deskriptor | [Fitur penelitian](../data/builds/perfume-m2048-d8-v7/) |
| Run utama lengkap | [Hasil kampus](../.local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v7-m2048-d8/perfume-campus-grouped-v7-m2048-d8/runs/perfume-five-grouped-v7-m2048-d8/) |
| Ringkasan interpretasi dan audit | [Laporan analisis](../reports/grouped_v7_analysis_20260928/) |

Alamat absolut run utama saat ini:

```text
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\.local-tools\campus-transfer-full-20260909\perfume-campus-grouped-v7-m2048-d8\perfume-campus-grouped-v7-m2048-d8\runs\perfume-five-grouped-v7-m2048-d8
```

Dua folder `perfume-campus-grouped-v7-m2048-d8` berurutan memang ada pada hasil ekstraksi sekarang; bukan salah ketik.

### Dataset beku

| File | Isi |
|---|---|
| `records.csv` | SMILES, label tercatat, sumber, identitas baris |
| `audit.json`, `rejected_records.json` | Jumlah/penjelasan baris ditolak |
| `label_mappings.csv`, `unmapped_descriptors.csv` | Pemetaan istilah dan istilah yang tidak cocok |
| `provenance.json` | Jejak baris/sumber yang membentuk rekaman gabungan |
| `groups.csv`, `splits.json` | Grup molekul dan indeks train/test/fold |
| `label_selection.csv`, `train_label_support.csv` | Pemilihan label dan dukungan positif/negatif train |
| `train.npz`, `test.npz` | Target dan indeks; juga menyimpan fitur build dasar 1024+5 |
| `dataset_manifest.json` | Urutan label, spesifikasi build dan checksum |

**Jangan salah mengambil fitur:** fitur aktual model 2048+8 berada di folder fitur penelitian, file `features.npz`. Isinya Morgan, delapan deskriptor, indeks baris, ID sumber, dan nama deskriptor. `descriptors.csv` adalah versi tabel deskriptor yang mudah dibuka. `manifest.json` mencatat spesifikasi dan identitasnya.

### Run utama

| Lokasi dalam run | Isi |
|---|---|
| `comparison_all.csv` | Tabel delapan kandidat; titik awal untuk membandingkan hasil |
| `RESULTS_SUMMARY.md` | Ringkasan run |
| `selection_frozen.json` | Kandidat/resep hasil seleksi yang dibekukan sebelum evaluasi test run ini |
| `complete.json` | Penanda dan ringkasan run selesai |
| `inference_bundle.json` | Petunjuk label, fitur, ambang dan lokasi classifier untuk prediksi; bukan file bobot model |
| `identity.json`, `completion_manifest.json` | Identitas kode/data/environment dan checksum artefak |
| `events.jsonl`, `progress.json` | Riwayat/tahap eksekusi |
| `xgb_A` sampai `xgb_D`, `lgbm_A` sampai `lgbm_D` | Hasil dan classifier tiap kandidat |
| `robustness/` | Hasil CV lima seed serta OOF |

Isi penting folder kandidat, misalnya `lgbm_D`:

| File | Isi |
|---|---|
| `recipe.json` | Hyperparameter yang dipakai fitting |
| `models.json` | Pemetaan nama label ke file classifier dan checksum |
| `label_000.joblib` … `label_108.joblib` | 109 classifier biner tersimpan; lihat `models.json` untuk mengetahui aroma tiap file |
| `trials.csv`, `optuna.sqlite3` | Riwayat trial dan database studi; tersedia untuk kandidat tuned C/D |
| `tuning_summary.json`, `budget.json` | Ringkasan pencarian dan pemakaian waktu |
| `test_predictions.npz` | Target, skor, keputusan ambang 0,5 dan nama label |
| `test_summary.json`, `test_per_label.json` | Metrik ringkasan dan per label pada ambang 0,5 |
| Berkas dengan akhiran `_validation_thresholds` | Evaluasi tambahan memakai ambang yang dipilih dari OOF |

Untuk hasil validasi, buka `robustness/lgbm_D`:

- `seed_7`, `seed_21`, `seed_42`, `seed_84`, `seed_100`: hasil setiap pengulangan.
- Dalam setiap folder seed: `folds.json`, `cv_fold_metrics.json`, `cv_per_label.json`, `cv_summary.json`, `cv_oof_predictions.npz`.
- `robustness_summary.json`: rata-rata dan simpangan baku skor antar-seed.
- `pooled_oof_predictions.npz`: prediksi OOF yang dirata-ratakan antar-seed.
- `thresholds.json`: ambang tambahan per label dari OOF.

OOF memakai baris **train**, bukan file prediksi pada 1.333 baris test. Perlu membaca keduanya sebagai tahap yang berbeda.

Catatan interpretasi: CV berulang ini bukan nested CV. Set test juga pernah diperiksa pada eksperimen terdahulu, sehingga jangan menjelaskannya sebagai bukti konfirmatori dari test yang sama sekali belum pernah dilihat sepanjang pengembangan. Teks `status` pada protokol merupakan snapshot saat persiapan; keberadaan hasil selesai diperiksa lewat artefak run, bukan hanya kalimat status lama itu.

## 10. Cara menjalankan contoh tanpa training ulang

PowerShell, dari lokasi mana pun:

```powershell
Set-Location 'C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier'

.\.venv\Scripts\python.exe KODE_ML\01_fitur.py
.\.venv\Scripts\python.exe KODE_ML\02_split.py
.\.venv\Scripts\python.exe KODE_ML\03_optuna.py
.\.venv\Scripts\python.exe KODE_ML\04_training_evaluasi.py
.\.venv\Scripts\python.exe KODE_ML\05_prediksi.py --smiles 'COc1cc(C=O)ccc1O' --top 10
```

Notebook bisa langsung dibuka di VS Code/Jupyter untuk melihat output tersimpan. Untuk menjalankannya ulang, pilih kernel Python dengan dependensi proyek dan `ipykernel`; jika belum tersedia, editor akan meminta pemasangannya. Jalankan sel dari atas ke bawah. Notebook membaca data yang ada dan menghitung fitur satu contoh sebagai pemeriksaan, tanpa menulis build baru.

Format file: `.csv` tabel, `.json` pengaturan/ringkasan terstruktur, `.npz` kumpulan array NumPy terkompresi, `.joblib` objek model, `.sqlite3` database Optuna. NPZ dan joblib bukan teks yang bisa dipahami dengan membuka isinya seperti CSV.

## 11. Pola sintaks yang akan sering terlihat

| Potongan | Maksudnya di proyek ini |
|---|---|
| `def nama(...):` | Mendefinisikan fungsi; fungsi belum berjalan sampai dipanggil |
| `if __name__ == "__main__":` | Jalankan bagian CLI jika file dijalankan langsung |
| `argparse` | Membaca parameter terminal seperti `--smiles` dan `--top` |
| `sys.path.insert(0, str(ROOT))` | Membuat modul asli di root proyek bisa diimpor oleh skrip contoh |
| `np.asarray(...)` | Mengubah daftar menjadi array angka |
| `.shape` | Ukuran array: jumlah baris dan kolom |
| `y[:, j]` | Semua baris dari label ke-j |
| `deskriptor[0]` | Delapan nilai untuk molekul pertama |
| `astype(np.float32)` | Menyamakan tipe angka fitur dengan yang dipakai model |
| `with np.load(..., allow_pickle=False)` | Memuat array NPZ lalu menutup berkas setelah blok selesai |
| `assert ...` | Memeriksa syarat saat eksekusi normal; jika gagal program berhenti di situ |
| `print(...)` | Menampilkan teks ke terminal; bukan menyimpan file |

## 12. Setelah memahami contoh, lanjut ke implementasi asli

| Tahap | File/fungsi asli |
|---|---|
| Preprocessing lengkap | [perfume_data.py](../alignment/perfume_data.py): `load_sources`, `merge_sources`, `encode_and_split`, `prepare` |
| Helper identitas | [build_dataset.py](../src/build_dataset.py): `_attach_smiles`, `_attach_sigma`, `_cas_bridge`; normalisasi string dari [label_harmonization.py](../src/label_harmonization.py) |
| Grup/split | [grouped_splits.py](../alignment/grouped_splits.py): `structure_feature_groups`, `outer_split`, `inner_folds` |
| Fitur aktual | [descriptors_v7.py](../alignment/descriptors_v7.py): `calculate_morgan`, `calculate` |
| Aturan eksperimen | [protocol_v7.json](../alignment/protocol_v7.json) |
| Tuning | [tuning_v6.py](../alignment/tuning_v6.py): `suggest`, `tune` |
| Fitting dan validasi | [experiments.py](../alignment/experiments.py): `fit`, `cross_validate`, `repeated_grouped_cv` |
| Pengatur eksperimen | [experiments_v7.py](../alignment/experiments_v7.py): `run_all` |
| Fitting final | [experiments_v5.py](../alignment/experiments_v5.py): `fit_final_models` |
| Metrik | [metrics.py](../alignment/metrics.py): `binary_metrics`, `summarize` |
| Inferensi | [predict_v7.py](../alignment/predict_v7.py): `predict` |

Urutan belajar: notebook → fitur satu molekul → split satu label → riwayat Optuna → evaluasi → prediksi. Setelah itu telusuri satu label pada `fit` dan `cross_validate` untuk memahami proses training yang sebenarnya.
