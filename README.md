---
title: Essenza Fingerprint API
emoji: 🧪
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "5.49.1"
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Versioned RDKit features for Essenza mobile inference
---

## Grouped-v4

Paket `perfume-campus-grouped-v4.zip` melanjutkan studi Optuna C dan D untuk
XGBoost serta LightGBM dari grouped-v3 selama 2 jam tambahan per studi. Kedua
algoritma menerima tambahan budget yang sama. Dataset, grouped split, ruang
pencarian, bobot kelas, dan metrik tidak berubah. Hasil v4 bersifat eksploratori;
grouped-v3 tetap menjadi perbandingan utama XGBoost–LightGBM.

Instruksi PC kampus tersedia di [WINDOWS_GUIDE_V4.md](WINDOWS_GUIDE_V4.md).

# Perfume MultiLabel Classifier

## Protokol eksperimen lanjutan — 11 September 2026

Runner aktif adalah **perfume-five-grouped-v3**. Dataset dan split kelompok tetap
memakai `perfume-five-grouped-v2` agar perubahannya hanya berada pada metode training
dan evaluasi. Hasil v2 tidak ditimpa.

Perbaikan v3:

- XGBoost dan LightGBM memakai bobot kelas seimbang dengan rumus yang sama;
- Optuna memakai dua jam penuh untuk C dan dua jam penuh untuk D per algoritma;
- metrik tiap fold dan prediksi out-of-fold disimpan;
- satu finalis per algoritma diperiksa dengan lima pengulangan grouped CV;
- threshold per label dipilih hanya dari validasi dan dilaporkan terpisah dari
  enam metrik utama pada threshold 0,5.

Gunakan [WINDOWS_GUIDE.md](WINDOWS_GUIDE.md) dan paket
`.local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v3.zip`.
Kode v3 sudah siap diuji; training v3 belum dijalankan. Bagian bertanggal lebih lama
di bawah dipertahankan sebagai riwayat.

## Revisi split kelompok — 10 September 2026

Versi aktif: **perfume-five-grouped-v2**. Seluruh 6.686 molekul dipertahankan.
Data latih 5.353, test 1.333, target 109 label, 5.363 kelompok; semua label memiliki 5 fold.
Tidak ada kelompok/fingerprint identik yang melintasi latih-test atau 545 fold validasi.
Daftar target mengikuti training baru: narcissus masuk, vetiver tidak memenuhi ambang training.
Gunakan [WINDOWS_GUIDE.md](WINDOWS_GUIDE.md) dan RUN_PERFUME_FIVE.cmd.
Paket lengkap: .local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v2.zip.
Profil PC kampus aktif memakai 6 thread CPU per model pada i7-8700K/32 GB; XGBoost dan LightGBM berjalan berurutan.
Dataset lama perfume-five-v1 dipertahankan sebagai riwayat dan ditolak runner aktif.
Bukti: reports/grouped_v2_20260910/validation.json. Training penelitian belum dijalankan.
Catatan versi sebelumnya di bawah adalah riwayat.



## Revisi aktif — 10 September 2026

Eksperimen aktif hanya memakai **GoodScents, IFRA 2019, Leffingwell, Arctander 1960, Sigma-Aldrich 2014**.
Gunakan [WINDOWS_GUIDE.md](WINDOWS_GUIDE.md), RUN_PERFUME_FIVE.cmd, serta
[notebook preprocessing](notebooks/02_perfume_five_preprocessing.ipynb).
Hasil: 6.686 molekul, 109 label, fitur 1.024 atau 1.029; metadata dan split tersimpan pada perfume-five-v1.
Preprocessing/notebook dan 21 tes komponen lulus. Training penelitian terbaru belum dijalankan.
Revisi ini menggantikan eksperimen 25 label dan rencana penggunaan seluruh dataset Suh.
Model/API historis dan arsip tetap disimpan; bagian prosedur lama di bawah bukan panduan run terbaru.


Proyek ini mempelajari hubungan antara struktur kimia molekul dan karakter aromanya, seperti floral, citrus, sweet, atau woody. Satu molekul dapat memiliki beberapa karakter aroma sekaligus, sehingga tugas ini menggunakan klasifikasi **multilabel**.

Repository ini berisi pengolahan data, pelatihan model XGBoost dan LightGBM, evaluasi, serta API untuk aplikasi Android **Essenza**. Proyek dikembangkan oleh Marvel Kevin Nathanael sebagai bagian dari penelitian skripsi.

Aplikasi mobile ada di repository [Essenza Frontend, branch `marvel`](https://github.com/yogawyas/Essenza_Frontend/tree/marvel).

## Daftar isi

- [Cara kerja](#cara-kerja)
- [Status proyek](#status-proyek)
- [Isi repository](#isi-repository)
- [Mencoba API di komputer sendiri](#mencoba-api-di-komputer-sendiri)
- [Data dan fitur molekul](#data-dan-fitur-molekul)
- [Menjalankan eksperimen](#menjalankan-eksperimen)
- [Menggunakan model di Essenza](#menggunakan-model-di-essenza)
- [Analisis model dan katalog parfum](#analisis-model-dan-katalog-parfum)
- [Pengujian dan masalah umum](#pengujian-dan-masalah-umum)
- [Batasan dan referensi](#batasan-dan-referensi)

## Cara kerja

Ada dua alur utama: menyiapkan model di komputer dan menggunakan model tersebut di aplikasi.

**Saat menyiapkan model:**

```text
Data molekul dan label aroma
  -> Pembersihan data dan pembagian development/test
  -> Perhitungan fitur dengan RDKit
  -> Pelatihan dan tuning XGBoost / LightGBM
  -> Evaluasi dan ekspor model ke ONNX
```

**Saat menggunakan Essenza:**

```text
SMILES dari aplikasi
  -> API menghitung fitur molekul
  -> Aplikasi menerima fitur
  -> Model ONNX berjalan di ponsel
  -> Aplikasi menampilkan skor label aroma
```

**SMILES** adalah penulisan struktur molekul dalam bentuk teks. Contohnya, `CCO` adalah SMILES untuk etanol. RDKit mengubah struktur tersebut menjadi angka yang dapat dibaca model.

Masing-masing algoritme memakai pendekatan **Binary Relevance**: satu model dilatih untuk setiap label aroma. Dengan 25 label, satu kandidat terdiri dari 25 model. Setiap label juga mempunyai *threshold*, yaitu batas skor untuk menentukan apakah label tersebut dipilih.

Koneksi internet diperlukan ketika aplikasi meminta fitur ke API. Setelah fitur diterima, perhitungan model berlangsung di ponsel. Fitur **Explorer** memakai katalog parfum lokal dan menyimpan parfum buatan pengguna melalui AsyncStorage.

## Status proyek

| Bagian | Status |
| --- | --- |
| Pengolahan data | Dataset versi terbaru sudah dibangun dan diaudit: 6.703 molekul, 25 label. |
| Pengaman runtime | Step 2: 90 tes ML lulus; runner smoke: 8 tes sintetis lulus. Step 4 attempt 03 STOP sebelum fitting, suhu akhir 84,875°C. [Bukti terbaru](reports/step4_20260908/attempt-03/README.md). |
| Protokol eksperimen | Step 3 GO: [praregistrasi](PREREGISTRATION.md) dan [audit](reports/step3_20260908/README.md) terkunci; cap tuning 4 jam/model, jumlah trial menunggu pengukuran resource. |
| Pelatihan | Kode baseline, tuning terpisah, dan evaluasi tersedia. Eksperimen dengan protokol terbaru belum dijalankan pada data nyata. |
| Model aplikasi | Masih menggunakan 25 model XGBoost dari eksperimen sebelumnya. |
| API | REST dan Gradio tersedia untuk dijalankan lokal. Deployment layanan yang diperbarui masih menunggu. |
| Android | Build APK dan pengujian di perangkat untuk versi terbaru masih menunggu. |

Hasil evaluasi dari protokol terbaru belum tersedia. Model dan laporan eksperimen lama tetap disimpan sebagai riwayat penelitian. Rincian pemeriksaan ada di [catatan implementasi](IMPLEMENTATION_STATUS.md).

## Isi repository

```text
src/                      Pengolahan data, fitur, eksperimen, dan evaluasi
tests/                    Pengujian otomatis
app.py                    Antarmuka Gradio dan endpoint untuk aplikasi
api_light.py              REST API untuk menghitung fitur molekul
config.yaml               Konfigurasi dasar/historis; bukan profil training operasional
config.safe-smoke.yaml     Profil smoke 1 thread, 5 menit
config.safe-training.yaml  Profil training 2 thread, 20 menit
feature_spec.json         Spesifikasi fitur untuk model aplikasi yang lama
export_for_mobile.py      Ekspor model terlatih ke ONNX
requirements_api.txt      Dependency untuk menjalankan API
requirements_training.txt Dependency untuk training dan ekspor model
requirements_dev.txt      Dependency lengkap, termasuk pengujian dan plot
requirements_lock.txt     Catatan versi dependency lingkungan Windows yang diuji
data/                     Data dan hasil pengolahannya
models/                   Model dari eksperimen sebelumnya
mobile_assets/            Aset model untuk aplikasi dari eksperimen sebelumnya
reports/                  Audit, hasil evaluasi, dan gambar
notebooks/                Notebook eksplorasi dan pemodelan sebelumnya
Dockerfile                Menjalankan REST API melalui Docker
```

Hasil build dataset baru berada di `data/builds/`, cache sumber di `data/snapshots/`, dan hasil eksperimen di `runs/`. Ketiga folder ini diabaikan oleh Git, sehingga perlu dibuat melalui perintah di bawah setelah clone.

Untuk eksperimen baru, gunakan modul `src.build_dataset` dan `src.train_and_evaluate`. Notebook lama, `etl_pipeline.py`, `generate_plot.py`, serta `api.py` dipertahankan sebagai bagian dari implementasi sebelumnya.

## Mencoba API di komputer sendiri

Siapkan **Python 3.12**, Git, dan koneksi internet untuk mengunduh dependency. Proyek menggunakan CPU; GPU tidak diperlukan.

Contoh perintah berikut memakai PowerShell pada Windows. Jalankan dari folder repository. Pada Linux/macOS, gunakan `.venv/bin/python` sebagai pengganti `.\.venv\Scripts\python.exe`.

### 1. Instalasi

Pastikan `python --version` menunjukkan Python 3.12, lalu jalankan:

```powershell
git clone https://github.com/marvelkn/Perfume-MultiLabel-Classifier.git
cd Perfume-MultiLabel-Classifier
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_api.txt
```

Versi dependency sudah ditentukan di file requirements. RDKit yang digunakan adalah **2026.03.4**. Versi ini ikut dicatat dalam spesifikasi fitur karena perubahan RDKit dapat memengaruhi angka yang diterima model.

### 2. Jalankan REST API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api_light:app --host 127.0.0.1 --port 8000
```

Buka [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) untuk melihat dan mencoba endpoint melalui browser.

| Endpoint | Kegunaan |
| --- | --- |
| `GET /` | Memeriksa layanan dan melihat ID versi fitur. |
| `POST /fingerprint` | Menghitung fitur dan informasi molekul dari SMILES atau nama senyawa. |

Untuk mencoba dari terminal PowerShell kedua:

```powershell
$body = '{"smiles":"CCO"}'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/fingerprint" -Method Post -ContentType "application/json" -Body $body
```

Respons berisi `fingerprint` dengan 2.053 angka, SMILES yang sudah dinormalisasi, rumus molekul, berat molekul, serta `feature_schema_id` dan `feature_spec`. Nilai `fingerprint` inilah yang diteruskan ke model aroma di aplikasi.

Selain `smiles`, request dapat memakai `compound_name`, misalnya `{"compound_name":"ethanol"}`. Pencarian nama memerlukan akses ke PubChem. Jika SMILES sudah diberikan, API langsung memprosesnya tanpa pencarian nama. Field opsional `feature_schema_id` dapat dikirim untuk memeriksa kecocokan versi fitur.

### Pilihan lain: Gradio atau Docker

Gradio menyediakan halaman sederhana untuk memasukkan SMILES:

```powershell
.\.venv\Scripts\python.exe app.py
```

Buka [http://127.0.0.1:7860](http://127.0.0.1:7860). Aplikasi mobile menggunakan endpoint Gradio `/gradio_api/call/predict`. File `app.py` juga menjadi entry point untuk Hugging Face Spaces.

Jika Docker sudah terpasang, REST API dapat dijalankan dengan:

```powershell
docker build -t essenza-api .
docker run --rm -p 7860:7860 essenza-api
```

Dokumentasi REST untuk pilihan Docker tersedia di [http://127.0.0.1:7860/docs](http://127.0.0.1:7860/docs).

## Data dan fitur molekul

Data berasal dari arsip Pyrfume yang mencakup GoodScents, Leffingwell, Arctander, Sigma, dan Flavornet. Versi sumber ditetapkan melalui `data_revision` di [config.yaml](config.yaml), saat ini `8054ea98ed675005ec10e67359902f500e4911b0`.

Data dari tiap sumber dicocokkan melalui identitas senyawa, kemudian SMILES dinormalisasi dan nama label diseragamkan. Molekul yang sama digabungkan. Catatan yang identitasnya belum jelas, SMILES tidak valid, atau mengandung beberapa fragmen dipisahkan dari dataset yang digunakan.

Ringkasan [audit dataset terbaru](reports/audit_20260904/dataset_audit.json):

| Isi dataset | Jumlah |
| --- | --- |
| Molekul | 6.703 |
| Development, untuk pelatihan dan pemilihan model | 5.383 |
| Test, untuk evaluasi akhir | 1.320 |
| Label aroma | 25 |
| Fitur per molekul | 2.053 |

Label pada dataset tersebut:

```text
apple, balsamic, citrus, earthy, ethereal, fatty, floral, fresh,
fruity, green, herbal, meaty, mint, musty, nutty, oily, rose,
spicy, sulfurous, sweet, tropical, vegetable, waxy, winey, woody
```

Fitur terdiri dari **Morgan fingerprint 2.048 bit dengan radius 2** dan lima deskriptor: berat molekul (`MolWt`), `MolLogP`, jumlah donor dan akseptor ikatan hidrogen, serta luas permukaan polar (`TPSA`). Seluruhnya disimpan sebagai `float32`.

Daftar label dipilih berdasarkan frekuensi pada data development. Jika sumber, seed, atau aturan pemilihan berubah, hasil dataset dapat berbeda. File `dataset_manifest.json` pada setiap build menyimpan label, spesifikasi fitur, asal data, dan checksum untuk memeriksa keutuhan file.

## Menjalankan eksperimen

Perintah di bawah adalah panduan untuk tahap yang sudah diizinkan pada [SAFE_EXECUTION_PLAN.md](SAFE_EXECUTION_PLAN.md). Per 8 September 2026, Step 0–3 selesai. Izin terbaru mencakup Step 4 sampai Step 5. Step 4 attempt 03 STOP sebelum smoke karena kondisi termal; [persiapan Step 5](reports/step5_20260908/README.md) terverifikasi, tetapi jumlah trial dan run final menunggu smoke GO. Training penelitian belum dimulai; Step 6 dan seterusnya belum diizinkan.

Pasang dependency lengkap untuk training, pengujian, dan visualisasi:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_dev.txt
```

Untuk training dan ekspor tanpa pengujian atau plot, tersedia `requirements_training.txt`. `requirements_lock.txt` mencatat lingkungan Windows yang pernah diuji; pemasangan di OS lain tetap perlu diperiksa.

### 1. Siapkan dataset dan folder eksperimen

Pilih profil sebelum proses Python dijalankan. Contoh berikut untuk run final setelah Step 5 diizinkan; smoke pada Step 4 memakai `config.safe-smoke.yaml` dan folder run terpisah.

```powershell
$env:ESSENZA_CONFIG = (Resolve-Path .\config.safe-training.yaml).Path
$env:ESSENZA_THREADS = '2'
.\.venv\Scripts\python.exe -m src.build_dataset --output data/builds/experiment-01
.\.venv\Scripts\python.exe -m src.audit_dataset --dataset data/builds/experiment-01 --output reports/experiment-01-audit.json
.\.venv\Scripts\python.exe -m src.train_and_evaluate init --dataset data/builds/experiment-01 --run runs/experiment-01
```

Gunakan nama folder baru untuk setiap build dan eksperimen. Perintah `init` menyiapkan pembagian data dan konfigurasi; training dimulai melalui perintah berikutnya.

Nama `X_train.npz` pada hasil build merujuk pada seluruh data development. Sebanyak 15% dari development disisihkan untuk memilih threshold. Sisanya dibagi menjadi tiga fold untuk validasi silang. Di dalam setiap fold, data untuk melatih model, menentukan kapan training berhenti, dan menghitung skor dipisahkan.

Test disimpan untuk evaluasi akhir. Untuk protokol aktif gunakan dataset audit v2 yang sudah dibekukan. Sensitivitas memakai partisi development pada [PREREGISTRATION.md](PREREGISTRATION.md); jangan memakai `--split scaffold` atau `--seed` untuk membangun ulang outer test protokol ini. Opsi build tersebut tersedia untuk rancangan eksperimen terpisah.

### 2. Siapkan pemantauan komputer

Profil operasional terpisah menjaga konfigurasi dan bukti run lama tetap utuh:

| Pengaturan | safe-smoke | safe-training |
| --- | --- | --- |
| Thread learner | 1 | 2 |
| RAM proses dan anak | 3 GiB | 4 GiB |
| Waktu per sesi | 5 menit | 20 menit |
| Suhu untuk mulai | <75°C | <80°C |
| Stop suhu berkelanjutan | ≥80°C selama 5 detik | ≥85°C selama 10 detik |

Kedua profil mensyaratkan minimal 4 GiB RAM bebas, telemetry maksimal 10 detik, dan stop pada pembacaan ≥90°C tanpa toleransi durasi. Batas diperiksa secara kooperatif: sebelum/sesudah pemuatan data, antarlabel/fold, pada callback boosting, dan saat sesi ditutup. Operasi native yang sedang berjalan tidak dapat diputus di tengah oleh guard ini; interval pemeriksaan 1 detik bukan jaminan latensi penghentian 1 detik.

Pembaca sensor Windows kini tersedia melalui adapter LibreHardwareMonitor pada [SENSOR_SETUP.md](SENSOR_SETUP.md). Gunakan sensor nyata yang aktif; contoh JSON atau file berstatus STOP tidak dapat dipakai untuk menjalankan training.

Perintah berikut memakai `C:\sensors\cpu.json` sebagai contoh lokasi. Ganti sesuai telemetry aktif. Rincian konfigurasi, log, cooldown, dan resume ada di [RUNTIME_SAFETY.md](RUNTIME_SAFETY.md).

### 3. Jalankan baseline dan tuning

Baseline memberi pembanding sebelum pencarian parameter. Masing-masing algoritme dicoba dengan tiga cara: data asli, pembobotan kelas, dan penambahan sampel kelas yang jarang muncul (*random oversampling*).

```powershell
.\.venv\Scripts\python.exe -m src.train_and_evaluate baseline --run runs/experiment-01 --model xgb --temperature-file C:\sensors\cpu.json
.\.venv\Scripts\python.exe -m src.train_and_evaluate baseline --run runs/experiment-01 --model lgbm --temperature-file C:\sensors\cpu.json
```

Baseline menyimpan prediksi validasi per strategi/fold/label beserta checksum. Setelah pendinginan dan pemeriksaan kondisi, jalankan perintah yang sama untuk melanjutkan; label yang terputus dikerjakan ulang. Checkpoint yang selesai dipakai kembali dan skor dihitung ulang dari prediksinya.

Setelah itu, Optuna mencari kombinasi parameter dan cara penanganan kelas yang memberikan skor validasi terbaik:

```powershell
.\.venv\Scripts\python.exe -m src.train_and_evaluate tune --run runs/experiment-01 --model xgb --trials 1 --temperature-file C:\sensors\cpu.json
.\.venv\Scripts\python.exe -m src.train_and_evaluate tune --run runs/experiment-01 --model lgbm --trials 1 --temperature-file C:\sensors\cpu.json
```

XGBoost dan LightGBM mempunyai studi terpisah. Ruang pencariannya mengikuti parameter masing-masing algoritme:

| Parameter | XGBoost | LightGBM |
| --- | --- | --- |
| `max_depth` | 3–8 | 4–12 |
| `min_child_weight` | 1–30 (log) | Tidak digunakan |
| `num_leaves` | Tidak digunakan | 8 sampai `min(128, 2^max_depth)` |
| `min_child_samples` | Tidak digunakan | 10–150 |
| `learning_rate` | 0.01–0.2 (log) | 0.005–0.15 (log) |
| `gamma` / `min_split_gain` | `gamma`: 0–5 | `min_split_gain`: 0–1 |
| `subsample` | 0.6–1 | 0.6–1 |
| `subsample_freq` | Tidak digunakan | 1–7 |
| `colsample_bytree` | 0.4–1 | 0.5–1 |
| L1: `reg_alpha` | 0, atau 0.00001–10 (log) | 0, atau 0.00001–10 (log) |
| L2: `reg_lambda` | 0.001–100 (log) | 0.001–50 (log) |
| `max_bin` | Default | 63, 127, atau 255 |
| Metode | `tree_method="hist"` | `boosting_type="gbdt"` |
| Batas boosting rounds | 2.000 | 3.000 |

`log` berarti pencarian dilakukan pada skala logaritmik. Angka pada tabel adalah rentang yang akan diuji; parameter terbaik baru diketahui setelah eksperimen selesai.

Kedua algoritme memakai *early stopping* ketika logloss tidak membaik selama 50 putaran. Peringkat trial ditentukan oleh **macro average precision (AP)**: skor dihitung untuk setiap label, lalu dirata-ratakan agar semua label ikut dinilai. Optuna memakai TPE dengan 10 trial awal dan MedianPruner untuk menghentikan trial yang kurang menjanjikan setelah penilaian fold.

`--trials 1` berarti menambah satu trial pada pemanggilan itu, bukan total budget penelitian. Budget final ditetapkan di Step 5 berdasarkan resource smoke test setelah aturan dipraregistrasikan di Step 3. Waktu pengerjaan kedua algoritme bisa berbeda. Riwayat disimpan di `runs/experiment-01/studies.sqlite3` dan file CSV per algoritme. Jalankan kembali perintah `tune` untuk melanjutkan studi; trial yang sudah selesai tetap tersimpan, sedangkan trial yang terputus tidak dilanjutkan dari tengah.

Opsi `--include-mlsmote` menambahkan percobaan MLSMOTE yang diadaptasi untuk fitur biner dan numerik. Pilihan ini masih eksperimental: fitur sintetis yang dihasilkan belum tentu mewakili molekul nyata. Seluruh resampling dilakukan hanya pada bagian data training.

### 4. Latih model akhir dan evaluasi

Setelah tuning selesai, latih kedua kandidat dengan parameter terpilih:

```powershell
.\.venv\Scripts\python.exe -m src.train_and_evaluate fit --run runs/experiment-01 --model xgb --temperature-file C:\sensors\cpu.json
.\.venv\Scripts\python.exe -m src.train_and_evaluate fit --run runs/experiment-01 --model lgbm --temperature-file C:\sensors\cpu.json
```

Jumlah putaran training setiap label diambil dari median putaran terbaik antar-fold. Threshold dipilih menggunakan bagian development yang disisihkan sebelumnya. Jika proses terhenti, perintah `fit` dapat dijalankan kembali dengan konfigurasi yang sama untuk melanjutkan model yang belum selesai.

Selesaikan kedua kandidat sebelum membuka test:

```powershell
.\.venv\Scripts\python.exe -m src.train_and_evaluate evaluate --run runs/experiment-01 --model xgb
.\.venv\Scripts\python.exe -m src.train_and_evaluate evaluate --run runs/experiment-01 --model lgbm
```

Saat evaluasi pertama dimulai, pilihan model berdasarkan validasi disimpan dan run dikunci untuk perubahan tuning atau fitting. Hasil test digunakan untuk pelaporan akhir, sehingga jangan menggunakannya sebagai dasar mencoba parameter baru.

Hasil tiap kandidat tersimpan di `runs/experiment-01/xgb/` atau `runs/experiment-01/lgbm/`:

| File | Isi |
| --- | --- |
| `model_manifest.json` | Parameter, label, threshold, versi fitur, dan daftar model. |
| `test_metrics.json` | AP, F1, precision, recall, subset accuracy, Hamming loss, dan ROC-AUC. |
| `test_predictions.npz` | Skor prediksi dan label sebenarnya pada test. |

AP dan F1 dilaporkan sebagai macro/micro, disertai rincian per label. AP di sini mengikuti perhitungan average precision, bukan integrasi trapezoid kurva precision-recall.

## Menggunakan model di Essenza

Ekspor kandidat yang dipilih berdasarkan validasi. Contoh berikut memakai XGBoost:

```powershell
.\.venv\Scripts\python.exe export_for_mobile.py --models runs/experiment-01/xgb --features data/builds/experiment-01/X_train.npz --output runs/experiment-01/mobile-bundle
.\.venv\Scripts\python.exe -m src.benchmark_onnx --bundle runs/experiment-01/mobile-bundle --features data/builds/experiment-01/X_train.npz --output runs/experiment-01/benchmark.json
```

Untuk LightGBM, ganti folder model `xgb` menjadi `lgbm` dan gunakan folder output baru. Ekspor menghasilkan file ONNX, `feature_spec.json`, dan `manifest.json`. Proses ini memeriksa keutuhan file, urutan label, serta kesesuaian skor dan keputusan threshold antara model Python dan ONNX.

Benchmark mengukur waktu dan penggunaan memori model di komputer tempat perintah dijalankan. Pengujian waktu respons keseluruhan tetap perlu dilakukan di ponsel.

**API dan model harus memakai spesifikasi fitur yang sama.** Model aplikasi lama memakai `include_chirality=false`, sedangkan eksperimen baru memakai `true` untuk menyertakan informasi stereokimia. ID versi fitur membantu aplikasi mendeteksi jika keduanya tidak cocok.

Untuk melayani bundle baru melalui API lokal, atur lokasi spesifikasinya sebelum menyalakan server:

```powershell
$env:ESSENZA_FEATURE_SPEC = "runs/experiment-01/mobile-bundle/feature_spec.json"
.\.venv\Scripts\python.exe -m uvicorn api_light:app --host 127.0.0.1 --port 8000
```

Secara default, API memakai `feature_spec.json` di root repository untuk model aplikasi lama. Pengaturan `ESSENZA_FEATURE_SPEC` juga dibaca oleh Gradio. Saat memperbarui aplikasi, gunakan bundle dan spesifikasi fitur dari hasil ekspor yang sama. Panduan aplikasi tersedia di [repository frontend](https://github.com/yogawyas/Essenza_Frontend/tree/marvel).

## Analisis model dan katalog parfum

### Melihat fitur yang memengaruhi model

Setelah fitting selesai, jalankan TreeSHAP untuk melihat kontribusi fitur pada prediksi:

```powershell
.\.venv\Scripts\python.exe -m src.explain_model --run runs/experiment-01 --model xgb --output runs/experiment-01/explanations --temperature-file C:\sensors\cpu.json
.\.venv\Scripts\python.exe -m src.label_distance_viz --explanations runs/experiment-01/explanations --output runs/experiment-01/label-distances
```

Hasilnya mencakup ringkasan kontribusi fitur untuk seluruh label, heatmap jarak cosine, dan peta MDS dua dimensi. Visualisasi ini menggambarkan kemiripan pola yang dipelajari model. Kedekatan dua label pada gambar belum membuktikan bahwa aromanya dirasakan mirip oleh manusia.

### Menyiapkan katalog Explorer

Katalog Explorer berisi data parfum jadi. Kategori aroma atau *accord* pada katalog mempunyai daftar tersendiri, terpisah dari label prediksi molekul.

Untuk membuat katalog baru, siapkan CSV dengan kolom `pid,brand,name,gender,rating,accords`. Kolom `accords` berisi objek JSON, misalnya `{"woody":0.7,"fresh":0.4}`; nilai contoh ini hanya menjelaskan format. Bobot sebenarnya harus berasal dari sumber data, dalam rentang 0–1.

```powershell
.\.venv\Scripts\python.exe -m src.export_catalog --input normalized-catalog.csv --output catalog-build --source-url https://source.example/catalog --source-version snapshot-id
```

Ganti URL dan versi contoh dengan sumber yang benar. Exporter menghasilkan `perfumes.json` dan `catalog_manifest.json`. Katalog yang dipakai aplikasi saat ini berisi 2.029 parfum; berkas sumber mentah dan riwayat pengolahan lengkapnya masih perlu dilengkapi.

## Pengujian dan masalah umum

Setelah memasang `requirements_dev.txt`, jalankan:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Pengujian mencakup pencocokan data, kesesuaian fitur API/training, pembagian data, pembatas sumber daya, kelanjutan fitting, dan ekspor kedua algoritme. Tes pemodelan memakai data sintetis kecil; hasilnya memeriksa fungsi kode, sementara kualitas prediksi diukur melalui eksperimen.

| Masalah | Yang perlu diperiksa |
| --- | --- |
| `CPU temperature unavailable` atau data sensor terlalu lama | Pastikan pembaca sensor berjalan dan file suhu terus diperbarui. |
| Training berhenti karena waktu atau RAM | Baca pesan penghentiannya. Periksa log `sessions/`, lakukan cooldown sesuai plan, lalu lanjutkan `baseline`, `tune`, atau `fit` dengan run yang sama. |
| `FEATURE_SCHEMA_MISMATCH` atau versi RDKit berbeda | Cocokkan versi RDKit, file spesifikasi API, dan bundle aplikasi. |
| `UNSUPPORTED_MIXTURE` | Gunakan SMILES satu molekul yang terhubung; input beberapa fragmen tidak didukung. |
| Pencarian nama gagal dengan `LOOKUP_UNAVAILABLE` | Periksa koneksi ke PubChem atau masukkan SMILES yang sudah diketahui. |
| Folder output sudah ada | Gunakan nama baru saat build, init, atau ekspor agar hasil sebelumnya tetap tersimpan. |
| `Study protocol/code differs` | Buat run baru jika kode, dependency, atau protokol eksperimen berubah. |
| Ekspor ONNX gagal setelah mengubah dependency | Gunakan versi pada requirements; pasangan eksportir yang diuji memakai `protobuf==5.29.5`. |

`config.yaml` dipertahankan sebagai konfigurasi dasar/historis. Inisialisasi run operasional wajib memilih profil aman melalui `ESSENZA_CONFIG` sebelum menjalankan Python. `baseline`, `tune`, `fit`, dan `explain` memvalidasi profil dari `run.json`; mengganti environment variable tidak mengubah run yang sudah ada. Arsip pada `runs/_reference/` tidak boleh dijalankan.

## Batasan dan referensi

Penelitian ini berfokus pada **aroma satu molekul**. Prediksi campuran parfum dan interaksi antarbahannya belum divalidasi. Skor model juga belum dikalibrasi sebagai probabilitas persepsi aroma manusia.

Label yang tidak tercatat dalam sumber data diperlakukan sebagai negatif, walaupun ketiadaan aroma tersebut belum diuji langsung. Sebagian molekul pernah digunakan dalam eksperimen sebelumnya, sehingga pembagian data terbaru belum menjadi pengujian eksternal independen. Kerangka molekul juga dapat beririsan antara development dan test; rincian ada pada audit dataset.

Model dalam format pickle atau joblib sebaiknya hanya dimuat dari hasil eksperimen yang sumbernya dipercaya.

Dasar pemilihan metode dan tautan penelitian tersedia di [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md), termasuk karya tentang XGBoost, LightGBM, Optuna, pembagian data multilabel, evaluasi model, dan SHAP. Catatan tersebut juga menjelaskan pemeriksaan yang sudah dilakukan serta pekerjaan yang masih menunggu hasil eksperimen.
