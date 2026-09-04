---
title: Essenza Fingerprint API
emoji: ðŸ§ª
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

# Essenza ML project

Pipeline penelitian multilabel aroma molekul dan layanan fitur untuk aplikasi Essenza.
Status implementasi, keterbatasan, dan dasar penelitian ada di [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).
Laporan skripsi belum diperbarui. Tidak ada hasil tuning baru yang diklaim.

## Arsitektur

Pyrfume pada revision tetap â†’ join Stimulus/CID/CAS yang divalidasi â†’ canonical single-molecule SMILES â†’ harmonisasi label â†’ pembagian data â†’ Morgan radius 2, 2.048 bit + lima descriptor float32 â†’ Binary Relevance XGBoost/LightGBM â†’ threshold per label â†’ ekspor ONNX dengan pemeriksaan kesesuaian prediksi â†’ bundle aplikasi.

API menghitung fitur menggunakan fungsi yang sama dengan training. Aplikasi meminta fitur melalui internet, lalu menjalankan seluruh model ONNX di perangkat. Explorer memakai katalog JSON dan AsyncStorage secara lokal. Accord katalog berbeda dari label prediksi molekul.

**Kontrak versi:** schema fitur mencakup RDKit 2026.03.4, chirality, radius, panjang fingerprint, descriptor, tipe data, dan kebijakan satu molekul. Bundle lama tetap memakai chirality=false. Eksperimen baru memakai chirality=true; model lama tidak boleh menerima fitur eksperimen baru. API dan bundle harus diperbarui bersama.

## Instalasi lokal

Python 3.12. Gunakan lingkungan terpisah:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

requirements_training.txt untuk eksperimen; requirements_api.txt untuk server. requirements_lock.txt mencatat dependency transitif lingkungan Windows yang diverifikasi, bukan jaminan lock lintas OS. Pin protobuf 5.29.5 diperlukan oleh kombinasi eksportir/ONNX yang diuji; protobuf 7.36.1 gagal saat konversi XGBoost. Pin RDKit dipilih setelah pemeriksaan kompatibilitas seluruh 5.091 baris training historis yang berupa molekul tunggal: selisih fitur maksimum 0.

## Data dan protokol

Build selalu meminta direktori baru, tidak menimpa data historis:

```powershell
.\.venv\Scripts\python.exe -m src.build_dataset --output data/builds/experiment-01
.\.venv\Scripts\python.exe -m src.audit_dataset --dataset data/builds/experiment-01 --output reports/experiment-01-audit.json
.\.venv\Scripts\python.exe -m src.train_and_evaluate init --dataset data/builds/experiment-01 --run runs/experiment-01
```

Revision sumber: 8054ea98ed675005ec10e67359902f500e4911b0. Cache menyimpan hash file dan URL. ID ambigu tidak dicocokkan berdasarkan posisi baris. Sigma hanya memakai alias CAS yang semuanya menunjuk satu identitas terverifikasi.

Dataset terkini pada mesin ini: data/builds/audited-20260904-v2; run: runs/audited-20260904-v2. Berisi 6.703 molekul, 5.383 development dan 1.320 test, 25 label. Audit: reports/audit_20260904/dataset_audit.json.
Build awal tanpa akhiran v2 berasal dari versi RDKit awal dan tidak dipakai untuk eksperimen berikutnya.

Test dipisahkan lebih dahulu. Pemilihan 25 label memakai frekuensi development saja. Dari development, 15% khusus pemilihan threshold. Sisa development memakai tiga fold; setiap fold memiliki train, early-stop holdout, dan score holdout yang terpisah. Resampling hanya diterapkan ke train. Seluruh label ikut penilaian macro average precision (AP); AP bukan integrasi trapezoid kurva PR.

Sensitivitas scaffold tersedia melalui --split scaffold saat build. Pemisahan development ikut memakai kelompok scaffold; kelas yang tidak cukup pada suatu partisi menghentikan persiapan. Proporsi GroupShuffleSplit dihitung atas kelompok, sehingga persentase molekul dapat berbeda. Semua molekul acyclic berada pada scaffold kosong yang sama. Ini eksperimen tambahan, bukan pengganti otomatis stratifikasi multilabel. Untuk stabilitas, tetapkan seed tambahan (misalnya 42, 123, 2026) sebelum membuka test; --seed membuat dataset/run terpisah.

Unrecorded odor dianggap negatif karena sumber tidak menyediakan verifikasi negatif lengkap. Molekul yang sudah pernah dipakai dalam eksperimen lama dapat muncul dalam split baru: hasilnya bukan validasi eksternal independen.

## Training terkendali â€” belum dijalankan pada data nyata

Default: CPU, dua thread, satu trial sekaligus; sesi 30 menit; RAM tersedia minimal 4 GiB; RSS proses beserta anak maksimal 4 GiB. Suhu CPU >=90Â°C selama 30 detik menghentikan proses. Ini batas operasional konservatif proyek, bukan ambang kerusakan pabrikan. Pemeriksaan berlangsung pada callback; tidak menjamin batas RAM keras dari OS atau ketiadaan lonjakan di antara pemeriksaan.

Pada Windows, training membutuhkan sensor CPU yang benar-benar terhubung. --temperature-file menerima JSON yang diperbarui oleh pembaca sensor nyata:

```json
{"timestamp": 1788500000.0, "cpu_c": 72.5}
```

timestamp adalah Unix seconds, maksimal berumur 10 detik. Contoh di atas bukan telemetry aktif. Jangan membuat suhu palsu untuk melewati pemeriksaan. Jika sensor belum tersedia, perintah berhenti sebelum fitting. Pembatas ini tidak mengubah power plan, firmware, atau pengaturan sistem. Gunakan permukaan keras, ventilasi terbuka, dan power mode bawaan; hindari menjalankan build Android bersamaan dengan training.

Setelah sensor tersedia, perintah berikut adalah **tahap terpisah**, bukan satu rangkaian yang sudah dieksekusi:

```powershell
.\.venv\Scripts\python.exe -m src.train_and_evaluate baseline --run runs/experiment-01 --model xgb --temperature-file C:\sensors\cpu.json
.\.venv\Scripts\python.exe -m src.train_and_evaluate baseline --run runs/experiment-01 --model lgbm --temperature-file C:\sensors\cpu.json
.\.venv\Scripts\python.exe -m src.train_and_evaluate tune --run runs/experiment-01 --model xgb --trials 20 --temperature-file C:\sensors\cpu.json
.\.venv\Scripts\python.exe -m src.train_and_evaluate tune --run runs/experiment-01 --model lgbm --trials 20 --temperature-file C:\sensors\cpu.json
```

20 adalah contoh jumlah trial tambahan per pemanggilan. Jangan mengasumsikan keduanya menghabiskan waktu yang sama. Trial/score/durasi disimpan dalam SQLite dan CSV. TPE memulai dengan 10 startup trials; MedianPruner menilai agregat fold yang sebanding. Trial terputus tidak menjadi trial terbaik; trial yang selesai dan keadaan sampler tersimpan. Ulangi perintah tune untuk menambah trial pada studi yang sama. Perubahan kode, dependency, atau protokol meminta run baru. File pickle studi/model hanya untuk artefak lokal tepercaya.

Tiga kandidat default: tanpa resampling, class weighting, dan random oversampling. --include-mlsmote menambahkan adaptasi MLSMOTE sebagai ablation eksperimental. Bit sintetis menggunakan voting biner; descriptor numerik diinterpolasi. Molekul sintetis yang valid tidak dijamin; strategi ini tidak dianggap lebih baik sebelum dibuktikan pada validation.

| XGBoost | LightGBM |
|---|---|
| depth 3â€“8; min_child_weight 1â€“30 log | depth 4â€“12; num_leaves 8â€“min(128,2^depth); min_child_samples 10â€“150 |
| learning_rate .01â€“.2 log; gamma 0â€“5 | learning_rate .005â€“.15 log; min_split_gain 0â€“1 |
| subsample .6â€“1; colsample .4â€“1 | subsample .6â€“1; subsample_freq 1â€“7 aktif; colsample .5â€“1 |
| L1 opsional; L2 .001â€“100 log | L1 opsional; L2 .001â€“50 log; max_bin 63/127/255 |
| hist; batas 2.000 boosting rounds | gbdt; batas 3.000 boosting rounds |

Keduanya memakai logloss untuk early stopping (patience 50), kemudian macro AP untuk peringkat trial. Ruang dan batas di atas merupakan hipotesis eksperimen sesuai kontrol masing-masing learner, bukan rentang optimal yang dibuktikan untuk dataset ini.

## Fitting akhir dan test

```powershell
.\.venv\Scripts\python.exe -m src.train_and_evaluate fit --run runs/experiment-01 --model xgb --temperature-file C:\sensors\cpu.json
.\.venv\Scripts\python.exe -m src.train_and_evaluate fit --run runs/experiment-01 --model lgbm --temperature-file C:\sensors\cpu.json
.\.venv\Scripts\python.exe -m src.train_and_evaluate evaluate --run runs/experiment-01 --model xgb
.\.venv\Scripts\python.exe -m src.train_and_evaluate evaluate --run runs/experiment-01 --model lgbm
```

Fitting akhir memakai median jumlah boosting rounds terbaik antar-fold untuk setiap label. Progress tersimpan per model; jika berhenti, jalankan kembali fit dengan resep yang sama. Threshold dipilih pada holdout khusus, lalu disimpan dalam manifest. Kedua kandidat harus selesai sebelum salah satu test dibuka. Perintah evaluate membekukan hash kedua kandidat dan pilihan berdasarkan CV; setelah itu tuning/fitting pada run tersebut ditolak. Ini tidak menggantikan disiplin penelitian lintas run: jangan membuat run baru untuk mengejar skor test.

Metrik: macro/micro AP dan F1, precision/recall, subset accuracy, Hamming loss, ROC-AUC dan dukungan per label. Hasil lama di reports/ belum merupakan hasil protokol ini.

## Ekspor dan penjelasan

```powershell
.\.venv\Scripts\python.exe export_for_mobile.py --models runs/experiment-01/xgb --features data/builds/experiment-01/X_train.npz --output runs/experiment-01/mobile-bundle
.\.venv\Scripts\python.exe -m src.benchmark_onnx --bundle runs/experiment-01/mobile-bundle --features data/builds/experiment-01/X_train.npz --output runs/experiment-01/benchmark.json
.\.venv\Scripts\python.exe -m src.explain_model --run runs/experiment-01 --model xgb --output runs/experiment-01/explanations --temperature-file C:\sensors\cpu.json
.\.venv\Scripts\python.exe -m src.label_distance_viz --explanations runs/experiment-01/explanations --output runs/experiment-01/label-distances
```

Ganti xgb dengan lgbm untuk kandidat kedua. Ekspor memeriksa checksum, urutan label, jumlah fitur, ONNX checker, error numerik, dan keputusan threshold untuk seluruh model. manifest.json ditulis terakhir. Benchmark melaporkan seluruh label, load/first prediction/median/p95, ukuran dan RSS snapshot; bukan latensi Android atau keseluruhan permintaan jaringan.

TreeSHAP memakai sampel development dan seluruh label, memeriksa penjumlahan kontribusi terhadap raw margin. Heatmap cosine adalah tampilan utama; MDS 2D merupakan pendekatan dengan stress yang dilaporkan. Kedekatan penjelasan model bukan bukti kesamaan persepsi manusia. Morgan bit dapat mengalami collision.

## API dan katalog

REST lokal: .venv\\Scripts\\python.exe -m uvicorn api_light:app --host 127.0.0.1 --port 8000.
Gradio lokal: .venv\\Scripts\\python.exe app.py.
Docker menjalankan REST pada port 7860; Hugging Face Gradio menjalankan app.py.

REST GET / memberi schema ID; POST /fingerprint menerima smiles, compound_name opsional, dan feature_schema_id opsional. Gradio 5 memakai /gradio_api/call/predict. Keduanya mengembalikan schema ID, fitur, canonical SMILES, formula, MW, dan error terstruktur. SMILES langsung tidak membutuhkan PubChem. Tidak ada aturan MW>400 sebagai kepastian senyawa tidak volatil; tidak ada prediksi campuran.

ESSENZA_FEATURE_SPEC menunjuk feature_spec.json dari bundle yang akan dilayani. Default root feature_spec.json untuk bundle historis aplikasi yang kompatibilitas fiturnya sudah diperiksa. Deployment baru belum dilakukan.

Katalog baru harus berasal dari snapshot eksplisit:

```powershell
.\.venv\Scripts\python.exe -m src.export_catalog --input normalized-catalog.csv --output catalog-build --source-url https://source.example/catalog --source-version snapshot-id
```

CSV membutuhkan pid,brand,name,gender,rating,accords; accords berupa JSON object dengan bobot 0â€“1 yang berasal dari sumber. Tidak ada tebakan bobot dari nama notes. Katalog 2.029 parfum yang sedang dibundel dipertahankan dengan hash; raw export/provenance aslinya masih belum ditemukan.

Notebook 01_eda dan 02_modeling serta plot/ETL lama adalah arsip. Struktur notebook divalidasi; eksekusi historis sengaja diblokir agar tidak menimpa artefak atau memakai evaluasi lama.
