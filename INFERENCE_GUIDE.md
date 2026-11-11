# Inferensi model final dan aplikasi lokal

Status arsitektur **7 November 2026**: **RDKit dan model LightGBM D asli berjalan di komputer lokal**. Android terhubung melalui USB, mengirim struktur satu molekul, menerima 109 skor, dan menyimpan Riwayat/Koleksi pada HP. Tidak ada inferensi estimator di HP. Setelah dependensi serta paket model tersedia, katalog, fitur, dan prediksi dapat berjalan tanpa hosting atau internet; komputer harus tetap menyala untuk prediksi baru.

## Model dan kontrak aktif

- Run **perfume-five-grouped-v7-m2048-d8**, kandidat **lgbm_D**, **109 classifier biner** hasil training yang sudah ada, berformat **joblib**. Tidak ada pelatihan atau penyimpanan ulang estimator saat pengemasan.
- **2.048 bit Morgan**, radius **2**, tanpa chirality; lalu **MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA, NumRotatableBonds, HeavyAtomCount, RingCount**. Total **2.056 fitur**, `float32`, satu molekul terhubung, seluruh nilai finite. Tidak ada scaler tambahan.
- Model memuat kelas `[0, 1]`; `predict_proba(..., num_threads=1)[:, 1]` mengambil skor kelas positif. Ambang utama **0,5**; tidak mengaktifkan threshold OOF tambahan atau memaksa top-k bila semua skor di bawah ambang.
- Skor belum dikalibrasi; bukan intensitas, komposisi, kadar, keamanan bahan, atau probabilitas persepsi manusia yang pasti. Prediksi campuran tidak didukung.

API aplikasi yang benar berada di **repo frontend**: `Essenza_Frontend/feature_api/app.py`, entrypoint **feature_api.app:app**. Model dimuat sekali saat startup oleh `feature_api/native_model.py`. Health hanya siap setelah paket berhasil diverifikasi dan dimuat.

## Menjalankan PC dan HP

Terminal pertama:

```powershell
Set-Location -LiteralPath 'C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend'
.\scripts\start-local-api.ps1
```

Terminal kedua, setelah USB debugging diizinkan:

```powershell
Set-Location -LiteralPath 'C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend'
.\scripts\run-local-android.ps1
```

API default `http://127.0.0.1:8000`. `adb reverse tcp:8000 tcp:8000` menghubungkan loopback HP ke PC. APK varian local membawa bundle JavaScript dan tidak membutuhkan Metro untuk digunakan. Riwayat/Koleksi tetap lokal di HP.

Setup memakai Python **3.12** dan paket terpin pada `Essenza_Frontend/feature_api/requirements*.txt`. Setup awal dapat membutuhkan internet untuk memasang dependensi. PubChem mati secara default; katalog dan input SMILES tidak memerlukannya. `-AllowPubChem` hanya mengaktifkan pencarian struktur tambahan dan memerlukan internet.

## Pengemasan dan verifikasi model

`scripts/prepare-local-models.py` membaca bundle serta completion manifest run v7, memverifikasi 109 checksum asli, kemudian menyalin byte model ke **feature_api/models/v7/**. Manifest aplikasi ada di **src/assets/metadata/v7_model_manifest.json** dan memakai ID paket native. Model biner tidak masuk Git.

Script startup menemukan run final pada repo penelitian di sebelah frontend. Bila lokasi berbeda, gunakan `start-local-api.ps1 -SourceRun 'C:\path\ke\runs\perfume-five-grouped-v7-m2048-d8'`. Jangan memilih eksperimen terbaru otomatis. Setelah paket tersalin, operasi normal hanya memerlukan frontend, environment API, dan paket model; sumber penelitian tidak dibutuhkan untuk startup berikutnya.

Pemeriksaan paket dan konsistensi, dari root frontend:

```powershell
.\.venv-feature-api\Scripts\python.exe -B scripts\prepare-local-models.py --verify-only
.\.venv-feature-api\Scripts\python.exe -B scripts\verify-local-predictions.py --api-url http://127.0.0.1:8000 --output .local-run\native-parity.json
```

Audit kedua membutuhkan sumber model dan vektor penelitian beku. Ia membandingkan model salinan dengan model sumber, fitur RDKit dengan vektor train beku, skor referensi Vanillin/Ethanol, dan respons HTTP. Keberhasilan prosedur harus dibaca dari JSON hasil aktual; perintah di atas bukan klaim bahwa pengujian sudah lulus. Konsistensi runtime bukan ukuran kualitas aroma baru atau validasi sensori.

## Kontrak API aplikasi

| Endpoint | Fungsi |
| --- | --- |
| `GET /health` | Status kesiapan model, runtime LightGBM, ID paket/skema, 2.056 fitur, 109 label, dan threshold 0,5. |
| `POST /resolve` | Mencocokkan nama/rumus dan mengembalikan kandidat struktur untuk dipilih eksplisit. |
| `POST /predict` | Memeriksa `bundle_id` dan `feature_schema_id`, menghitung fitur, menjalankan model asli, mengembalikan 109 skor berurutan. Tidak mengirim vektor fitur ke HP untuk inferensi. |
| `POST /fingerprint` | Diagnosis/validasi fitur; bukan endpoint prediksi yang dipakai frontend aktif. |

## Jalur penelitian dan kompatibilitas historis

Notebook 08 dan `python -B -m research.inference --smiles 'COc1cc(C=O)ccc1O'` menjalankan inferensi Python dari run v7 dalam environment penelitian. Kode utama tetap **KODE_ML/03_Kode/**; data, label, grup/split, hyperparameter, model, dan angka hasil tidak diubah oleh migrasi aplikasi.

**Jangan memakai `src.inference.Predictor`, `src.release`, `api_light.py`, `api.py`, `app.py`, atau `feature_spec.json` root backend untuk aplikasi v7.** Jalur tersebut adalah kompatibilitas **grouped-v6 / 1.029 fitur**. Bukti di `reports/backend_alignment_20260921` berlaku untuk jalur v6 pada tanggal tersebut, bukan bukti migrasi aplikasi sekarang.

`requirements_lock.txt`, completion manifests, protokol, arsip, dan audit bertanggal adalah rekaman lingkungan/implementasi historis. Jangan menginstal API baru dari freeze tersebut atau menghapus/mengubah hasil penelitian untuk menghilangkan kata terkait teknologi lama. `requirements_research.txt` mendefinisikan lingkungan penelitian aktif; requirements API aplikasi dipisahkan agar tidak membawa alat tuning/konversi yang tidak dipakai untuk serving.
