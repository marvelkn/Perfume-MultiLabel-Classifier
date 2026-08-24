# 🧪 PROJECT CONTEXT — Essenza / Perfume-MultiLabel-Classifier
> **Untuk AI yang baru bergabung:** Baca dokumen ini dari awal sampai akhir sebelum menyentuh kode apapun. Dokumen ini menjelaskan **apa proyek ini, mengapa dibuat, arsitekturnya secara menyeluruh, serta apa yang TERAKHIR dikerjakan dan apa yang harus dikerjakan selanjutnya.**

---

## 📋 Ringkasan Singkat (TL;DR)

Proyek skripsi S1 milik **Marvel Kevin Nathanael** (Universitas Multimedia Nusantara) yang bertujuan membangun sebuah sistem prediksi profil aroma parfum berbasis Machine Learning dan mengemasnya ke dalam aplikasi mobile Android bernama **"Essenza"**.

Pengguna cukup memasukkan struktur kimia sebuah senyawa (SMILES string), dan aplikasi akan memprediksi label-label aroma yang dikandungnya (misalnya: *floral, citrus, woody, sweet*) secara **langsung di HP** (on-device inference) tanpa memerlukan koneksi ke server untuk proses inferensi.

---

## 🗂️ Ekosistem Repositori (3 Repo)

Proyek ini terdiri dari **3 repositori GitHub** yang saling berkaitan:

| Repositori | Lokasi Lokal | Peran |
|---|---|---|
| `Perfume-MultiLabel-Classifier` | `C:\Users\marvel\AndroidStudioProjects\Perfume-MultiLabel-Classifier` | **[UTAMA]** Backend ML, Data Pipeline, FastAPI Server |
| `Essenza_Frontend` | `C:\Users\marvel\AndroidStudioProjects\Essenza_Frontend` | **[UTAMA]** Aplikasi Android (React Native) |
| `AromaML` | `C:\Users\marvel\AndroidStudioProjects\AromaML` | **[DEPRECATED]** Proyek lama, sudah digantikan oleh Essenza_Frontend |

> ⚠️ **AromaML sudah tidak digunakan.** Seluruh fiturnya telah dimigrasikan ke `Essenza_Frontend`. Jangan lakukan perubahan ke `AromaML`.

---

## 🏗️ Arsitektur Sistem End-to-End

```
┌─────────────────────────────────────────────────────────────┐
│                    APLIKASI MOBILE (Essenza)                │
│                    React Native (Android)                   │
│                                                             │
│  ┌──────────────┐        ┌──────────────────────────────┐  │
│  │ Explorer Tab │        │       Chemist Tab            │  │
│  │              │        │                              │  │
│  │ Search 2000+ │        │  1. User input SMILES atau   │  │
│  │ perfumes by  │        │     pilih dari Molecule Chips│  │
│  │ name/brand   │        │                              │  │
│  │              │        │  2. ─── HTTP POST ──────────►│  │
│  │ Filter by    │        │     /fingerprint             │  │
│  │ scent labels │        │     (Railway / HuggingFace)  │  │
│  │              │        │                              │  │
│  │ CRUD custom  │        │  3. ◄── Fingerprint [2053] ──│  │
│  │ perfumes     │        │                              │  │
│  │ (AsyncStorage│        │  4. On-Device ONNX Inference │  │
│  │  offline)    │        │     25 XGBoost models lokal  │  │
│  │              │        │                              │  │
│  │              │        │  5. Tampilkan hasil prediksi │  │
│  └──────────────┘        └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                                  │
                                  │ HTTP (hanya untuk fingerprint)
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│              BACKEND API (FastAPI)                          │
│              Host: Railway (sementara) → Migrasi ke HF      │
│              URL: https://perfume-multilabel-classifier-     │
│                   production.up.railway.app                  │
│                                                             │
│  Endpoint:                                                  │
│  GET  /            → health check                           │
│  POST /fingerprint → RDKit SMILES → Morgan FP [2053]        │
│  POST /recommend   → Cosine similarity (DEPRECATED di app)  │
│                                                             │
│  Library Kritis: rdkit-pypi, onnxruntime, fastapi           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Struktur Direktori Detail

### 1. `Perfume-MultiLabel-Classifier/` (Backend + ML Pipeline)

```
├── api.py                    # ⭐ FastAPI server utama (endpoint /fingerprint)
├── Procfile                  # Gunicorn command untuk Railway deployment
├── requirements.txt          # Dependensi production
├── requirements_dev.txt      # Dependensi dev (pytest, shap, etc.)
├── config.yaml               # Konfigurasi global (seed, n_bits, radius, dll)
├── etl_pipeline.py           # Pipeline ETL untuk build dataset parfum (Fragrantica)
├── export_for_mobile.py      # Script export model ke ONNX format
├── inspect_params.py         # Script debugging hyperparameter model
│
├── src/
│   ├── data_acquisition.py   # Download data dari Pyrfume (GoodScents + Leffingwell)
│   ├── featurize.py          # Konversi SMILES → Morgan Fingerprint (RDKit)
│   ├── label_harmonization.py # Mapping GoodScents → Leffingwell taxonomy
│   ├── splits.py             # Iterative-stratified train/test split
│   ├── build_dataset.py      # Orchestrator pipeline (jalankan: python -m src.build_dataset)
│   ├── resampling.py         # ML-SMOTE untuk handle imbalanced labels
│   ├── cross_validate.py     # Cross-validation script
│   ├── explain_model.py      # SHAP explainability analysis
│   ├── label_distance_viz.py # Visualisasi MDS label distance
│   └── build_perfume_db.py   # Build SQLite database parfum dari Fragrantica CSV
│
├── data/
│   └── processed/            # Output pipeline: X_train.npz, Y_train.npz, label_names.json
│
├── dataset/
│   └── perfume_db.sqlite     # Database 2000+ parfum Fragrantica (untuk /recommend)
│
├── models/
│   └── xgb_onnx/            # 25 file .onnx (satu per label) yang di-deploy ke mobile
│
├── mobile_assets/
│   ├── xgb_meta.json         # Metadata model: label names + threshold per model
│   └── leffingwell_to_fragrantica.json  # Crosswalk mapping untuk rekomendasi
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   └── 02_modeling.ipynb     # Training XGBoost + export ONNX
│
└── tests/
    └── test_api.py           # Pytest untuk endpoint API
```

### 2. `Essenza_Frontend/` (Aplikasi Android)

```
├── App.js                    # ⭐ Komponen utama (semua screen ada di sini)
├── index.js                  # Entry point React Native
├── package.json              # Dependensi RN: onnxruntime-react-native, axios, dll
│
├── src/
│   ├── services/
│   │   ├── InferenceService.ts    # ⭐ Logic ML: fetch fingerprint + run ONNX
│   │   └── DatabaseService.ts    # ⭐ Logic data: search/filter parfum + CRUD AsyncStorage
│   │
│   └── assets/
│       ├── perfumes.json          # Dataset 2000+ parfum (bundled, offline)
│       ├── metadata/
│       │   └── xgb_meta.json      # Threshold per label untuk ONNX inference
│       └── data/
│           └── ingredients.json   # Data bahan/senyawa parfum
│
└── android/
    └── app/src/main/
        ├── assets/
        │   ├── index.android.bundle      # ⭐ JS bundle (hasil: npx react-native bundle)
        │   └── models/                  # 25 file .onnx XGBoost (di-copy dari mobile_assets/)
        └── res/                         # Android resources
```

---

## 🤖 ML Model — Detail Teknis

### Dataset
- **Sumber Data:** GoodScents + Leffingwell + Arctander + Sigma + Flavornet (via Pyrfume)
- **Ukuran:** 7.036 senyawa, 111 label aroma (Leffingwell taxonomy)
- **Split:** Iterative-stratified 80/20 (train/test)

### Featurization
- **Input:** SMILES string (format struktur kimia standar)
- **Morgan Fingerprint:** 2048-bit (radius=2, via RDKit)
- **Physical Descriptors (5):** Molecular Weight, LogP, H-Donors, H-Acceptors, TPSA
- **Total input vector:** **2053 fitur** per senyawa

### Model
- **Algoritma:** XGBoost dengan Binary Relevance (1 model per label)
- **Jumlah model:** **25 model** (dari 111 label, dipilih yang paling relevan untuk wewangian)
- **Format deploy:** ONNX (via `skl2onnx`)
- **Threshold:** Per-label, disimpan di `xgb_meta.json`

### Inferensi di Mobile
Proses inferensi berjalan **100% offline** di HP:
1. **Fingerprint** dihitung via API (karena RDKit tidak bisa dijalankan di JavaScript)
2. ONNX models di-copy dari APK assets ke writable storage (`DocumentDirectoryPath`) pada first run
3. `onnxruntime-react-native` menjalankan setiap model secara berurutan
4. Hanya label yang probabilitasnya ≥ threshold yang ditampilkan ke user
5. Hasil diurutkan berdasarkan probabilitas tertinggi

---

## 📱 Aplikasi Essenza — Fitur Lengkap

### Layar Onboarding (Mode Selector)
Saat pertama buka, user memilih antara 2 mode:
- **🌸 Explorer Mode** → untuk user awam yang ingin browse/cari parfum
- **⚗️ Chemist Mode** → untuk user yang mau input SMILES dan analisis senyawa

### Explorer Mode
Tab-based interface dengan 3 tab:

| Tab | Fitur |
|---|---|
| 🔍 Search | Cari parfum berdasarkan nama atau brand (dari `perfumes.json`, 2000+ entri, **offline**) |
| 🏷️ By Label | Pilih scent labels (floral, citrus, dll) → tampilkan parfum yang cocok via cosine similarity |
| 🧪 My Lab | CRUD custom perfume — buat, edit, hapus formula parfum sendiri (disimpan di AsyncStorage) |

### Chemist Mode
- **Input SMILES:** Text field untuk input manual struktur kimia
- **Scent Mixology (FITUR BARU):** 10 molekul pre-defined dikelompokkan dalam 5 kategori notes yang bisa diklik:

| Kategori | Molekul |
|---|---|
| 🍬 Sweet / Gourmand | Vanillin, Coumarin |
| 🍋 Citrus | Limonene, Citral |
| 🌸 Floral | Linalool, Geraniol |
| 🪵 Spicy / Woody | Eugenol, Iso E Super |
| 🦨 Musk | Galaxolide |

- **Dot-concatenation Mixology:** Saat user klik beberapa chip, SMILES-nya digabungkan dengan notasi titik (`.`), contoh: `O=Cc1ccc(O)c(OC)c1.CC1=CCC(CC1)C(=C)C` (Vanillin + Limonene). Ini adalah representasi *disconnected graph* yang valid secara RDKit dan diproses sebagai campuran fisik.
- **Prediksi:** Tekan tombol "Predict Odor Profile" → API → ONNX → tampilkan bar chart probabilitas per label
- **Molecule Info Card:** Menampilkan Molecular Formula, Molecular Weight, IUPAC Name
- **Error Modal:** Pesan error user-friendly jika SMILES invalid atau senyawa terlalu berat

---

## 🌐 Backend API — Detail Endpoint

**Base URL (saat ini):** `https://perfume-multilabel-classifier-production.up.railway.app`

### `POST /fingerprint`
Endpoint utama yang digunakan oleh mobile app.

**Request:**
```json
{
  "smiles": "O=Cc1ccc(O)c(OC)c1",
  "compound_name": null
}
```

**Response:**
```json
{
  "smiles": "O=Cc1ccc(O)c(OC)c1",
  "compound_name": "Unknown",
  "fingerprint": [0.0, 1.0, 0.0, ...],  // array 2053 float
  "predictions": {
    "vanilla": {"probability": 0.87, "predicted": 1},
    "sweet": {"probability": 0.72, "predicted": 1}
  },
  "molecular_formula": null,
  "molecular_weight": null,
  "iupac_name": null,
  "warning": null
}
```

**Validasi di API:**
1. **Filter MW (per-fragment):** Dicek berat molekul **TERBESAR** dari setiap fragmen dalam campuran (bukan total). Jika ada fragmen > 400 g/mol, ditolak. *(Bug fix terakhir: sebelumnya menjumlahkan semua berat, sehingga campuran 3 molekul kecil ditolak.)*
2. **Filter AI Confidence:** Jika `max_probability < 0.3`, API mengirim `warning` ke client.

---

## 🚀 Deployment & Build Process

### Kondisi Saat Ini: Railway (Trial — Akan Habis)
- **Platform:** [Railway.app](https://railway.app)
- **Status:** ⚠️ **Trial credits hampir habis — perlu migrasi SEGERA**

### Alternatif yang Direkomendasikan: Hugging Face Spaces (Docker)
- **Kenapa HF Spaces?** Gratis permanen, RAM 16GB (lega untuk rdkit), 2 vCPU
- **Kenapa tidak Render?** RAM hanya 512MB — rdkit sangat boros memori, akan OOM/crash
- **Kenapa tidak Vercel?** Serverless function limit 50MB — rdkit saja >100MB
- **Trade-off HF Spaces:** Jika tidak ada traffic selama 48 jam, space tertidur. Perlu wake-up request pertama.

### Langkah Migrasi ke Hugging Face Spaces (BELUM DIKERJAKAN)
1. Buat akun di [huggingface.co](https://huggingface.co) jika belum ada
2. Buat **Docker Space** baru di bawah username `marvelkn`
3. Tambahkan `Dockerfile` ke root repo `Perfume-MultiLabel-Classifier`
4. Hubungkan HF Space ke GitHub repo ini (auto-deploy dari branch `main`)
5. Update `API_URL` di `Essenza_Frontend/src/services/InferenceService.ts` dari Railway URL ke HF Space URL (misal: `https://marvelkn-essenza-api.hf.space`)
6. Re-bundle dan re-install APK

### APK Build Process (Manual — Gradle Gagal di Windows)
> ⚠️ **PENTING:** `./gradlew assembleDebug` GAGAL di Windows karena bug file locking pada `react-android-0.86.0-debug.aar`. **Jangan coba Gradle.**

Gunakan **pipeline manual** berikut:

```powershell
# 1. Bundle JS
cd C:\Users\marvel\AndroidStudioProjects\Essenza_Frontend
npx react-native bundle --platform android --dev false --entry-file index.js `
  --bundle-output android/app/src/main/assets/index.android.bundle `
  --assets-dest android/app/src/main/res

# 2. Inject bundle ke APK (gunakan base.apk yang ada)
cd android/app/src/main
jar uf C:\Users\marvel\AndroidStudioProjects\Essenza_Frontend\base.apk assets/index.android.bundle

# 3. Zipalign
$zipalign = "C:\Users\marvel\AppData\Local\Android\Sdk\build-tools\36.1.0\zipalign.exe"
& $zipalign -f -p 4 base.apk app-aligned.apk

# 4. Sign dengan debug keystore
$apksigner = "C:\Users\marvel\AppData\Local\Android\Sdk\build-tools\36.1.0\apksigner.bat"
& $apksigner sign --ks "$env:USERPROFILE\.android\debug.keystore" `
  --ks-pass pass:android --key-pass pass:android `
  --ks-key-alias androiddebugkey app-aligned.apk

# 5. Install ke HP via ADB
$adb = "C:\Users\marvel\AppData\Local\Android\Sdk\platform-tools\adb.exe"
& $adb install -r app-aligned.apk
```

> **Tip:** Setelah install, tutup penuh (force close) aplikasi di HP lalu buka kembali agar tidak membaca cache bundle lama.

---

## 🛠️ Setup Environment Lokal

### Backend (Perfume-MultiLabel-Classifier)
```powershell
# Mesin ini tidak punya conda — gunakan venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Jalankan API lokal
python api.py
# Dokumentasi API: http://localhost:8000/docs
```

### Frontend (Essenza_Frontend)
```powershell
# Install dependencies
npm install

# Jalankan Metro bundler (untuk development)
npm start

# Jika HP terhubung via USB:
$adb = "C:\Users\marvel\AppData\Local\Android\Sdk\platform-tools\adb.exe"
& $adb reverse tcp:8081 tcp:8081
```

### ADB Setup (selalu jalankan di sesi PowerShell baru)
```powershell
$adb = "C:\Users\marvel\AppData\Local\Android\Sdk\platform-tools\adb.exe"
& $adb devices  # Pastikan HP terdeteksi sebagai "device"
```

---

## 📊 Progress Tracker Skripsi

### ✅ Selesai
- [x] **Data Pipeline** — 5 sumber data, 7036 senyawa, 111 label
- [x] **EDA** — Analisis distribusi label, imbalance ratio
- [x] **ML-SMOTE** — Handling imbalanced labels
- [x] **Modeling** — XGBoost Binary Relevance, tuning threshold per-label
- [x] **ONNX Export** — 25 model .onnx untuk 25 label wewangian utama
- [x] **Offline Data Layer** — 2000+ parfum bundled dalam `perfumes.json`
- [x] **Explorer Mode** — Search, filter by label, CRUD custom perfume (AsyncStorage)
- [x] **Chemist Mode** — SMILES input + API + on-device ONNX inference
- [x] **Scent Mixology UI** — Molecule chips dengan dot-concatenation logic
- [x] **API MW Filter Fix** — Validasi per-fragment (bukan total) untuk support mixology
- [x] **APK Manual Deploy** — Pipeline jar + zipalign + apksigner + adb install

### 🔴 Belum Selesai / Perlu Dikerjakan
- [ ] **Migrasi API ke Hugging Face Spaces** — Railway trial hampir habis
- [ ] **Update `API_URL`** di `InferenceService.ts` setelah migrasi HF Spaces
- [ ] **E2E Testing di HP** — Test semua flow: Search, Filter, CRUD, Mixology
- [ ] **Laporan Skripsi (LaTeX)** — Bab 1 dan Bab 5 perlu update untuk mencerminkan implementasi offline

---

## ⚙️ Dependency Penting

### Mobile (`Essenza_Frontend/package.json`)
| Package | Versi | Fungsi |
|---|---|---|
| `react-native` | 0.86.0 | Framework mobile |
| `onnxruntime-react-native` | ~1.20.0 | On-device ONNX inference |
| `react-native-fs` | ~2.23.0 | Akses filesystem (copy model ke writable dir) |
| `axios` | ~1.7.0 | HTTP client untuk panggil API fingerprint |
| `@react-native-async-storage/async-storage` | ~2.x | Penyimpanan custom perfume (CRUD) |

### Backend (`requirements.txt`)
| Package | Fungsi |
|---|---|
| `fastapi` | Web framework API |
| `uvicorn[standard]` | ASGI server |
| `rdkit-pypi` | Komputasi SMILES → Morgan Fingerprint |
| `onnxruntime` | ONNX inference di server (untuk /fingerprint endpoint preview) |
| `numpy`, `pandas` | Data manipulation |
| `scikit-learn` | Cosine similarity (rekomendasi) |

---

## 🔑 Keputusan Arsitektur Penting (Mengapa Begini?)

### Kenapa inferensi ONNX di HP, bukan di server?
**Jawaban:** Kecepatan + privasi + offline-first. Inferensi ONNX di HP hanya membutuhkan milidetik. Kalau inferensi dilakukan di server, setiap prediksi harus round-trip ke internet.

### Kenapa fingerprint tetap dihitung di server (API)?
**Jawaban:** RDKit ditulis dalam C++ dan tidak bisa dijalankan di JavaScript/React Native. Belum ada wrapper WASM yang mature dan ringan untuk RDKit.

### Kenapa parfum catalog disimpan sebagai JSON (bukan SQLite)?
**Jawaban:** Awalnya menggunakan `react-native-sqlite-storage` tapi gagal di build karena konflik native module. Migrasi ke JSON bundled + AsyncStorage untuk CRUD adalah keputusan pragmatis yang terbukti lebih stabil.

### Kenapa dot-concatenation (`.`) untuk Mixology?
**Jawaban:** Secara ilmiah, `mol1.mol2` dalam SMILES merepresentasikan *disconnected graph* — dua molekul yang ada dalam satu larutan tanpa ikatan kimia. RDKit memproses ini dan menghasilkan Morgan Fingerprint gabungan yang merepresentasikan "sidik jari" dari campuran tersebut. Ini adalah cara yang benar secara kimia tanpa perlu mengubah arsitektur model.

---

## 📌 Last Session Summary (Terakhir Dikerjakan: 20–24 Agustus 2026)

### Apa yang dikerjakan di sesi terakhir:
1. **Scent Mixology UI** — Menambahkan molekul chips ke `ChemistScreen` di `App.js`. Setiap kategori (`Sweet`, `Citrus`, `Floral`, `Spicy/Woody`, `Musk`) punya chip yang bisa diklik untuk menambah/menghapus SMILES dari input field.
2. **Fix Bug MW Filter** — Ditemukan bahwa campuran 3 molekul kecil (Vanillin + Limonene + Linalool, total ~442 g/mol) ditolak API karena sistem menjumlahkan berat semua komponen. Fix: gunakan `Chem.GetMolFrags()` untuk split fragmen, lalu validasi hanya berat fragmen terbesar (`max_wt`). Commit di-push ke GitHub → Railway auto-redeploy.
3. **APK Build & Install** — App.js di-bundle ulang via `npx react-native bundle`, di-inject ke `base.apk`, zipalign, sign, dan install ke HP fisik via ADB.

### Apa yang SEDANG jadi masalah:
- **Railway trial credit hampir habis.** API masih berfungsi di `https://perfume-multilabel-classifier-production.up.railway.app`, tapi perlu segera dipindahkan.
- **Solusi yang direkomendasikan:** Migrasi ke **Hugging Face Spaces** (Docker) — gratis permanen, RAM 16GB, tidak ada batas trial.
- Railway tidak bisa digantikan Vercel (limit 50MB) atau Render (RAM 512MB → OOM dengan rdkit).

### Langkah selanjutnya yang paling mendesak:
1. 🔴 **Buat Dockerfile** untuk `Perfume-MultiLabel-Classifier`
2. 🔴 **Buat HF Space** (Docker) di `huggingface.co/marvelkn`
3. 🔴 **Update `API_URL`** di `InferenceService.ts` ke URL HF Space yang baru
4. 🔴 **Re-bundle dan re-install APK** ke HP
5. 🟡 **Testing E2E** di HP — test semua flow termasuk Scent Mixology

---

## 📞 Kontak & Referensi
- **Mahasiswa:** Marvel Kevin Nathanael
- **Universitas:** Universitas Multimedia Nusantara (UMN)
- **Dokumen Skripsi:** `C:\Users\marvel\Documents\UMN\Semester 7\2521_skripsi_Marvel_Kevin_Nathanael\`
- **GitHub:** `marvelkn` (username)
- **Railway Dashboard:** [railway.app](https://railway.app) (akun yang terhubung ke repo ini)
