# 🧪 PROJECT CONTEXT — Essenza / Perfume-MultiLabel-Classifier
> **Untuk AI yang baru bergabung:** Baca dokumen ini dari awal sampai akhir sebelum menyentuh kode apapun.

---

## 📋 Ringkasan Singkat (TL;DR)

Proyek skripsi S1 milik **Marvel Kevin Nathanael** (Universitas Multimedia Nusantara).
Membangun sistem prediksi profil aroma parfum berbasis ML dalam aplikasi mobile Android **"Essenza"**.
User input SMILES → prediksi label aroma (floral, citrus, woody, dll) **100% on-device** tanpa internet untuk inferensi.

---

## 🗂️ Ekosistem Repositori

| Repo | Path Lokal | Peran |
|---|---|---|
| Perfume-MultiLabel-Classifier | C:\Users\marvel\AndroidStudioProjects\Perfume-MultiLabel-Classifier | **[UTAMA]** Backend ML + API |
| Essenza_Frontend | C:\Users\marvel\AndroidStudioProjects\Essenza_Frontend | **[UTAMA]** Android app (React Native) |
| AromaML | C:\Users\marvel\AndroidStudioProjects\AromaML | **[DEPRECATED]** Jangan diubah |

---

## 🌐 Backend API — STATUS TERKINI

### Hosting: Hugging Face Spaces (Gradio SDK)
- **URL:** https://marvelkn-essenza-fingerprint-api.hf.space
- **HF Space:** huggingface.co/spaces/marvelkn/essenza-fingerprint-api
- **SDK Version:** Gradio 4.44.0 (di README.md: sdk_version: "4.44.0")
- **Status saat ini:** ⚠️ DEBUGGING — server start tapi crash di TemplateResponse

### Riwayat hosting:
- Railway → ❌ Trial expired
- Render → ❌ Butuh CC
- HF Spaces → ⚠️ Sedang di-debug

### File kritis untuk HF deployment:
- **pp.py** — HF entry point. Berisi DUA monkey-patch wajib (lihat bawah)
- **pi_light.py** — FastAPI logic: compute_fingerprint(), 
ame_to_smiles()
- **equirements.txt** — **HANYA**: 
umpy, dkit, equests
  - ⚠️ JANGAN tambah fastapi/uvicorn/starlette → konflik dengan Gradio's own starlette

### Tiga bug HF environment yang sudah di-patch di app.py:
1. huggingface_hub >= 0.21 menghapus HfFolder → stub class ditambahkan sebelum import gradio
2. Starlette 1.x mengubah TemplateResponse(name, ctx) → TemplateResponse(req, name, ctx) → compat shim
3. astapi>=0.100 di requirements pull starlette baru → dihapus dari requirements.txt

### Cara push ke HF Space (repo bersih tanpa binary):
`powershell
cd "C:\Users\marvel\AppData\Local\Temp\hf-essenza"
Copy-Item "...\Perfume-MultiLabel-Classifier\app.py" . -Force
Copy-Item "...\Perfume-MultiLabel-Classifier\api_light.py" . -Force
Copy-Item "...\Perfume-MultiLabel-Classifier\requirements.txt" . -Force
Copy-Item "...\Perfume-MultiLabel-Classifier\README.md" . -Force
git add .; git commit -m "msg"
git push https://marvelkn:NEW_TOKEN@huggingface.co/spaces/marvelkn/essenza-fingerprint-api HEAD:main --force
`
⚠️ Token lama hf_GfljDT... sudah dishare di chat — HARUS sudah direvoke. Buat token baru.

---

## 📡 API Format (Mobile App)

Mobile app pakai Gradio endpoint, BUKAN REST biasa:

**Request:** POST /run/predict
`json
{ "data": ["O=Cc1ccc(O)c(OC)c1", ""] }
`
*(index 0: smiles, index 1: compound_name — kosong jika langsung SMILES)*

**Response:**
`json
{
  "data": [{
    "smiles": "O=Cc1ccc(O)c(OC)c1",
    "compound_name": "Unknown",
    "fingerprint": [0.0, 1.0, ...],
    "predictions": {},
    "molecular_weight": 152.15,
    "iupac_name": null,
    "molecular_formula": null,
    "warning": null
  }]
}
`

**InferenceService.ts** (di Essenza_Frontend dan AromaML):
`	ypescript
const API_URL = 'https://marvelkn-essenza-fingerprint-api.hf.space';
// Call: POST /run/predict dengan body { data: [smilesStr, ''] }
// Parse: response.data.data[0].fingerprint
`

---

## 🤖 ML Model

- **Input:** SMILES → Morgan Fingerprint 2048-bit (radius=2) + 5 physical descriptors = **2053 fitur**
- **Model:** 25 XGBoost models (Binary Relevance, 1 per label wewangian), format ONNX
- **Inference:** 100% on-device di HP via onnxruntime-react-native
- **Threshold:** Per-label, dari xgb_meta.json
- **Dataset:** 7036 senyawa, 111 label (Leffingwell taxonomy, via Pyrfume)

### Mixology (dot-concatenation):
SMILES campuran: mol1.mol2.mol3 → RDKit valid sebagai disconnected graph → FP gabungan
MW filter: dicek per-fragmen (max MW), bukan total → fix bug campuran 3 molekul kecil

---

## 📱 Fitur Essenza

### Explorer Mode (3 tabs):
- Search: cari parfum by nama/brand (perfumes.json, 2000+ entri, offline)
- By Label: filter by scent label via cosine similarity
- My Lab: CRUD custom perfume (AsyncStorage)

### Chemist Mode:
- Input SMILES manual
- Molecule chips (10 molekul dalam 5 kategori: Sweet, Citrus, Floral, Spicy/Woody, Musk)
- Dot-concatenation mixology
- Prediksi → bar chart probabilitas
- Molecule Info Card (formula, MW, IUPAC)

---

## 🔨 APK Build (Manual — Gradle GAGAL di Windows)

⚠️ ./gradlew assembleDebug GAGAL (bug file locking eact-android-0.86.0-debug.aar).

`powershell
# 1. Bundle JS
cd C:\Users\marvel\AndroidStudioProjects\Essenza_Frontend
npx react-native bundle --platform android --dev false --entry-file index.js --bundle-output android/app/src/main/assets/index.android.bundle --assets-dest android/app/src/main/res

# 2. Inject ke APK
cd android/app/src/main
jar uf C:\Users\marvel\AndroidStudioProjects\Essenza_Frontend\base.apk assets/index.android.bundle

# 3-5. Zipalign + Sign + Install
 = "C:\Users\marvel\AppData\Local\Android\Sdk\build-tools\36.1.0"
& "\zipalign.exe" -f -p 4 base.apk app-aligned.apk
& "\apksigner.bat" sign --ks "C:\Users\marvel\.android\debug.keystore" --ks-pass pass:android --key-pass pass:android --ks-key-alias androiddebugkey app-aligned.apk
& "C:\Users\marvel\AppData\Local\Android\Sdk\platform-tools\adb.exe" install -r app-aligned.apk
`
HP testing: RRCT903AH2M (Redmi)

---

## 📊 Progress Tracker

### ✅ Selesai
- Data Pipeline (7036 senyawa, 111 label)
- EDA + ML-SMOTE + XGBoost modeling
- ONNX Export (25 models)
- Explorer Mode (Search, By Label, My Lab)
- Chemist Mode + Scent Mixology UI
- MW filter fix (per-fragment)
- APK manual deploy pipeline
- Migrasi dari Railway ke HF Spaces (HF Space sudah dibuat, file sudah push)
- Update API_URL ke https://marvelkn-essenza-fingerprint-api.hf.space
- Update endpoint ke POST /run/predict (Gradio format)

### 🔴 Belum Selesai
- **Konfirmasi HF Space running** — monkey-patch di pp.py belum dikonfirmasi working
- **E2E test di HP** — semua flow termasuk API baru
- **Re-bundle + install APK** setelah API confirmed
- **Laporan Skripsi** — Bab 1, Bab 5 perlu update

---

## 🔑 Kenapa Begini?

| Keputusan | Alasan |
|---|---|
| ONNX inference di HP | Offline-first, kecepatan, privasi |
| Fingerprint di server | RDKit tidak bisa jalan di JS |
| JSON bukan SQLite | SQLite native module konflik di build |
| Dot-concatenation mixology | SMILES disconnected graph valid secara kimia |
| Gradio SDK bukan Docker | Docker HF butuh CC, Gradio gratis |
| /run/predict bukan /fingerprint | Gradio expose function via /run/predict |

---

## 📌 Last Session (28 Agustus 2026)

### Dikerjakan:
1. Migrasi Railway → HF Spaces (HF Space dibuat, clean repo di-push)
2. Debug 3 bug bertumpuk di HF environment (HfFolder, TemplateResponse, starlette conflict)
3. Update InferenceService.ts (URL + endpoint format)

### Status saat selesai:
- Build: ✅ sukses
- Server start: ✅ (Running on local URL: http://0.0.0.0:7860)
- Runtime: ❌ crash TypeError: unhashable type: 'dict' (TemplateResponse bug)
- Fix: monkey-patch shim sudah ada di app.py tapi **belum dikonfirmasi working**

### Pertama kali di sesi baru:
1. Push pp.py terbaru (sudah ada shim) ke HF Space → cek log
2. Test POST /run/predict via curl/browser
3. Re-bundle APK + install ke HP
4. E2E test

---

## 📞 Info
- **Mahasiswa:** Marvel Kevin Nathanael (UMN)
- **Skripsi:** C:\Users\marvel\Documents\UMN\Semester 7\2521_skripsi_Marvel_Kevin_Nathanael\
- **GitHub/HF:** marvelkn
- **HF Space:** huggingface.co/spaces/marvelkn/essenza-fingerprint-api
