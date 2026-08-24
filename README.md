---
title: Essenza Fingerprint API
emoji: 🧪
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# 🧪 Essenza Fingerprint API

FastAPI backend untuk aplikasi mobile **Essenza** — sistem prediksi profil aroma parfum berbasis XGBoost + ONNX.

## Endpoints

| Method | Path | Deskripsi |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/fingerprint` | Konversi SMILES → Morgan Fingerprint [2053] |
| `POST` | `/recommend` | Rekomendasi parfum via cosine similarity |

## Contoh Request `/fingerprint`

```bash
curl -X POST https://marvelkn-essenza-api.hf.space/fingerprint \
  -H "Content-Type: application/json" \
  -d '{"smiles": "O=Cc1ccc(O)c(OC)c1"}'
```

## Teknologi

- **FastAPI** + **Uvicorn** — web framework & ASGI server  
- **RDKit** — komputasi Morgan Fingerprint dari SMILES  
- **ONNX Runtime** — inferensi 25 model XGBoost Binary Relevance  
- **Pandas + NumPy** — data manipulation  

## Catatan Mixology

API mendukung **SMILES campuran** (dot-notation): `mol1.mol2.mol3`  
Validasi MW dilakukan **per-fragmen** (bukan total), sehingga campuran 3 senyawa kecil tetap bisa diproses.
