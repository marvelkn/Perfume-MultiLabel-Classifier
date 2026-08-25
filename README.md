---
title: Essenza Fingerprint API
emoji: 🧪
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: mit
short_description: RDKit Morgan Fingerprint API for Essenza mobile app
---

# 🧪 Essenza Fingerprint API

FastAPI + Gradio backend untuk aplikasi mobile **Essenza** — sistem prediksi profil aroma parfum berbasis XGBoost + ONNX.

## Endpoint

| Method | Path | Deskripsi |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/fingerprint` | SMILES → Morgan Fingerprint [2053] |
| `GET` | `/ui` | Gradio UI untuk testing manual |

## Contoh Request

```bash
curl -X POST https://marvelkn-essenza-fingerprint-api.hf.space/fingerprint \
  -H "Content-Type: application/json" \
  -d '{"smiles": "O=Cc1ccc(O)c(OC)c1"}'
```

## Teknologi

- **FastAPI** + **Uvicorn** — web framework & ASGI server
- **RDKit** — komputasi Morgan Fingerprint dari SMILES
- **Gradio** — UI untuk testing + HF Spaces compatibility
