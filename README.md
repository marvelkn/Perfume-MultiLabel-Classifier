# Perfume Aroma-Profile Prediction — Gradient Boosting (Marvel Kevin Nathanael)

Predict the multi-label odor profile of perfume molecules from their structure
(SMILES → Morgan fingerprint → **XGBoost & LightGBM**, Binary Relevance), and ship the
best model to a **React Native** app via on-device **ONNX Runtime**.

This repo currently contains the **shared data layer** (built by the scaffold). Modeling,
ONNX export, and the mobile app are the thesis contribution and start downstream.

## 🚀 Progress Tracker
- [x] **Step 1: Data Pipeline Execution** (DONE - EXPANDED) - 5 sumber data: GoodScents + Leffingwell + Arctander + Sigma + Flavornet. Morgan fingerprinting menghasilkan **X=(7036, 2048), Y=(7036, 111)**. Iterative-Stratified 80/20 Split. Artefak tersimpan di `data/processed/`.
- [x] **Step 2: Exploratory Data Analysis (EDA)** - Review `notebooks/01_eda.ipynb` untuk melihat statistik sebaran label data.
- [x] **Step 3: Imbalance Handling (ML-SMOTE)** - Implementasi fungsi pada `src/resampling.py`.
- [x] **Step 4: Modeling (XGBoost & LightGBM)** - Training model menggunakan `notebooks/02_modeling.ipynb`.
- [x] **Step 5: Model Export (ONNX)** - Export model ke format ONNX untuk dipakai di React Native.

## Data
GS-LF = `goodscents` + `leffingwell` Pyrfume archives. Label taxonomy is anchored on
Leffingwell's curated odor classes; GoodScents free-text descriptors are mapped onto it.

## Setup
This machine has **no conda**, so use a venv:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
For Google Colab / conda machines, use `environment.yml` instead
(`conda env create -f environment.yml && conda activate skripsi-aroma`).

## Run the data pipeline
```powershell
python -m src.build_dataset      # downloads GS-LF, harmonizes, fingerprints, splits
pytest                            # unit tests for featurization
jupyter lab notebooks/01_eda.ipynb
```
Outputs land in `data/processed/`: `X_train/X_test.npz`, `Y_train/Y_test.npz`,
`label_names.json`, `smiles_*.txt`, `dataset_summary.json`.

## Layout
```
config.yaml            # single source of truth (seed, fingerprint, split, paths)
src/
  config.py            # config loader
  data_acquisition.py  # pyrfume download (+ raw-GitHub fallback)
  featurize.py         # canonical SMILES + Morgan fingerprint (RDKit)
  label_harmonization.py  # GoodScents free-text -> Leffingwell taxonomy
  splits.py            # iterative-stratified 80/20 split
  build_dataset.py     # orchestrator (python -m src.build_dataset)
  resampling.py        # ML-SMOTE stub (implement at modeling time)
notebooks/
  01_eda.ipynb         # imbalance / cardinality / stratification / unmapped report
  02_modeling.ipynb    # YOUR scaffold: BR + XGB/LGBM + thresholds + ONNX
tests/test_featurize.py
```

## Handoff boundary
The data layer ends at the saved 80/20 split. Your work begins at **class-imbalance
handling** (ML-SMOTE on train only) and **model training** — see `notebooks/02_modeling.ipynb`
and the plan at `~/.claude/plans/`.

---

## 🌐 Server Deployment (Railway API)
This repository also serves as the **backend API** deployed on Railway. 
The mobile app (React Native) uses this API solely to compute the **Morgan Fingerprint (RDKit)** and 5 physical descriptors of a given SMILES string.

**Kenapa harus di-host di Railway?**
Karena *RDKit* (pustaka kemoinformatika yang diperlukan untuk memproses SMILES ke dalam *fingerprint*) ditulis dalam bahasa C++ & Python dan tidak dapat dieksekusi secara native/lokal di dalam *React Native/JavaScript*.

**File Relevan untuk Deployment:**
- `api.py`: FastAPI server yang terekspos ke internet. Menerima request berisi `smiles` dan merespon dengan vektor numerik (fingerprint).
- `Procfile`: Command untuk menjalankan Gunicorn + Uvicorn di Railway.
- `requirements.txt`: Dependensi production (FastAPI, RDKit, dll).

## 📱 Mobile App (Essenza)
Aplikasi *mobile* React Native yang mengkonsumsi API ini dan menjalankan model ONNX secara on-device berada di repository **[Essenza_Frontend](../Essenza_Frontend)**. 

*(AromaML sebelumnya digunakan sebagai tempat riset, tetapi sekarang seluruh fiturnya sudah diintegrasikan ke Essenza_Frontend).*
