"""
Export threshold per label dan daftar label ke JSON untuk dipakai oleh mobile app.
Jalankan: python export_for_mobile.py
"""
import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
MODELS = BASE / "models"
OUT = BASE / "mobile_assets"
OUT.mkdir(exist_ok=True)

# Derive label names from ONNX filenames (format: xgb_<label>.onnx)
xgb_onnx_dir = MODELS / "xgb_onnx"
onnx_files = sorted(xgb_onnx_dir.glob("*.onnx"))
labels = [f.stem.replace("xgb_", "").replace("_", " ") for f in onnx_files]

print(f"Derived {len(labels)} labels from ONNX filenames")

# Load XGBoost thresholds
xgb_thresh_path = MODELS / "xgb_thresholds.npy"
xgb_thresholds = np.load(xgb_thresh_path).tolist()
print(f"XGBoost thresholds: {len(xgb_thresholds)} values")

# Load LightGBM thresholds
lgbm_thresh_path = MODELS / "lgbm_thresholds.npy"
lgbm_thresholds = np.load(lgbm_thresh_path).tolist()
print(f"LightGBM thresholds: {len(lgbm_thresholds)} values")

# Buat metadata per label (label + threshold)
xgb_meta = [
    {"label": label, "threshold": float(thresh), "index": i}
    for i, (label, thresh) in enumerate(zip(labels, xgb_thresholds))
]

lgbm_meta = [
    {"label": label, "threshold": float(thresh), "index": i}
    for i, (label, thresh) in enumerate(zip(labels, lgbm_thresholds))
]

# Simpan ke JSON
with open(OUT / "xgb_meta.json", "w") as f:
    json.dump(xgb_meta, f, indent=2)

with open(OUT / "lgbm_meta.json", "w") as f:
    json.dump(lgbm_meta, f, indent=2)

with open(OUT / "labels.json", "w") as f:
    json.dump(labels, f, indent=2)

# Buat manifest ONNX files
xgb_onnx_dir = MODELS / "xgb_onnx"
onnx_files = sorted([f.name for f in xgb_onnx_dir.glob("*.onnx")])
with open(OUT / "xgb_manifest.json", "w") as f:
    json.dump(onnx_files, f, indent=2)

print(f"\n✅ Exported to {OUT}/")
print(f"   labels.json      : {len(labels)} labels")
print(f"   xgb_meta.json    : {len(xgb_meta)} entries (label + threshold)")
print(f"   lgbm_meta.json   : {len(lgbm_meta)} entries")
print(f"   xgb_manifest.json: {len(onnx_files)} ONNX files listed")
print(f"\nCopy folder 'models/xgb_onnx/' ke AromaML/assets/models/xgb/")
