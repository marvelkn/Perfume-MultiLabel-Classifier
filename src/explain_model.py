"""
Explainable AI (XAI) menggunakan SHAP untuk menjelaskan prediksi XGBoost.
Jalankan: python -m src.explain_model
"""
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from .train_and_evaluate import load_data

REPORTS = Path("reports")
MODELS = Path("models")

def run_shap():
    print("=" * 60)
    print("EXPLAINABLE AI (SHAP)")
    print("=" * 60)
    
    # 1. Load Data
    _, X_test, _, _, labels = load_data()
    
    # Generate feature names
    # Morgan bits: 0 to 2047
    feature_names = [f"Morgan_Bit_{i}" for i in range(2048)]
    # 5 Physical features
    feature_names.extend(["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA"])
    
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Ambil 3 label paling populer untuk dijelaskan
    # Cari modelnya di folder
    xgb_model_dir = MODELS / "xgb_models"
    
    if not xgb_model_dir.exists():
        print("Folder model tidak ditemukan. Jalankan training terlebih dahulu.")
        return
        
    # Contoh label: "floral", "citrus", "woody"
    target_labels = ["floral", "citrus", "woody"]
    
    for label in target_labels:
        if label not in labels:
            continue
            
        print(f"\nMenghitung SHAP values untuk aroma: {label.upper()}...")
        safe_label = label.replace(" ", "_").replace("/", "-")
        pkl_path = xgb_model_dir / f"xgb_{safe_label}.pkl"
        
        if not pkl_path.exists():
            print(f"Model untuk {label} tidak ditemukan.")
            continue
            
        model = joblib.load(pkl_path)
        
        # Ambil sampel 200 data untuk mempercepat kalkulasi
        X_sample = X_test_df.sample(n=min(200, len(X_test_df)), random_state=42)
        
        # XGBoost 2.0+ workaround: Gunakan native pred_contribs dari booster
        import xgboost as xgb
        dm = xgb.DMatrix(X_sample)
        contribs = model.get_booster().predict(dm, pred_contribs=True)
        shap_values = contribs[:, :-1] # kolom terakhir adalah bias/base_value
        
        # Buat summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        
        out_path = REPORTS / f"shap_summary_{safe_label}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grafik SHAP berhasil disimpan di: {out_path}")

    print("\nSelesai! Buka folder reports/ untuk melihat grafik penjelasan AI.")

if __name__ == "__main__":
    run_shap()
