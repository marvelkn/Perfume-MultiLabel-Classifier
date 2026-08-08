"""
K-Fold Cross Validation untuk mengevaluasi stabilitas model.
Jalankan: python -m src.cross_validate
"""
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from pathlib import Path
import json
import time

from .config import path
from .resampling import ml_smote
from .train_and_evaluate import load_data, find_best_threshold

REPORTS = Path("reports")

def run_cv():
    print("=" * 60)
    print("5-FOLD CROSS VALIDATION (XGBoost)")
    print("=" * 60)
    
    # 1. Load Data
    X_train, X_test, Y_train, Y_test, labels = load_data()
    # Combine back for CV
    X_all = np.vstack([X_train, X_test])
    Y_all = np.vstack([Y_train, Y_test])
    
    print(f"Total dataset: {X_all.shape[0]} sampel")
    
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_auc = []
    fold_f1 = []
    
    # Pilih 5 label terpopuler untuk menghemat waktu CV
    label_sums = Y_all.sum(axis=0)
    top_indices = np.argsort(label_sums)[::-1][:5]
    top_labels = [labels[i] for i in top_indices]
    print(f"\nMenjalankan CV untuk 5 label terpopuler: {top_labels}")
    
    for fold, (train_idx, val_idx) in enumerate(mskf.split(X_all, Y_all), 1):
        t0 = time.time()
        print(f"\n--- Fold {fold} ---")
        X_tr, Y_tr = X_all[train_idx], Y_all[train_idx]
        X_val, Y_val = X_all[val_idx], Y_all[val_idx]
        
        # Terapkan ML-SMOTE hanya pada data train
        X_tr_res, Y_tr_res = ml_smote(X_tr, Y_tr, k=5, sampling_ratio=0.5, seed=42)
        
        y_val_preds = []
        y_val_probs = []
        y_val_true = []
        
        for j in top_indices:
            y_tr_j = Y_tr_res[:, j]
            y_val_j = Y_val[:, j]
            y_val_true.append(y_val_j)
            
            n_pos = y_tr_j.sum()
            n_neg = len(y_tr_j) - n_pos
            spw = max(1.0, n_neg / max(n_pos, 1))
            
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=spw,
                tree_method="hist",
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
                random_state=42,
            )
            clf.fit(X_tr_res, y_tr_j)
            
            pv = clf.predict_proba(X_val)[:, 1]
            t_best = find_best_threshold(y_val_j, pv)
            pred = (pv >= t_best).astype(int)
            
            y_val_probs.append(pv)
            y_val_preds.append(pred)
            
        y_val_true = np.column_stack(y_val_true)
        y_val_probs = np.column_stack(y_val_probs)
        y_val_preds = np.column_stack(y_val_preds)
        
        auc = roc_auc_score(y_val_true, y_val_probs, average="macro")
        f1 = f1_score(y_val_true, y_val_preds, average="macro")
        
        fold_auc.append(auc)
        fold_f1.append(f1)
        
        print(f"Fold {fold} Selesai dalam {time.time()-t0:.1f}s | AUC: {auc:.4f}, F1: {f1:.4f}")

    print("\n" + "=" * 60)
    print("HASIL AKHIR CROSS VALIDATION (5-Folds)")
    print("=" * 60)
    print(f"ROC-AUC: {np.mean(fold_auc):.4f} ± {np.std(fold_auc):.4f}")
    print(f"F1-Score: {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")
    
    results = pd.DataFrame({
        "Fold": [1,2,3,4,5],
        "ROC-AUC": fold_auc,
        "F1-Macro": fold_f1
    })
    results.to_csv(REPORTS / "cv_results.csv", index=False)
    print("Hasil disimpan di reports/cv_results.csv")

if __name__ == "__main__":
    run_cv()
