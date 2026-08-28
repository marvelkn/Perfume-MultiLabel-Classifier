"""
Skrip modeling lengkap: ML-SMOTE → Binary Relevance (XGBoost & LightGBM)
→ Threshold tuning → Evaluasi → Export ONNX

Jalankan dengan:
    python -m src.train_and_evaluate

Output:
    reports/results_xgb.json
    reports/results_lgbm.json
    reports/comparison_table.csv
    models/xgb_aroma.onnx
    models/lgbm_aroma.onnx
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
import optuna

warnings.filterwarnings("ignore")

from .config import path, CONFIG
from .resampling import ml_smote


# ── Paths ─────────────────────────────────────────────────────────────────────
PROC = path("data_processed")
REPORTS = Path("reports")
MODELS = Path("models")
REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data():
    X_train = sp.load_npz(PROC / "X_train.npz").toarray().astype(np.float32)
    X_test  = sp.load_npz(PROC / "X_test.npz").toarray().astype(np.float32)
    Y_train = sp.load_npz(PROC / "Y_train.npz").toarray().astype(np.float32)
    Y_test  = sp.load_npz(PROC / "Y_test.npz").toarray().astype(np.float32)
    labels  = json.loads((PROC / "label_names.json").read_text())
    return X_train, X_test, Y_train, Y_test, labels


def find_best_threshold(y_true, y_prob):
    """Cari threshold yang memaksimalkan F1-score pada validation set."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def evaluate(Y_true, Y_pred_binary, Y_pred_proba, label_names):
    """Hitung semua metrik evaluasi."""
    return {
        "accuracy":  float(accuracy_score(Y_true, Y_pred_binary)),
        "precision": float(precision_score(Y_true, Y_pred_binary, average="macro", zero_division=0)),
        "recall":    float(recall_score(Y_true, Y_pred_binary, average="macro", zero_division=0)),
        "f1_macro":  float(f1_score(Y_true, Y_pred_binary, average="macro", zero_division=0)),
        "roc_auc":   float(roc_auc_score(Y_true, Y_pred_proba, average="macro")),
        "f1_per_label": {
            label: float(f1_score(Y_true[:, i], Y_pred_binary[:, i], zero_division=0))
            for i, label in enumerate(label_names)
        },
    }


def optimize_xgboost(X_tr, Y_tr, X_val, Y_val, labels, n_trials=30):
    from xgboost import XGBClassifier
    import random
    
    # Sample 8 representative labels for more stable optimization
    random.seed(42)
    sample_indices = random.sample(range(len(labels)), min(8, len(labels)))
    
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            # L1 & L2 regularization — critical for sparse 2053-dim fingerprint vectors
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "tree_method":      "hist",
            "eval_metric":      "logloss",
            "use_label_encoder": False,
            "verbosity":        0,
            "random_state":     42,
        }
        
        f1_scores = []
        for j in sample_indices:
            y_tr_j = Y_tr[:, j]
            y_val_j = Y_val[:, j]
            n_pos = y_tr_j.sum()
            n_neg = len(y_tr_j) - n_pos
            spw = max(1.0, n_neg / max(n_pos, 1))
            
            clf = XGBClassifier(scale_pos_weight=spw, **params)
            clf.fit(X_tr, y_tr_j)
            
            pv = clf.predict_proba(X_val)[:, 1]
            t_best = find_best_threshold(y_val_j, pv)
            f1 = f1_score(y_val_j, (pv >= t_best).astype(int), zero_division=0)
            f1_scores.append(f1)
            
        return np.mean(f1_scores)

    print("\n[Optuna] Mencari hyperparameter XGBoost terbaik (30 trials, 8 labels)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    print(f"  Best F1: {study.best_value:.4f}")
    print(f"  Best Params: {study.best_params}")
    return study.best_params


# ── XGBoost Binary Relevance ──────────────────────────────────────────────────
def train_xgboost(X_tr, Y_tr, X_val, Y_val, X_test, labels, best_params):
    from xgboost import XGBClassifier

    print("\n[XGBoost] Training Binary Relevance dengan Optuna Params ...")
    n_labels = Y_tr.shape[1]
    models, thresholds = [], []
    proba_val  = np.zeros((len(X_val),  n_labels))
    proba_test = np.zeros((len(X_test), n_labels))

    t0 = time.time()
    for j, label in enumerate(labels):
        y_tr_j  = Y_tr[:, j]
        y_val_j = Y_val[:, j]

        n_pos = y_tr_j.sum()
        n_neg = len(y_tr_j) - n_pos
        spw = max(1.0, n_neg / max(n_pos, 1))   # scale_pos_weight

        clf = XGBClassifier(
            scale_pos_weight=spw,
            tree_method="hist",
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
            **best_params
        )
        clf.fit(X_tr, y_tr_j)
        models.append(clf)

        pv = clf.predict_proba(X_val)[:, 1]
        pt = clf.predict_proba(X_test)[:, 1]
        proba_val[:, j]  = pv
        proba_test[:, j] = pt

        t_best = find_best_threshold(y_val_j, pv)
        thresholds.append(t_best)

        if (j + 1) % 20 == 0 or j == n_labels - 1:
            elapsed = time.time() - t0
            print(f"  Label {j+1}/{n_labels} selesai ({elapsed:.1f}s)")

    thresholds = np.array(thresholds)
    Y_pred = (proba_test >= thresholds).astype(int)

    # Simpan model dan threshold ke disk
    np.save(MODELS / "xgb_thresholds.npy", thresholds)
    xgb_model_dir = MODELS / "xgb_models"
    xgb_model_dir.mkdir(exist_ok=True)
    for j, (clf, label) in enumerate(zip(models, labels)):
        safe = label.replace(" ", "_").replace("/", "-")
        joblib.dump(clf, xgb_model_dir / f"xgb_{safe}.pkl")
    print(f"[XGBoost] Model tersimpan ke {xgb_model_dir}/")

    print(f"[XGBoost] Training selesai dalam {time.time()-t0:.1f}s")
    return models, thresholds, proba_test, Y_pred


# ── LightGBM Binary Relevance ─────────────────────────────────────────────────
def train_lightgbm(X_tr, Y_tr, X_val, Y_val, X_test, labels):
    from lightgbm import LGBMClassifier

    print("\n[LightGBM] Training Binary Relevance ...")
    n_labels = Y_tr.shape[1]
    models, thresholds = [], []
    proba_val  = np.zeros((len(X_val),  n_labels))
    proba_test = np.zeros((len(X_test), n_labels))

    t0 = time.time()
    for j, label in enumerate(labels):
        y_tr_j  = Y_tr[:, j]
        y_val_j = Y_val[:, j]

        clf = LGBMClassifier(
            n_estimators=300,
            num_leaves=63,
            learning_rate=0.07,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            verbosity=-1,
            random_state=42,
        )
        clf.fit(X_tr, y_tr_j)
        models.append(clf)

        pv = clf.predict_proba(X_val)[:, 1]
        pt = clf.predict_proba(X_test)[:, 1]
        proba_val[:, j]  = pv
        proba_test[:, j] = pt

        t_best = find_best_threshold(y_val_j, pv)
        thresholds.append(t_best)

        if (j + 1) % 20 == 0 or j == n_labels - 1:
            elapsed = time.time() - t0
            print(f"  Label {j+1}/{n_labels} selesai ({elapsed:.1f}s)")

    thresholds = np.array(thresholds)
    Y_pred = (proba_test >= thresholds).astype(int)

    np.save(MODELS / "lgbm_thresholds.npy", thresholds)
    lgbm_model_dir = MODELS / "lgbm_models"
    lgbm_model_dir.mkdir(exist_ok=True)
    for j, (clf, label) in enumerate(zip(models, labels)):
        safe = label.replace(" ", "_").replace("/", "-")
        joblib.dump(clf, lgbm_model_dir / f"lgbm_{safe}.pkl")
    print(f"[LightGBM] Model tersimpan ke {lgbm_model_dir}/")

    print(f"[LightGBM] Training selesai dalam {time.time()-t0:.1f}s")
    return models, thresholds, proba_test, Y_pred


# ── ONNX Export ───────────────────────────────────────────────────────────────
def export_onnx_xgb(models, labels, n_features):
    """Export XGBoost Binary Relevance ke ONNX."""
    try:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        print("\n[ONNX] Exporting XGBoost models ...")
        xgb_onnx_dir = MODELS / "xgb_onnx"
        xgb_onnx_dir.mkdir(exist_ok=True)

        for j, (clf, label) in enumerate(zip(models, labels)):
            initial_type = [("float_input", FloatTensorType([None, n_features]))]
            try:
                onnx_model = convert_xgboost(clf, initial_types=initial_type)
                fname = xgb_onnx_dir / f"xgb_{label.replace(' ', '_')}.onnx"
                fname.write_bytes(onnx_model.SerializeToString())
            except Exception as e:
                print(f"  Skip label {label}: {e}")

            if (j + 1) % 30 == 0:
                print(f"  Exported {j+1}/{len(labels)} models")

        n_ok = len(list(xgb_onnx_dir.glob("*.onnx")))
        print(f"[ONNX] XGBoost: {n_ok}/{len(labels)} models saved to {xgb_onnx_dir}/")
    except ImportError as e:
        print(f"[ONNX] Import error: {e}. Jalankan: pip install onnxmltools")


def export_onnx_lgbm(models, labels, n_features):
    """Export LightGBM Binary Relevance ke ONNX."""
    try:
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType

        print("\n[ONNX] Exporting LightGBM models ...")
        lgbm_onnx_dir = MODELS / "lgbm_onnx"
        lgbm_onnx_dir.mkdir(exist_ok=True)

        for j, (clf, label) in enumerate(zip(models, labels)):
            initial_type = [("float_input", FloatTensorType([None, n_features]))]
            try:
                onnx_model = convert_lightgbm(clf, initial_types=initial_type, zipmap=False)
                fname = lgbm_onnx_dir / f"lgbm_{label.replace(' ', '_')}.onnx"
                fname.write_bytes(onnx_model.SerializeToString())
            except Exception as e:
                print(f"  Skip label {label}: {e}")

            if (j + 1) % 30 == 0:
                print(f"  Exported {j+1}/{len(labels)} models")

        n_ok = len(list(lgbm_onnx_dir.glob("*.onnx")))
        print(f"[ONNX] LightGBM: {n_ok}/{len(labels)} models saved to {lgbm_onnx_dir}/")
    except ImportError as e:
        print(f"[ONNX] Import error: {e}. Jalankan: pip install onnxmltools")


# ── Main Orchestrator ─────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("MODELING PIPELINE: XGBoost & LightGBM Binary Relevance")
    print("=" * 60)

    # 1. Load data
    print("\n[1/7] Loading processed data ...")
    X_train, X_test, Y_train, Y_test, labels = load_data()
    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"  X_test : {X_test.shape},  Y_test : {Y_test.shape}")
    print(f"  Labels  : {len(labels)}")

    # 2. Carve validation set from train (for threshold tuning)
    print("\n[2/7] Membagi validation set dari training (80/20) ...")
    X_tr, X_val, Y_tr, Y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=42
    )
    print(f"  Train  : {X_tr.shape[0]} sampel")
    print(f"  Val    : {X_val.shape[0]} sampel")
    print(f"  Test   : {X_test.shape[0]} sampel (HOLD-OUT, tidak disentuh hingga evaluasi)")

    # 3. ML-SMOTE pada training portion only
    print("\n[3/7] Applying ML-SMOTE pada training set ...")
    X_tr_res, Y_tr_res = ml_smote(X_tr, Y_tr, k=5, sampling_ratio=0.8, seed=42)

    # 4. Train XGBoost
    xgb_params = optimize_xgboost(X_tr_res, Y_tr_res, X_val, Y_val, labels, n_trials=30)
    xgb_models, xgb_thresholds, xgb_proba, xgb_pred = train_xgboost(
        X_tr_res, Y_tr_res, X_val, Y_val, X_test, labels, xgb_params
    )

    # 5. Train LightGBM
    print("\n[5/7] Training LightGBM Binary Relevance ...")
    lgbm_models, lgbm_thresholds, lgbm_proba, lgbm_pred = train_lightgbm(
        X_tr_res, Y_tr_res, X_val, Y_val, X_test, labels
    )

    # 6. Evaluate on test set
    print("\n[6/7] Evaluasi pada test set (HOLD-OUT) ...")
    xgb_metrics  = evaluate(Y_test, xgb_pred,  xgb_proba,  labels)
    lgbm_metrics = evaluate(Y_test, lgbm_pred, lgbm_proba, labels)

    print("\n" + "=" * 50)
    print("HASIL EVALUASI")
    print("=" * 50)
    headers = ["Metrik", "XGBoost", "LightGBM"]
    rows = [
        ["Accuracy",  f"{xgb_metrics['accuracy']:.4f}",  f"{lgbm_metrics['accuracy']:.4f}"],
        ["Precision", f"{xgb_metrics['precision']:.4f}", f"{lgbm_metrics['precision']:.4f}"],
        ["Recall",    f"{xgb_metrics['recall']:.4f}",    f"{lgbm_metrics['recall']:.4f}"],
        ["F1-Macro",  f"{xgb_metrics['f1_macro']:.4f}",  f"{lgbm_metrics['f1_macro']:.4f}"],
        ["ROC-AUC",   f"{xgb_metrics['roc_auc']:.4f}",   f"{lgbm_metrics['roc_auc']:.4f}"],
    ]
    print(f"{'Metrik':<12} {'XGBoost':>10} {'LightGBM':>10}")
    print("-" * 34)
    for row in rows:
        print(f"{row[0]:<12} {row[1]:>10} {row[2]:>10}")

    # Simpan hasil
    (REPORTS / "results_xgb.json").write_text(
        json.dumps(xgb_metrics, indent=2), encoding="utf-8"
    )
    (REPORTS / "results_lgbm.json").write_text(
        json.dumps(lgbm_metrics, indent=2), encoding="utf-8"
    )

    comparison = pd.DataFrame({
        "Metrik":    ["Accuracy", "Precision", "Recall", "F1-Macro", "ROC-AUC"],
        "XGBoost":   [xgb_metrics[k]  for k in ["accuracy","precision","recall","f1_macro","roc_auc"]],
        "LightGBM":  [lgbm_metrics[k] for k in ["accuracy","precision","recall","f1_macro","roc_auc"]],
    })
    comparison.to_csv(REPORTS / "comparison_table.csv", index=False)
    print(f"\nHasil disimpan ke reports/")

    # 7. Export ONNX
    print("\n[7/7] Exporting ke ONNX ...")
    export_onnx_xgb(xgb_models, labels, X_train.shape[1])
    export_onnx_lgbm(lgbm_models, labels, X_train.shape[1])

    print("\n" + "=" * 60)
    print("PIPELINE SELESAI")
    print(f"  Tabel perbandingan : reports/comparison_table.csv")
    print(f"  Hasil XGBoost      : reports/results_xgb.json")
    print(f"  Hasil LightGBM     : reports/results_lgbm.json")
    print(f"  ONNX XGBoost       : models/xgb_onnx/")
    print(f"  ONNX LightGBM      : models/lgbm_onnx/")
    print("=" * 60)

    return xgb_metrics, lgbm_metrics


# ── Export-Only Mode ─────────────────────────────────────────────────────────
def export_only():
    """Load saved models dari disk dan ekspor ke ONNX tanpa re-training."""
    print("=" * 60)
    print("MODE: Export ONNX Only (load dari disk)")
    print("=" * 60)

    X_train, _, _, _, labels = load_data()
    n_features = X_train.shape[1]

    # Load XGBoost models
    xgb_model_dir = MODELS / "xgb_models"
    if not xgb_model_dir.exists():
        print("[ERROR] Folder models/xgb_models/ tidak ditemukan. Jalankan training dulu.")
    else:
        xgb_models = []
        for label in labels:
            safe = label.replace(" ", "_").replace("/", "-")
            pkl_path = xgb_model_dir / f"xgb_{safe}.pkl"
            if pkl_path.exists():
                xgb_models.append(joblib.load(pkl_path))
            else:
                print(f"  [SKIP] {pkl_path.name} tidak ditemukan")
                xgb_models.append(None)
        valid_xgb = [(m, l) for m, l in zip(xgb_models, labels) if m is not None]
        print(f"[XGBoost] Loaded {len(valid_xgb)}/{len(labels)} models dari disk")
        export_onnx_xgb([m for m, _ in valid_xgb], [l for _, l in valid_xgb], n_features)

    # Load LightGBM models
    lgbm_model_dir = MODELS / "lgbm_models"
    if not lgbm_model_dir.exists():
        print("[ERROR] Folder models/lgbm_models/ tidak ditemukan. Jalankan training dulu.")
    else:
        lgbm_models = []
        for label in labels:
            safe = label.replace(" ", "_").replace("/", "-")
            pkl_path = lgbm_model_dir / f"lgbm_{safe}.pkl"
            if pkl_path.exists():
                lgbm_models.append(joblib.load(pkl_path))
            else:
                print(f"  [SKIP] {pkl_path.name} tidak ditemukan")
                lgbm_models.append(None)
        valid_lgbm = [(m, l) for m, l in zip(lgbm_models, labels) if m is not None]
        print(f"[LightGBM] Loaded {len(valid_lgbm)}/{len(labels)} models dari disk")
        export_onnx_lgbm([m for m, _ in valid_lgbm], [l for _, l in valid_lgbm], n_features)

    print("\n" + "=" * 60)
    print("EXPORT SELESAI")
    print(f"  ONNX XGBoost  : models/xgb_onnx/")
    print(f"  ONNX LightGBM : models/lgbm_onnx/")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if "--export-only" in sys.argv:
        export_only()
    else:
        run()
