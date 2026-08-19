"""
Visualisasi Hubungan Struktural Antar Label Aroma menggunakan MDS + UMAP.

Mengadaptasi metodologi Suh et al. (2025) Fig. 4:
  1. Hitung mean |SHAP| per label → vektor kepentingan 2053-dim
  2. Hitung pairwise Cosine Distance antar 25 vektor → matrix 25x25
  3. UMAP: kompresi matrix jarak → embedding 10-dim
  4. MDS:  kompresi 10-dim → koordinat 2D
  5. Visualisasi scatter plot 2D berlabel & berwarna per grup aroma

Jalankan:
    python -m src.label_distance_viz
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy.sparse as sp
import xgboost as xgb
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances
from umap import UMAP

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC    = Path("data/processed")
MODELS  = Path("models/xgb_models")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

# ── Pengelompokan warna per grup aroma (dapat disesuaikan) ───────────────────
ODOR_GROUPS = {
    "Floral":        {"labels": ["floral", "rose", "jasmine", "violet", "orris"],
                      "color": "#E91E8C"},
    "Citrus/Fresh":  {"labels": ["citrus", "fresh", "lemon", "grapefruit", "green"],
                      "color": "#FFB300"},
    "Fruity":        {"labels": ["fruity", "apple", "tropical", "berry", "peach",
                                 "cherry", "pineapple", "banana", "melon"],
                      "color": "#FF6D00"},
    "Woody/Earthy":  {"labels": ["woody", "earthy", "spicy", "herbal", "mint",
                                 "camphoreous"],
                      "color": "#795548"},
    "Sweet/Balsamic":{"labels": ["sweet", "balsamic", "vanilla", "caramellic",
                                 "coumarinic"],
                      "color": "#9C27B0"},
    "Savory/Animal": {"labels": ["fatty", "oily", "waxy", "sulfurous", "meaty",
                                 "ethereal", "winey"],
                      "color": "#607D8B"},
}

def get_group_color(label: str) -> tuple[str, str]:
    """Return (group_name, hex_color) for a given label."""
    for group_name, info in ODOR_GROUPS.items():
        if label in info["labels"]:
            return group_name, info["color"]
    return "Other", "#9E9E9E"


def build_feature_names() -> list[str]:
    names = [f"Morgan_Bit_{i}" for i in range(2048)]
    names += ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA"]
    return names


def compute_mean_shap_per_label(
    X_test: np.ndarray,
    labels: list[str],
    n_sample: int = 300,
) -> np.ndarray:
    """
    Untuk setiap label, hitung rata-rata |SHAP value| atas sampel X_test.
    Returns shape: (n_labels, n_features)
    """
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=min(n_sample, len(X_test)), replace=False)
    X_sample = X_test[idx]

    feature_names = build_feature_names()
    import pandas as pd
    X_df = pd.DataFrame(X_sample, columns=feature_names)

    mean_shap_matrix = []
    n = len(labels)

    for i, label in enumerate(labels):
        pkl_path = MODELS / f"xgb_{label}.pkl"
        if not pkl_path.exists():
            print(f"  [SKIP] Model tidak ditemukan: {pkl_path}")
            mean_shap_matrix.append(np.zeros(X_sample.shape[1]))
            continue

        print(f"  [{i+1:2d}/{n}] Menghitung SHAP untuk '{label}'...")
        model = joblib.load(pkl_path)
        dm = xgb.DMatrix(X_df)
        contribs = model.get_booster().predict(dm, pred_contribs=True)
        shap_vals = contribs[:, :-1]          # buang kolom bias
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        mean_shap_matrix.append(mean_abs)

    return np.array(mean_shap_matrix)          # shape: (n_labels, 2053)


def run():
    print("=" * 60)
    print("VISUALISASI MDS + UMAP — HUBUNGAN ANTAR LABEL AROMA")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Memuat data test dan nama label...")
    X_test = sp.load_npz(PROC / "X_test.npz").toarray().astype(np.float32)
    labels: list[str] = json.loads((PROC / "label_names.json").read_text())
    print(f"      {len(labels)} label ditemukan: {labels}")

    # 2. Hitung mean |SHAP| per label
    print(f"\n[2/5] Menghitung mean |SHAP| untuk {len(labels)} label...")
    mean_shap = compute_mean_shap_per_label(X_test, labels, n_sample=300)

    # 3. Cosine Distance matrix
    print("\n[3/5] Menghitung Cosine Distance matrix antar label...")
    dist_matrix = cosine_distances(mean_shap)   # shape: (n_labels, n_labels)

    # 4. UMAP -> dimensi 10
    print("\n[4/5] Menerapkan UMAP (metric=precomputed) -> 10-dim...")
    umap_model = UMAP(
        n_components=10,
        metric="precomputed",
        random_state=42,
        n_neighbors=min(5, len(labels) - 1),
    )
    umap_embedding = umap_model.fit_transform(dist_matrix)

    # 5. MDS -> 2D
    print("      Menerapkan MDS -> 2D koordinat..."
    )
    mds = MDS(n_components=2, dissimilarity="euclidean", random_state=42, n_init=10)
    coords_2d = mds.fit_transform(umap_embedding)   # shape: (n_labels, 2)

    # 6. Plot
    print("\n[5/5] Membuat visualisasi scatter plot 2D...")
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    plotted_groups = {}
    for i, label in enumerate(labels):
        group_name, color = get_group_color(label)
        x, y = coords_2d[i, 0], coords_2d[i, 1]
        ax.scatter(x, y, c=color, s=180, zorder=3, edgecolors="white", linewidths=1.5)
        ax.annotate(
            label,
            (x, y),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )
        plotted_groups[group_name] = color

    # Legend
    legend_patches = [
        mpatches.Patch(color=col, label=grp)
        for grp, col in plotted_groups.items()
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        framealpha=0.9,
        fontsize=9,
        title="Grup Aroma",
        title_fontsize=10,
    )

    ax.set_title(
        "Peta Hubungan Struktural Antar Label Aroma\n"
        "(MDS atas UMAP Embedding Vektor Mean |SHAP| XGBoost)",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color="#222222",
    )
    ax.set_xlabel("MDS Dimensi 1", fontsize=10, color="#555555")
    ax.set_ylabel("MDS Dimensi 2", fontsize=10, color="#555555")
    ax.grid(True, linestyle="--", alpha=0.4, color="#CCCCCC")
    ax.spines[["top", "right"]].set_visible(False)

    out_path = REPORTS / "label_distance_mds.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] Visualisasi berhasil disimpan di: {out_path}")
    print("   Buka file tersebut untuk melihat peta inter-label aroma.\n")


if __name__ == "__main__":
    run()
