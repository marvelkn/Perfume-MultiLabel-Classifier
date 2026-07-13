"""ML-SMOTE untuk data multi-label (Charte et al. 2015, Knowledge-Based Systems).

Prinsip:
  - Identifikasi sampel minoritas berdasarkan jumlah label aktif vs rata-rata dataset.
  - Untuk setiap sampel minoritas, cari k tetangga terdekat di ruang fitur.
  - Buat sampel sintetis baru dengan interpolasi linear fitur + union label.

ATURAN ANTI-LEAKAGE (NON-NEGOTIABLE):
  - Panggil HANYA pada X_train / Y_train setelah train/test split.
  - Jangan pernah sentuh X_test / Y_test di sini.
  - Jika memakai cross-validation, terapkan per fold (bukan secara global).
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def ml_smote(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 5,
    sampling_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """ML-SMOTE: Over-sampling untuk data multi-label.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Matriks fitur (Morgan Fingerprint). Harus dense array.
    Y : np.ndarray, shape (n_samples, n_labels)
        Matriks label binary (0/1).
    k : int
        Jumlah tetangga terdekat untuk interpolasi (default=5).
    sampling_ratio : float
        Berapa kali total sampel minoritas yang akan di-sintetis.
        0.5 = tambah 50% dari jumlah sampel minoritas.
        1.0 = tambah 100% (double semua sampel minoritas).
    seed : int
        Random seed untuk reproduksibilitas.

    Returns
    -------
    X_out, Y_out : np.ndarray
        Dataset augmented (original + sintetis).
    """
    rng = np.random.default_rng(seed)

    n_samples, n_labels = Y.shape
    labels_per_sample = Y.sum(axis=1)          # jumlah label aktif per sampel
    mean_labels = labels_per_sample.mean()      # rata-rata label aktif di dataset

    # Sampel minoritas = sampel dengan label aktif < rata-rata dataset
    minority_mask = labels_per_sample < mean_labels
    minority_idx = np.where(minority_mask)[0]

    if len(minority_idx) == 0:
        print("  [ML-SMOTE] Tidak ada sampel minoritas yang ditemukan. Data dikembalikan apa adanya.")
        return X, Y

    print(f"  [ML-SMOTE] Sampel minoritas: {len(minority_idx)} / {n_samples}")

    # Fit k-NN pada SEMUA sampel (bukan hanya minoritas) — sesuai paper Charte 2015
    actual_k = min(k + 1, len(X))
    nbrs = NearestNeighbors(n_neighbors=actual_k, algorithm="ball_tree", metric="euclidean")
    nbrs.fit(X)
    _, neighbor_indices = nbrs.kneighbors(X[minority_idx])
    # neighbor_indices[i, 0] adalah diri sendiri, [i, 1..k] adalah k tetangga

    n_synth = int(len(minority_idx) * sampling_ratio)
    chosen = rng.choice(len(minority_idx), size=n_synth, replace=True)

    X_synth_list, Y_synth_list = [], []
    for idx in chosen:
        sample_global_idx = minority_idx[idx]
        # Pilih salah satu dari k tetangga (indeks 1..k, skip 0 = diri sendiri)
        neighbor_local = rng.integers(1, actual_k)
        neighbor_global_idx = neighbor_indices[idx, neighbor_local]

        # Interpolasi fitur
        alpha = rng.uniform(0, 1)
        x_new = X[sample_global_idx] + alpha * (X[neighbor_global_idx] - X[sample_global_idx])

        # Label baru = union dari kedua sampel
        y_new = np.maximum(Y[sample_global_idx], Y[neighbor_global_idx])

        X_synth_list.append(x_new)
        Y_synth_list.append(y_new)

    X_synth = np.array(X_synth_list, dtype=X.dtype)
    Y_synth = np.array(Y_synth_list, dtype=Y.dtype)

    X_out = np.vstack([X, X_synth])
    Y_out = np.vstack([Y, Y_synth])

    print(f"  [ML-SMOTE] Sampel sintetis dibuat: {n_synth}")
    print(f"  [ML-SMOTE] Dataset final: {X_out.shape[0]} sampel (dari {n_samples} asli)")

    return X_out, Y_out
