"""Development-only imbalance candidates.

Synthetic samples are feature-space examples, not validated molecular structures.
Charte et al. (2015), doi:10.1016/j.knosys.2015.07.019.
"""
import numpy as np

def resample(X, Y, strategy="none", *, seed=42, ratio=.5, k=5, n_bits=2048):
    if strategy in ("none", "class_weight"):
        return X, Y
    if strategy not in ("random_oversample", "mlsmote"):
        raise ValueError(f"Unknown imbalance strategy: {strategy}")
    if not 0 <= ratio <= 1 or k < 1:
        raise ValueError("Invalid resampling budget")
    rng = np.random.default_rng(seed)
    counts = Y.sum(axis=0)
    present = counts > 0
    if not present.any():
        return X, Y
    imbalance = np.zeros(Y.shape[1], dtype=float)
    imbalance[present] = counts.max() / counts[present]
    minority = np.flatnonzero(present & (imbalance > imbalance[present].mean()))
    if not len(minority) or ratio == 0:
        return X, Y
    extra_x, extra_y = [], []
    scale = np.std(X[:, n_bits:], axis=0)
    scale[scale == 0] = 1
    for _ in range(int(len(X) * ratio)):
        label = rng.choice(minority)
        bag = np.flatnonzero(Y[:, label])
        seed_index = rng.choice(bag)
        if strategy == "random_oversample" or len(bag) < 2:
            extra_x.append(X[seed_index].copy())
            extra_y.append(Y[seed_index].copy())
            continue
        # Mixed distance: binary Hamming + standardized descriptor mean-square.
        # This is an explicit adaptation (not original VDM) and is evaluated separately.
        others = bag[bag != seed_index]
        distance = np.mean(X[others, :n_bits] != X[seed_index, :n_bits], axis=1)
        if X.shape[1] > n_bits:
            distance += np.mean(((X[others, n_bits:] - X[seed_index, n_bits:]) / scale) ** 2, axis=1)
        neighbors = others[np.argsort(distance, kind="stable")[:k]]
        neighborhood = np.r_[seed_index, neighbors]
        reference = rng.choice(neighbors)
        new_x = X[seed_index].copy()
        new_x[:n_bits] = (X[neighborhood, :n_bits].mean(axis=0) >= .5)
        new_x[n_bits:] += rng.random() * (X[reference, n_bits:] - new_x[n_bits:])
        new_y = (Y[neighborhood].mean(axis=0) >= .5).astype(Y.dtype)
        extra_x.append(new_x)
        extra_y.append(new_y)
    if not extra_x:
        return X, Y
    return np.vstack([X, np.asarray(extra_x, dtype=X.dtype)]), np.vstack([Y, np.asarray(extra_y, dtype=Y.dtype)])

def ml_smote(X, Y, k=5, sampling_ratio=.5, seed=42):
    return resample(X, Y, "mlsmote", seed=seed, ratio=sampling_ratio, k=k)
