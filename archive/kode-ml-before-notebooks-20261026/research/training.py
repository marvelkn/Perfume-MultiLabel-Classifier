"""Satu classifier biner per label; bobot hanya dari bagian latih."""
import joblib
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import xgboost
from .runtime import compute_profile, Limits, read, write, digest, WEIGHTING_POLICY

def make_model(algorithm, parameters):
    params = dict(parameters)
    profile = compute_profile()
    threads = profile["threads_per_model"]
    if algorithm == "xgb":
        return XGBClassifier(n_jobs=threads, random_state=42, eval_metric="logloss",
                             tree_method=profile["xgboost_tree_method"], **params)
    if algorithm == "lgbm":
        return LGBMClassifier(n_jobs=threads, random_state=42, verbosity=-1, **params)
    raise ValueError("Unknown algorithm.")


def balanced_sample_weights(y):
    """Use the same fold-local weighting rule for XGBoost and LightGBM."""
    values = np.asarray(y, dtype=np.int8)
    if values.ndim != 1 or len(values) == 0 or not np.isin(values, [0, 1]).all():
        raise ValueError("Expected a non-empty binary target vector.")
    counts = np.bincount(values, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Balanced weighting requires both classes.")
    per_class = len(values) / (2.0 * counts)
    return per_class[values]


def fit(algorithm, parameters, x, y, limits):
    limits.check(force=True)
    if np.unique(y).size != 2:
        raise ValueError("Training requires both classes.")
    model = make_model(algorithm, parameters)
    weights = balanced_sample_weights(y)
    if algorithm == "xgb":
        class Callback(xgboost.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                limits.check()
                return False
        model.set_params(callbacks=[Callback()])
        model.fit(x, y, sample_weight=weights, verbose=False)
        model.set_params(callbacks=None)
    else:
        def callback(env):
            limits.check()
        callback.order = 5
        model.fit(x, y, sample_weight=weights, callbacks=[callback])
    limits.check(force=True)
    return model

def fit_final_models(folder, algorithm, params, x, y, labels):
    saved = read(folder / "models.json") if (folder / "models.json").exists() else {}
    for j, name in enumerate(labels):
        if name in saved:
            if digest(folder / saved[name]["file"]) != saved[name]["sha256"]:
                raise ValueError("Final-model checkpoint changed.")
            continue
        model = fit(algorithm, params, x, y[:, j], Limits())
        filename = f"label_{j:03d}.joblib"
        joblib.dump(model, folder / filename)
        saved[name] = {"file": filename, "sha256": digest(folder / filename)}
        write(folder / "models.json", saved)
