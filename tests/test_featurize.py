"""Unit tests for the featurization layer (no network required)."""
import numpy as np

from src.config import CONFIG
from src.featurize import canonical_smiles, fingerprint_matrix, morgan_fingerprint

N_BITS = CONFIG["fingerprint"]["n_bits"]


def test_canonical_smiles_roundtrip():
    # Two equivalent inputs for benzaldehyde must canonicalize identically.
    assert canonical_smiles("O=Cc1ccccc1") == canonical_smiles("c1ccccc1C=O")


def test_invalid_smiles_returns_none():
    assert canonical_smiles("not_a_smiles") is None
    assert morgan_fingerprint("not_a_smiles") is None
    assert canonical_smiles("") is None


def test_fingerprint_shape_and_dtype():
    fp = morgan_fingerprint("CCO")  # ethanol
    assert fp is not None
    assert fp.shape == (N_BITS,)
    assert fp.dtype == np.uint8
    assert set(np.unique(fp)).issubset({0, 1})


def test_carvone_enantiomers_stay_distinct():
    # R- and S-carvone smell different (spearmint vs caraway) -> must NOT collapse.
    r = canonical_smiles("CC1=CC[C@@H](CC1=O)C(=C)C")
    s = canonical_smiles("CC1=CC[C@H](CC1=O)C(=C)C")
    assert r is not None and s is not None
    assert r != s


def test_fingerprint_matrix_filters_invalid():
    X, mask = fingerprint_matrix(["CCO", "garbage", "c1ccccc1"])
    assert mask.tolist() == [True, False, True]
    assert X.shape == (2, N_BITS)
