"""SMILES -> canonical SMILES and Morgan fingerprints (RDKit).

Uses the modern, non-deprecated rdFingerprintGenerator API.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from .config import CONFIG

# We track parse failures ourselves; silence RDKit's per-molecule console spam.
RDLogger.DisableLog("rdApp.*")

_FP = CONFIG["fingerprint"]
_N_BITS = int(_FP["n_bits"])
_PRESERVE_STEREO = bool(_FP["preserve_stereo"])
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=int(_FP["radius"]), fpSize=_N_BITS)


def canonical_smiles(smiles: str) -> Optional[str]:
    """Return RDKit-canonical SMILES, or None if the string cannot be parsed.

    Stereochemistry is preserved by default (isomericSmiles=True): carvone
    enantiomers smell different (spearmint vs caraway), so we must NOT collapse them.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=_PRESERVE_STEREO)


def morgan_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """Return a (n_bits,) uint8 Morgan fingerprint, or None if SMILES is invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _GEN.GetFingerprintAsNumPy(mol).astype(np.uint8)


def fingerprint_matrix(smiles_list: Iterable[str]):
    """Featurize many SMILES.

    Returns (X, mask): X is (n_valid, n_bits) uint8; mask is a bool array over the
    input marking which SMILES produced a fingerprint (others were unparseable).
    """
    rows, mask = [], []
    for smi in smiles_list:
        fp = morgan_fingerprint(smi)
        mask.append(fp is not None)
        if fp is not None:
            rows.append(fp)
    X = np.vstack(rows) if rows else np.empty((0, _N_BITS), dtype=np.uint8)
    return X, np.asarray(mask, dtype=bool)
