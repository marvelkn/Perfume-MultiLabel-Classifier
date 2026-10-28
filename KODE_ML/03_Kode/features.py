"""SMILES -> Morgan 2048 dan delapan angka deskriptor, dengan urutan tetap."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from .runtime import ROOT
N_BITS = 2048
RADIUS = 2
DESCRIPTORS8 = ("MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds", "HeavyAtomCount", "RingCount")
EXTRA = ROOT / "data/builds/perfume-m2048-d8-v7"
CANDIDATES = tuple(f"{algorithm}_{condition}" for algorithm in ("xgb", "lgbm") for condition in "ABCD")

def molecule(smiles, index):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"Invalid or multicomponent structure at row {index}; no rows may be dropped.")
    return mol


def calculate(smiles):
    values = np.empty((len(smiles), 8), dtype=np.float32)
    for i, value in enumerate(smiles):
        mol = molecule(value, i)
        values[i] = [getattr(Descriptors, name)(mol) for name in DESCRIPTORS8]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite descriptors; no silent imputation.")
    return values


def calculate_morgan(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, fpSize=N_BITS, includeChirality=False)
    return np.asarray([generator.GetFingerprintAsNumPy(molecule(smi, i))
                       for i, smi in enumerate(smiles)], dtype=np.uint8)

def descriptor_count(key):
    if key not in CANDIDATES:
        raise ValueError(f"Unknown candidate: {key}")
    return 8 if key[-1] in "BD" else 0


def features(key, morgan, descriptors):
    if (morgan.ndim != 2 or morgan.shape[1] != N_BITS or descriptors.shape != (len(morgan), 8)
            or not np.isfinite(descriptors).all()):
        raise ValueError("Expected Morgan 2048 and eight ordered descriptors per molecular row.")
    return np.column_stack([morgan, descriptors[:, :descriptor_count(key)]]).astype(np.float32)

def verify(dataset=None, extra=EXTRA):
    from .dataset import DATASET, verify as verify_dataset
    _, _, meta, morgan, descriptors = verify_dataset(DATASET if dataset is None else dataset, extra)
    return meta, morgan, descriptors
