"""One feature implementation for training, REST, and Gradio.

Chirality and descriptor choices are explicit and included in the schema ID.
Multi-fragment SMILES are outside the validated single-molecule task.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from .artifacts import json_hash

DESCRIPTORS = ("MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA")

@dataclass(frozen=True)
class FeatureSpec:
    radius: int = 2
    n_bits: int = 2048
    include_chirality: bool = False
    use_descriptors: bool = True
    fragment_policy: str = "single"
    dtype: str = "float32"
    version: int = 1
    rdkit_version: str = "2026.03.4"

    def __post_init__(self):
        if self.rdkit_version != rdBase.rdkitVersion:
            raise ValueError(f"Feature schema requires RDKit {self.rdkit_version}; installed {rdBase.rdkitVersion}")
        if self.radius < 1 or self.n_bits < 8 or self.fragment_policy != "single" or self.dtype != "float32":
            raise ValueError("Unsupported feature specification")

    @property
    def n_features(self):
        return self.n_bits + (len(DESCRIPTORS) if self.use_descriptors else 0)

    def to_dict(self):
        return asdict(self)

    @property
    def schema_id(self):
        return json_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        return cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})

class MoleculeError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code

def parse_molecule(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        raise MoleculeError("INVALID_SMILES", "Enter a non-empty SMILES string.")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        raise MoleculeError("INVALID_SMILES", "The SMILES string cannot be parsed.")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise MoleculeError("UNSUPPORTED_MIXTURE", "Prediction supports one connected molecule. Mixture odor has not been validated.")
    return mol

def canonical_smiles(smiles):
    try:
        return Chem.MolToSmiles(parse_molecule(smiles), isomericSmiles=True)
    except MoleculeError:
        return None

@lru_cache(maxsize=16)
def _generator(spec):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=spec.radius, fpSize=spec.n_bits, includeChirality=spec.include_chirality)

def features_and_metadata(smiles, spec=None):
    spec = spec or FeatureSpec()
    mol = parse_molecule(smiles)
    vector = _generator(spec).GetFingerprintAsNumPy(mol).astype(np.float32)
    descriptor_values = [getattr(Descriptors, name)(mol) for name in DESCRIPTORS]
    if spec.use_descriptors:
        vector = np.concatenate([vector, np.asarray(descriptor_values, dtype=np.float32)])
    if vector.shape != (spec.n_features,) or not np.isfinite(vector).all():
        raise MoleculeError("INVALID_FEATURES", "Molecular features are not finite.")
    return vector, {"smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
                    "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
                    "molecular_weight": float(descriptor_values[0]),
                    "feature_schema_id": spec.schema_id,
                    "feature_spec": spec.to_dict()}

def extract_features(smiles, spec=None):
    try:
        return features_and_metadata(smiles, spec)[0]
    except MoleculeError:
        return None

def fingerprint_matrix(smiles_list, spec=None):
    spec = spec or FeatureSpec()
    rows, mask = [], []
    for smiles in smiles_list:
        vector = extract_features(smiles, spec)
        mask.append(vector is not None)
        if vector is not None:
            rows.append(vector)
    return (np.vstack(rows) if rows else np.empty((0, spec.n_features), dtype=np.float32)), np.asarray(mask, dtype=bool)
