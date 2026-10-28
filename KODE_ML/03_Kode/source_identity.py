"""Pencocokan identitas molekul dan perapian istilah; tanpa konfigurasi lama."""
from collections import defaultdict
import re
import pandas as pd
from rdkit import Chem
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")

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

def normalize(token: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = str(token).strip().lower()
    t = _NON_ALNUM.sub(" ", t)
    return _WS.sub(" ", t).strip()

def _columns(frame):
    if frame.index.name and frame.index.name not in frame:
        frame = frame.reset_index()
    return frame.copy()


def _key(series):
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True).replace("", pd.NA)


def _unique(frame, key, value):
    frame = frame[[key, value]].dropna().copy()
    frame[key] = _key(frame[key])
    frame = frame.dropna().drop_duplicates()
    if frame[key].duplicated().any():
        raise ValueError(f"Conflicting {key} -> {value} mapping")
    return frame


def _attach_smiles(beh, sti, mol):
    beh, sti, mol = map(_columns, (beh, sti, mol))
    cid = "CID" if "CID" in sti else "new_CID"
    if cid not in sti or "Stimulus" not in beh or "Stimulus" not in sti:
        raise ValueError("Explicit Stimulus -> CID bridge is required")
    smiles_column = next((c for c in ("IsomericSMILES", "SMILES", "CanonicalSMILES") if c in mol), None)
    if smiles_column is None or "CID" not in mol:
        raise ValueError("Explicit CID -> SMILES mapping is required")
    beh["Stimulus"], sti["Stimulus"] = _key(beh["Stimulus"]), _key(sti["Stimulus"])
    sti[cid], mol["CID"] = _key(sti[cid]), _key(mol["CID"])
    bridge = _unique(sti, "Stimulus", cid).rename(columns={cid: "__cid__"})
    lookup = _unique(mol, "CID", smiles_column).rename(columns={"CID": "__cid__", smiles_column: "__smiles__"})
    # Null identifiers are excluded from mappings; pandas must not match null keys.
    frame = beh.merge(bridge, on="Stimulus", how="left", validate="many_to_one")
    return frame.merge(lookup, on="__cid__", how="left", validate="many_to_one")


def _truthy(value):
    return str(value).strip().lower() in {"1", "1.0", "true"}


def _attach_sigma(beh, sti, cas_map):
    """Require every listed CAS alias to resolve to one validated identity."""
    beh, sti = map(_columns, (beh, sti))
    beh["Stimulus"], sti["Stimulus"] = _key(beh["Stimulus"]), _key(sti["Stimulus"])
    mapping, excluded = [], []
    for stimulus, rows in sti.dropna(subset=["Stimulus"]).groupby("Stimulus"):
        aliases = rows["CAS"].dropna().astype(str).str.strip().unique().tolist()
        identities = {cas_map.get(cas) for cas in aliases}
        if aliases and None not in identities and len(identities) == 1:
            mapping.append({"Stimulus": stimulus, "__smiles__": next(iter(identities))})
        else:
            excluded.append({"stimulus": stimulus, "cas": aliases,
                             "reason": "unresolved_or_conflicting_cas_aliases"})
    bridge = pd.DataFrame(mapping, columns=["Stimulus", "__smiles__"])
    return beh.merge(bridge, on="Stimulus", how="left", validate="many_to_one"), excluded


def _cas_bridge(dfs, frames):
    identities = defaultdict(set)
    for source, frame in frames.items():
        sti = _columns(dfs[f"{source}_stimuli"])
        cas_col = next((c for c in sti if c.lower() == "cas"), None)
        if source == "goodscents":
            pairs = frame[["Stimulus", "__smiles__"]].rename(columns={"Stimulus": "__cas__"})
        elif cas_col:
            sti["Stimulus"] = _key(sti["Stimulus"])
            mapping = _unique(sti, "Stimulus", cas_col).rename(columns={cas_col: "__cas__"})
            pairs = frame[["Stimulus", "__smiles__"]].merge(mapping, on="Stimulus", validate="many_to_one")
        else:
            continue
        for cas, smiles in pairs[["__cas__", "__smiles__"]].itertuples(index=False, name=None):
            canonical = canonical_smiles(smiles)
            if isinstance(cas, str) and canonical:
                identities[cas.strip()].add(canonical)
    return {k: next(iter(v)) for k, v in identities.items() if len(v) == 1}, sorted(k for k,v in identities.items() if len(v)>1)
