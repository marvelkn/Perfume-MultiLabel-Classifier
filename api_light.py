"""
Essenza Fingerprint API — Lightweight Edition
Hanya menghitung Morgan Fingerprint (RDKit) + 5 physical descriptors.
ONNX inference dilakukan on-device di HP (mobile app), bukan di server ini.

Dioptimasi untuk deployment di Render Free Tier (512MB RAM):
- Tidak load ONNX models (hemat ~300MB RAM)
- Tidak load SQLite database parfum
- Hanya rdkit + fastapi = ~200MB RAM
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI(
    title="Essenza Fingerprint API",
    description="Lightweight fingerprint API for Essenza mobile app",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────

class FingerprintRequest(BaseModel):
    smiles: str | None = None
    compound_name: str | None = None


class FingerprintResponse(BaseModel):
    smiles: str
    compound_name: str
    fingerprint: list[float]
    predictions: dict   # selalu {} — inference dilakukan on-device
    iupac_name: str | None = None
    molecular_formula: str | None = None
    molecular_weight: float | None = None
    warning: str | None = None


# ── Helpers ────────────────────────────────────────────────────

def get_rdkit():
    """Lazy import rdkit agar startup cepat dan tidak buang RAM di import."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    return Chem, AllChem


def name_to_smiles(name: str) -> dict:
    """Cari SMILES dari nama senyawa via PubChem REST API."""
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(name)}/property/"
        f"IsomericSMILES,IUPACName,MolecularFormula,MolecularWeight/JSON"
    )
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise HTTPException(404, f"Senyawa '{name}' tidak ditemukan di PubChem.")
    data = resp.json()["PropertyTable"]["Properties"][0]
    return {
        "smiles": data.get("IsomericSMILES", ""),
        "iupac_name": data.get("IUPACName"),
        "molecular_formula": data.get("MolecularFormula"),
        "molecular_weight": data.get("MolecularWeight"),
    }


def compute_fingerprint(smiles: str) -> list[float]:
    """
    Konversi SMILES → Morgan Fingerprint (2048-bit) + 5 RDKit physical descriptors.
    Total output: 2053 float values.

    Untuk campuran (Scent Mixology dengan dot-notation mol1.mol2.mol3):
    - Morgan FP dihitung dari keseluruhan disconnected graph
    - MW dicek per-fragmen (max MW), bukan total MW semua fragmen
    """
    Chem, AllChem = get_rdkit()
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(400, f"SMILES tidak valid: {smiles}")

    # Morgan Fingerprint (2048 bit, radius=2)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = list(fp)

    # Cek MW per komponen (fix untuk Scent Mixology — jangan jumlahkan semua)
    frags = Chem.GetMolFrags(mol, asMols=True)
    max_wt = max(float(Descriptors.MolWt(f)) for f in frags)

    # 5 Physical Descriptors
    logp = float(Descriptors.MolLogP(mol))
    hdon = float(Descriptors.NumHDonors(mol))
    hacc = float(Descriptors.NumHAcceptors(mol))
    tpsa = float(Descriptors.TPSA(mol))

    fp_array.extend([max_wt, logp, hdon, hacc, tpsa])
    return fp_array


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "version": "2.0.0", "message": "Essenza Fingerprint API berjalan!"}


@app.post("/fingerprint", response_model=FingerprintResponse)
def get_fingerprint(req: FingerprintRequest):
    if not req.smiles and not req.compound_name:
        raise HTTPException(400, "Berikan 'smiles' atau 'compound_name'.")

    meta = {"iupac_name": None, "molecular_formula": None, "molecular_weight": None}
    name = req.compound_name or "Unknown"

    if req.compound_name and not req.smiles:
        result = name_to_smiles(req.compound_name)
        smiles = result["smiles"]
        name = req.compound_name
        meta = {k: result[k] for k in ["iupac_name", "molecular_formula", "molecular_weight"]}
    else:
        smiles = req.smiles

    fp = compute_fingerprint(smiles)

    # Filter volatilitas: cek MW fragmen terbesar
    wt = fp[-5]
    if wt > 400:
        raise HTTPException(
            400,
            f"Senyawa terlalu berat ({wt:.2f} g/mol) dan tidak mudah menguap. "
            f"Kemungkinan besar bukan wewangian."
        )

    return FingerprintResponse(
        smiles=smiles,
        compound_name=name,
        fingerprint=fp,
        predictions={},  # Inference dilakukan on-device, bukan di server
        warning=None,
        **meta,
    )


# ── Entry Point ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import socket

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*50}")
    print(f"  Essenza Fingerprint API (Lightweight)")
    print(f"  Akses dari laptop : http://localhost:8000")
    print(f"  Akses dari HP     : http://{local_ip}:8000")
    print(f"  Dokumentasi API   : http://localhost:8000/docs")
    print(f"{'='*50}\n")

    uvicorn.run("api_light:app", host="0.0.0.0", port=8000, reload=True)
