"""
AromaML Fingerprint API
Menghitung Morgan Fingerprint (2048-bit, radius 2) dari SMILES atau nama senyawa.
Jalankan dengan: python api.py
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import requests
import json

app = FastAPI(title="AromaML Fingerprint API", version="1.0.0")

# Izinkan akses dari semua origin (untuk development / mobile app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy import RDKit
def get_rdkit():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    return Chem, AllChem

class FingerprintRequest(BaseModel):
    smiles: str | None = None
    compound_name: str | None = None

class FingerprintResponse(BaseModel):
    smiles: str
    compound_name: str
    fingerprint: list[int]   # 2048-bit vector
    iupac_name: str | None = None
    molecular_formula: str | None = None
    molecular_weight: float | None = None


def name_to_smiles(name: str) -> dict:
    """Cari SMILES dari nama senyawa via PubChem."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/property/IsomericSMILES,IUPACName,MolecularFormula,MolecularWeight/JSON"
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


def compute_fingerprint(smiles: str) -> list[int]:
    """Hitung Morgan Fingerprint 2048-bit radius 2 dari SMILES."""
    Chem, AllChem = get_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(400, f"SMILES tidak valid: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return list(fp)


@app.get("/")
def root():
    return {"status": "ok", "message": "AromaML Fingerprint API berjalan!"}


@app.post("/fingerprint", response_model=FingerprintResponse)
def get_fingerprint(req: FingerprintRequest):
    """
    Hitung Morgan Fingerprint dari nama senyawa atau SMILES.
    Kirim salah satu dari `compound_name` atau `smiles`.
    """
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

    return FingerprintResponse(
        smiles=smiles,
        compound_name=name,
        fingerprint=fp,
        **meta,
    )


@app.get("/demo-molecules")
def demo_molecules():
    """Daftar molekul parfum populer untuk demo."""
    return {
        "molecules": [
            {"name": "linalool",      "description": "Lavender, floral, woody"},
            {"name": "geraniol",      "description": "Rose, floral, citrus"},
            {"name": "limonene",      "description": "Citrus, orange, lemon"},
            {"name": "vanillin",      "description": "Vanilla, sweet, creamy"},
            {"name": "menthol",       "description": "Mint, fresh, cooling"},
            {"name": "eugenol",       "description": "Clove, spicy, woody"},
            {"name": "benzyl alcohol","description": "Floral, sweet, rose"},
            {"name": "citronellol",   "description": "Rose, citrus, floral"},
            {"name": "camphor",       "description": "Camphor, medicinal, woody"},
            {"name": "coumarin",      "description": "Sweet, hay, tonka bean"},
            {"name": "isoeugenol",    "description": "Spicy, floral, clove"},
            {"name": "carvone",       "description": "Spearmint, herbal, fresh"},
            {"name": "cinnamaldehyde","description": "Cinnamon, spicy, sweet"},
            {"name": "hexanal",       "description": "Grassy, fresh, green"},
            {"name": "benzaldehyde",  "description": "Almond, cherry, sweet"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    import socket

    # Tampilkan IP lokal agar mudah diakses dari HP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*50}")
    print(f"  AromaML Fingerprint API")
    print(f"  Akses dari laptop : http://localhost:8000")
    print(f"  Akses dari HP     : http://{local_ip}:8000")
    print(f"  Dokumentasi API   : http://localhost:8000/docs")
    print(f"{'='*50}\n")

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
