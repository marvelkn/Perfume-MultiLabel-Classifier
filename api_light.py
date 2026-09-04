"""Versioned CPU fingerprint service; shares featurization with training."""
import json
import os
from pathlib import Path
from urllib.parse import quote
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.featurize import FeatureSpec, MoleculeError, features_and_metadata

SPEC_PATH = Path(os.environ.get("ESSENZA_FEATURE_SPEC", Path(__file__).with_name("feature_spec.json")))
SPEC = FeatureSpec.from_dict(json.loads(SPEC_PATH.read_text(encoding="utf-8")))
app = FastAPI(title="Essenza Fingerprint API",version="3.0.0")

class FingerprintRequest(BaseModel):
    smiles: str | None = Field(default=None,max_length=10000)
    compound_name: str | None = Field(default=None,max_length=200)
    feature_schema_id: str | None = None

def name_to_smiles(name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name,safe='')}/property/SMILES,IUPACName/JSON"
    try:
        response = requests.get(url,timeout=(5,10))
        if response.status_code == 404:
            raise MoleculeError("COMPOUND_NOT_FOUND","Compound name was not found.")
        response.raise_for_status()
        records = response.json()["PropertyTable"]["Properties"]
        if len(records) != 1:
            raise MoleculeError("AMBIGUOUS_COMPOUND","Use a verified SMILES for this ambiguous name.")
        record = records[0]
        smiles = record.get("SMILES") or record.get("IsomericSMILES")
        if not smiles:
            raise ValueError("SMILES missing")
        return {"smiles":smiles,"iupac_name":record.get("IUPACName")}
    except MoleculeError:
        raise
    except (requests.RequestException,KeyError,ValueError) as exc:
        raise MoleculeError("LOOKUP_UNAVAILABLE","Compound lookup is unavailable. You can enter SMILES directly.") from exc

def compute_fingerprint(smiles):
    return features_and_metadata(smiles,SPEC)[0].tolist()

def fingerprint_result(smiles="",compound_name="",feature_schema_id=None):
    smiles,compound_name = (smiles or "").strip(), (compound_name or "").strip()
    if feature_schema_id and feature_schema_id != SPEC.schema_id:
        raise MoleculeError("FEATURE_SCHEMA_MISMATCH","The app model and feature service use different versions.")
    if len(smiles) > 10000 or len(compound_name) > 200:
        raise MoleculeError("INVALID_INPUT","Input exceeds the supported length.")
    iupac = None
    if not smiles and compound_name:
        found = name_to_smiles(compound_name)
        smiles,iupac = found["smiles"],found["iupac_name"]
    vector,meta = features_and_metadata(smiles,SPEC)
    return {"status":"ok",**meta,"compound_name":compound_name or None,"fingerprint":vector.tolist(),
            "iupac_name":iupac,"warning":None}

@app.get("/")
def root():
    return {"status":"ok","version":"3.0.0","feature_schema_id":SPEC.schema_id}

@app.post("/fingerprint")
def fingerprint(req: FingerprintRequest):
    try:
        return fingerprint_result(req.smiles,req.compound_name,req.feature_schema_id)
    except MoleculeError as exc:
        status = 503 if exc.code == "LOOKUP_UNAVAILABLE" else 422
        raise HTTPException(status,detail={"code":exc.code,"message":str(exc)}) from exc
