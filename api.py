"""
AromaML Fingerprint & Prediction API (B2B Duplication Tool)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import requests
import json
import os
import sqlite3
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import onnxruntime as ort
from pathlib import Path

app = FastAPI(title="AromaML B2B Duplication API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    fingerprint: list[float]
    predictions: dict[str, dict[str, float | int]]
    iupac_name: str | None = None
    molecular_formula: str | None = None
    molecular_weight: float | None = None

class RecommendRequest(BaseModel):
    label_probabilities: dict[str, float] 
    top_k: int = 10

class PerfumeResult(BaseModel):
    pid: str
    brand: str
    name: str
    main_photo: str | None
    top_accords: str
    similarity_score: float
    rating: float | None

PERFUME_DB_PATH = os.path.join("dataset", "perfume_db.sqlite")
CROSSWALK_PATH = os.path.join("mobile_assets", "leffingwell_to_fragrantica.json")
ONNX_DIR = os.path.join("models", "xgb_onnx")
META_PATH = os.path.join("mobile_assets", "xgb_meta.json")

df_perfumes = pd.DataFrame()
crosswalk = {}

class OnnxPredictor:
    """Loads all XGBoost ONNX sessions at startup; returns probs + binary labels."""
    def __init__(self, onnx_dir, meta_path):
        self.sessions = {}
        self.meta = []
        if not os.path.exists(meta_path):
            print(f"Warning: {meta_path} not found. Prediction disabled.")
            return
            
        self.meta = json.loads(Path(meta_path).read_text())
        
        # Optimize memory usage for cloud deployment (prevents OOM on Railway)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        
        for entry in self.meta:
            label = entry["label"]
            fname = Path(onnx_dir) / f"xgb_{label.replace(' ', '_')}.onnx"
            if fname.exists():
                self.sessions[label] = ort.InferenceSession(str(fname), sess_options=opts)

    def predict(self, fingerprint: list[float]) -> dict:
        if not self.sessions:
            return {}
            
        x = np.array(fingerprint, dtype=np.float32).reshape(1, -1)
        results = {}
        for entry in self.meta:
            label, threshold = entry["label"], entry["threshold"]
            sess = self.sessions.get(label)
            if sess:
                try:
                    res = sess.run(None, {"float_input": x})
                    # Attempt to extract probability (class 1) safely
                    if isinstance(res[1][0], dict):
                        prob = float(res[1][0].get(1, 0.0))
                    else:
                        prob = float(res[1][0][1])
                except Exception:
                    prob = 0.0
                results[label] = {"probability": prob, "predicted": int(prob >= threshold)}
        return results

predictor = None

@app.on_event("startup")
def startup_event():
    global predictor, df_perfumes, crosswalk
    print("Loading ONNX models...")
    predictor = OnnxPredictor(ONNX_DIR, META_PATH)
    
    print("Loading Fragrantica database...")
    if os.path.exists(PERFUME_DB_PATH):
        conn = sqlite3.connect(PERFUME_DB_PATH)
        df_perfumes = pd.read_sql("SELECT * FROM perfumes", conn)
        conn.close()
        print(f"Loaded {len(df_perfumes)} perfumes.")
    else:
        print(f"Warning: {PERFUME_DB_PATH} not found. Recommendation disabled.")
        
    if os.path.exists(CROSSWALK_PATH):
        with open(CROSSWALK_PATH, "r") as f:
            crosswalk = json.load(f)
        print("Loaded crosswalk mapping.")

def name_to_smiles(name: str) -> dict:
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

def compute_fingerprint(smiles: str) -> list[float]:
    Chem, AllChem = get_rdkit()
    from rdkit.Chem import Descriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(400, f"SMILES tidak valid: {smiles}")
        
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = list(fp)
    
    wt = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    hdon = float(Descriptors.NumHDonors(mol))
    hacc = float(Descriptors.NumHAcceptors(mol))
    tpsa = float(Descriptors.TPSA(mol))
    
    fp_array.extend([wt, logp, hdon, hacc, tpsa])
    return fp_array

@app.get("/")
def root():
    return {"status": "ok", "message": "AromaML B2B API berjalan!"}

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
    preds = predictor.predict(fp) if predictor else {}

    return FingerprintResponse(
        smiles=smiles,
        compound_name=name,
        fingerprint=fp,
        predictions=preds,
        **meta,
    )

@app.post("/recommend", response_model=list[PerfumeResult])
def recommend_perfumes(req: RecommendRequest):
    if df_perfumes.empty:
        raise HTTPException(500, "Database parfum belum siap/belum di-generate (Jalankan etl_pipeline.py).")
    
    if not req.label_probabilities:
        return []

    # Map Leffingwell probabilities to Fragrantica accords using crosswalk
    desired_accords = {}
    for label, prob in req.label_probabilities.items():
        if prob == 0:
            continue
        mapped_accord = crosswalk.get(label.lower())
        if mapped_accord:
            # If multiple labels map to the same accord, take the max probability
            desired_accords[mapped_accord] = max(desired_accords.get(mapped_accord, 0.0), prob)
            
    if not desired_accords:
        return []

    all_accords_set = set()
    parsed_accords_list = []
    
    for _, row in df_perfumes.iterrows():
        acc_dict = json.loads(row['accords_parsed'])
        parsed_accords_list.append(acc_dict)
        all_accords_set.update(acc_dict.keys())
        
    all_accords = sorted(list(all_accords_set))
    
    db_vectors = np.zeros((len(df_perfumes), len(all_accords)))
    for i, acc_dict in enumerate(parsed_accords_list):
        for j, acc_name in enumerate(all_accords):
            db_vectors[i, j] = acc_dict.get(acc_name, 0.0)
            
    query_vector = np.zeros((1, len(all_accords)))
    for j, acc_name in enumerate(all_accords):
        match = next((v for k, v in desired_accords.items() if k.lower() == acc_name.lower()), 0.0)
        query_vector[0, j] = match
        
    similarities = cosine_similarity(query_vector, db_vectors)[0]
    
    top_indices = np.argsort(similarities)[::-1][:req.top_k]
    
    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score > 0: 
            row = df_perfumes.iloc[idx]
            results.append(PerfumeResult(
                pid=str(row['pid']),
                brand=str(row['brand']),
                name=str(row['name']),
                main_photo=str(row['main_photo']) if pd.notna(row['main_photo']) else None,
                top_accords=str(row['top_accords']),
                similarity_score=float(score),
                rating=float(str(row['rating']).split(';')[0]) if pd.notna(row['rating']) and str(row['rating']) != 'nan' else None
            ))
            
    return results

if __name__ == "__main__":
    import uvicorn
    import socket

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*50}")
    print(f"  AromaML B2B Duplication API")
    print(f"  Akses dari laptop : http://localhost:8000")
    print(f"  Akses dari HP     : http://{local_ip}:8000")
    print(f"  Dokumentasi API   : http://localhost:8000/docs")
    print(f"{'='*50}\n")

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
