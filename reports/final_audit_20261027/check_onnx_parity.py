"""Compare existing ONNX assets with frozen native models; do not export or fit."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json, warnings
import numpy as np
import onnxruntime as ort
import joblib
ROOT=Path(r'C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier')
FRONT=ROOT.parent/'Essenza_Frontend'
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(FRONT))
from research.results import main_run
from research.features import calculate, calculate_morgan
from feature_api.app import calculate as api_calculate
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
run=main_run(); bundle=read(run/'inference_bundle.json')
manifest=read(FRONT/'src/assets/metadata/v7_model_manifest.json')
indices=read(ROOT/'data/builds/perfume-five-grouped-v2/splits.json')['train'][:32]
with np.load(ROOT/'data/builds/perfume-m2048-d8-v7/features.npz',allow_pickle=False) as z:
    x=np.concatenate([z['morgan'][indices],z['descriptors'][indices]],axis=1).astype(np.float32)
max_error=0.; decision_differences=0
settings=ort.SessionOptions(); settings.intra_op_num_threads=1; settings.inter_op_num_threads=1
for entry in manifest['models']:
    native=joblib.load(run/'lgbm_D'/bundle['models'][entry['label']]['file'])
    session=ort.InferenceSession(str(FRONT/'android/app/src/main/assets/models/v7'/entry['filename']),sess_options=settings,providers=['CPUExecutionProvider'])
    converted=session.run([entry['probability_output']],{entry['input_name']:x})[0][:,entry['positive_index']]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='X does not have valid feature names')
        original=native.predict_proba(x)[:,1]
    np.testing.assert_allclose(converted,original,rtol=1e-5,atol=1e-6)
    max_error=max(max_error,float(np.max(np.abs(original-converted))))
    decision_differences+=int(np.sum((converted>=.5)!=(original>=.5)))
    del session,native
assert decision_differences==0
smiles=['COc1cc(C=O)ccc1O','CCO','CC(=CCCC(C)(C=C)O)C']
expected=np.concatenate([calculate_morgan(smiles),calculate(smiles)],axis=1).astype(np.float32)
for i,s in enumerate(smiles): np.testing.assert_array_equal(expected[i],np.array(api_calculate(s)['fingerprint'],dtype=np.float32))
result=dict(date='2026-10-27',models=109,training_vectors=32,score_pairs=109*32,
            max_absolute_score_difference=max_error,threshold_decision_differences=decision_differences,
            rtol=1e-5,atol=1e-6,feature_api_vectors_exact=3,
            scope='desktop ONNX CPU execution, read-only; no Android device or HTTP server test')
Path(__file__).with_name('onnx_parity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
