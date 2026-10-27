"""Check existing evidence without changing project, dataset, or models."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, hashlib, importlib.util
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from rdkit import Chem
ROOT = Path(r'C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier')
FRONT = ROOT.parent/'Essenza_Frontend'
sys.path.insert(0,str(ROOT))
from research.results import main_run
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
data = ROOT/'data/builds/perfume-five-grouped-v2'
records = pd.read_csv(data/'records.csv')
split = read(data/'splits.json'); labels = read(data/'dataset_manifest.json')['labels']
with np.load(ROOT/'data/builds/perfume-m2048-d8-v7/features.npz',allow_pickle=False) as z:
    morgan, desc = z['morgan'],z['descriptors']
truth = np.zeros((len(records),len(labels)),dtype=np.uint8)
for role in ('train','test'):
    with np.load(data/f'{role}.npz',allow_pickle=False) as z: truth[z['row_index']] = z['truth']
groups = defaultdict(list)
for i in range(len(records)): groups[morgan[i].tobytes()+desc[i].tobytes()].append(i)
duplicate_groups = [ids for ids in groups.values() if len(ids)>1]
conflict_groups = [ids for ids in duplicate_groups if len({truth[i].tobytes() for i in ids})>1]
test = set(split['test']); train = set(split['train'])
cross = [ids for ids in duplicate_groups if set(ids)&test and set(ids)&train]
assert not cross
def nonstereo(smiles):
    mol=Chem.MolFromSmiles(smiles); Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)
different_nonstereo = [ids for ids in conflict_groups if len({nonstereo(records.iloc[i].smiles) for i in ids})>1]
summary = dict(feature_identical_groups=len(duplicate_groups),molecules_in_identical_groups=sum(map(len,duplicate_groups)),
               groups_with_conflicting_selected_targets=len(conflict_groups),molecules_in_conflicting_groups=sum(map(len,conflict_groups)),
               feature_identical_groups_crossing_outer_split=len(cross),
               conflicting_groups_with_different_nonstereo_structures=len(different_nonstereo),
               conflicting_groups_same_nonstereo_structure=len(conflict_groups)-len(different_nonstereo))
for role in ('train','test'):
    ids = set(split[role]); selected = [g for g in conflict_groups if set(g)&ids]
    summary[role] = dict(conflicting_groups=len(selected),molecules=sum(map(len,selected)),
                        minimum_exact_profile_mismatches_due_to_identical_features=sum(len(g)-max(Counter(truth[i].tobytes() for i in g).values()) for g in selected),
                        all_zero_target_rows=int(np.sum(truth[split[role]].sum(axis=1)==0)))
summary['examples'] = [[dict(row=int(i),smiles=records.iloc[i].smiles,labels=[labels[j] for j in np.flatnonzero(truth[i])]) for i in ids] for ids in conflict_groups[:3]]
bundle = read(main_run()/'inference_bundle.json')
manifest = read(FRONT/'src/assets/metadata/v7_model_manifest.json')
assert manifest['source_bundle_sha256']==sha(main_run()/'inference_bundle.json')
assert manifest['feature_spec']==bundle['feature_spec']
assert manifest['labels']==bundle['labels']==[m['label'] for m in manifest['models']]
assert manifest['primary_threshold']==bundle['primary_threshold']==.5
assert manifest['model_id'].endswith('/'+bundle['selected_candidate'])
assert len(manifest['models'])==109
for m in manifest['models']:
    assert sha(FRONT/'android/app/src/main/assets/models/v7'/m['filename'])==m['sha256']
result = dict(date='2026-10-27',data=summary,
              frontend=dict(source_bundle_matches=True,feature_spec_and_label_order_matches=True,
                            onnx_hashes_verified=109,threshold=.5,selected=bundle['selected_candidate'],
                            evidence_document=str(FRONT/'docs/VERIFICATION.md'),
                            android_device_not_retested_today=True,
                            onnxruntime_available=importlib.util.find_spec('onnxruntime') is not None))
Path(__file__).with_name('data_and_mobile_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
