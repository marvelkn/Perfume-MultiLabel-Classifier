"""Read-only independent checks; write audit evidence only next to this script."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, hashlib, importlib.util
import numpy as np

ROOT = Path(r'C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier')
sys.path.insert(0, str(ROOT))
from research.results import main_run

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def independent_metrics(y, p):
    # AP integrates precision at distinct descending score thresholds.
    y = y.astype(np.int64)
    d = p >= .5
    tp = int(np.sum((y == 1) & d)); fp = int(np.sum((y == 0) & d))
    tn = int(np.sum((y == 0) & ~d)); fn = int(np.sum((y == 1) & ~d))
    order = np.argsort(-p, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(p[order])), len(y)-1]
    positives = np.cumsum(y[order])[ends]
    ap = float(np.sum(np.diff(np.r_[0, positives]) / y.sum() * positives / (ends+1)))
    # AUROC is the probability that a positive outranks a negative, with half credit for ties.
    _, inverse = np.unique(p, return_inverse=True)
    positive_counts = np.bincount(inverse, weights=y)
    negative_counts = np.bincount(inverse, weights=1-y)
    lower_negative_counts = np.cumsum(negative_counts) - negative_counts
    auc = float(np.sum(positive_counts*(lower_negative_counts + .5*negative_counts)) / (y.sum()*(len(y)-y.sum())))
    return dict(accuracy=(tp+tn)/len(y), auroc=auc, auprc=ap,
                specificity=tn/(tn+fp), precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn), macro_f1_0_5=2*tp/(2*tp+fp+fn))

run = main_run()
manifest = read(run/'completion_manifest.json')['files']
bad = [name for name, expected in manifest.items() if sha(run/name) != expected]
assert not bad, bad
old_source = read(ROOT/'KODE_ML/HASIL_VERIFIKASI.json')['source_hashes']
changed_sources = [name for name, expected in old_source.items() if sha(ROOT/name) != expected]
with np.load(ROOT/'data/builds/perfume-five-grouped-v2/test.npz', allow_pickle=False) as z:
    expected_truth = z['truth']
expected_labels = read(ROOT/'data/builds/perfume-five-grouped-v2/dataset_manifest.json')['labels']
completed = read(run/'complete.json')
results=[]
for row in completed['comparison']:
    candidate = row['candidate']
    with np.load(run/candidate/'test_predictions.npz', allow_pickle=False) as z:
        y, p, decisions, labels = z['truth'], z['probabilities'], z['decisions'], z['labels']
    assert np.array_equal(y, expected_truth)
    assert labels.tolist() == expected_labels
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert np.array_equal(decisions, p >= .5)
    values = [independent_metrics(y[:,j], p[:,j]) for j in range(y.shape[1])]
    macro = {key: float(np.mean([m[key] for m in values])) for key in values[0]}
    errors = {key: abs(value-row[key]) for key, value in macro.items()}
    assert max(errors.values()) < 1e-12, (candidate, errors)
    results.append(dict(candidate=candidate, metrics=macro, max_abs_error=max(errors.values()),
                        exact_match_accuracy=float(np.mean(np.all(decisions == y, axis=1))),
                        hamming_loss=float(np.mean(decisions != y))))
selected = max(completed['comparison'], key=lambda r:r['cv_ap_mean'])['candidate']
assert selected == completed['overall_selected_from_repeated_grouped_cv']
notebooks = {}
for file in sorted((ROOT/'KODE_ML').glob('*.ipynb')):
    nb = read(file)
    code = [cell for cell in nb['cells'] if cell['cell_type']=='code']
    errors = [output for cell in code for output in cell.get('outputs',[]) if output['output_type']=='error']
    assert not errors
    notebooks[file.name] = dict(code_cells=len(code), unexecuted=sum(cell.get('execution_count') is None for cell in code), error_outputs=len(errors))
evidence = dict(date='2026-10-27', scope='read-only, no fitting or dataset modification', run=str(run),
                primary_manifest_files_verified=len(manifest), source_changed_since_oct26=changed_sources,
                all_eight_test_results_independently_recomputed=results,
                primary_selected=selected, all_zero_prediction_accuracy=float(np.mean(expected_truth==0)),
                label_test_support_min=int(expected_truth.sum(axis=0).min()),
                label_test_support_max=int(expected_truth.sum(axis=0).max()),
                notebooks_saved_outputs=notebooks,
                notebook_packages_installed={name: importlib.util.find_spec(name) is not None for name in ('nbformat','IPython','ipykernel','nbclient','nbconvert')})
Path(__file__).with_name('independent_checks.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
print(json.dumps(evidence,indent=2))
