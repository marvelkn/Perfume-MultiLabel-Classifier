"""Measure a complete bundle using real features; no hard-coded performance claims."""
import argparse
import json
import time
import psutil
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from export_for_mobile import session
from .featurize import FeatureSpec
from .artifacts import digest, write_json, environment

def benchmark(bundle_path, features, iterations=20):
    root=Path(bundle_path)
    manifest=json.loads((root/"manifest.json").read_text())
    spec=FeatureSpec.from_dict(manifest["feature_spec"])
    X=np.asarray(features,dtype=np.float32)
    if iterations < 1 or X.ndim != 2 or X.shape[1] != spec.n_features or not len(X) or not np.isfinite(X).all():
        raise ValueError("Invalid benchmark inputs")
    rss_before=psutil.Process().memory_info().rss
    sessions=[]
    start=time.perf_counter()
    for item in manifest["models"]:
        if Path(item["filename"]).name != item["filename"] or digest(root/item["filename"]) != item["sha256"]:
            raise ValueError("Corrupt model bundle")
        sessions.append((session(root/item["filename"]),item))
    load_ms=(time.perf_counter()-start)*1000
    X=np.asarray(features,dtype=np.float32)
    def predict(i):
        row=X[i % len(X):i % len(X)+1]
        for runtime,item in sessions:
            runtime.run([item["probability_output"]],{item["input_name"]:row})
    start=time.perf_counter()
    predict(0)
    cold_ms=(time.perf_counter()-start)*1000
    for i in range(3):
        predict(i)
    samples=[]
    for i in range(iterations):
        start=time.perf_counter()
        predict(i)
        samples.append((time.perf_counter()-start)*1000)
    return {"bundle_id":manifest["bundle_id"],"scope":"desktop CPU; all labels; excludes network",
            "model_count":len(sessions),"load_ms":load_ms,"first_prediction_ms":cold_ms,
            "median_ms":float(np.median(samples)),"p95_ms":float(np.percentile(samples,95)),
            "rss_before_bytes":rss_before,"rss_after_bytes":psutil.Process().memory_info().rss,
            "memory_scope":"RSS snapshots, not peak memory","iterations":iterations,"size_bytes":manifest["size_bytes"],"environment":environment()}
if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--bundle",required=True)
    p.add_argument("--features",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    write_json(a.output,benchmark(a.bundle,sp.load_npz(a.features).toarray()))
