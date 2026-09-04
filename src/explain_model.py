"""All-label native TreeSHAP on held-out development examples, in raw-margin units."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .artifacts import digest, write_json
from .experiments import context
from .featurize import FeatureSpec, DESCRIPTORS
from .runtime import ResourceGuard

def contributions(model, algorithm, X):
    if algorithm == "xgb":
        import xgboost as xgb
        matrix=xgb.DMatrix(X)
        values=model.get_booster().predict(matrix,pred_contribs=True)
        margin=model.get_booster().predict(matrix,output_margin=True)
    elif algorithm == "lgbm":
        values=model.booster_.predict(X,pred_contrib=True)
        margin=model.booster_.predict(X,raw_score=True)
    else:
        raise ValueError("Unknown learner")
    np.testing.assert_allclose(values.sum(axis=1),margin,rtol=1e-4,atol=1e-5)
    return np.asarray(values[:,:-1])

def explain(run, algorithm, output, samples=64, temperature_file=None):
    run,cfg,X,_,dataset=context(run)
    output=Path(output)
    if output.exists(): raise FileExistsError("Choose a fresh explanation directory")
    if samples < 1 or samples > 256: raise ValueError("Use 1 to 256 development samples")
    guard=ResourceGuard(cfg["resources"],temperature_file);guard.check(force=True)
    source=run/algorithm
    manifest=json.loads((source/"model_manifest.json").read_text())
    if manifest["dataset_id"] != dataset["dataset_id"]: raise ValueError("Dataset/model mismatch")
    if [m["label"] for m in manifest["models"]] != manifest["labels"]: raise ValueError("Label order mismatch")
    indices=np.random.default_rng(cfg["seed"]).permutation(cfg["splits"]["threshold"])[:samples]
    matrix=[]
    for item in manifest["models"]:
        guard.check(force=True)
        if Path(item["filename"]).name != item["filename"] or digest(source/item["filename"]) != item["sha256"]:
            raise ValueError("Model integrity failure")
        model=joblib.load(source/item["filename"])
        values=contributions(model,algorithm,X[indices])
        matrix.append(np.abs(values).mean(axis=0))
        del model
    output.mkdir(parents=True)
    spec=FeatureSpec.from_dict(manifest["feature_spec"])
    names=[f"Morgan_bit_{i}" for i in range(spec.n_bits)] + (list(DESCRIPTORS) if spec.use_descriptors else [])
    pd.DataFrame(matrix,index=manifest["labels"],columns=names).to_csv(output/"mean_absolute_shap.csv")
    write_json(output/"explanation_manifest.json",{"algorithm":algorithm,"dataset_id":dataset["dataset_id"],
        "model_manifest_sha256":digest(source/"model_manifest.json"),"sample_indices":indices.tolist(),
        "sample_partition":"development threshold holdout","units":"raw model margin (log odds), not probability",
        "interpretation":"Model feature contributions; Morgan bits can collide and are not unique causal chemical substructures."})
    return output

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True);p.add_argument("--model",choices=["xgb","lgbm"],required=True)
    p.add_argument("--output",required=True);p.add_argument("--samples",type=int,default=64);p.add_argument("--temperature-file")
    a=p.parse_args();print(explain(a.run,a.model,a.output,a.samples,a.temperature_file))
if __name__=="__main__": main()
