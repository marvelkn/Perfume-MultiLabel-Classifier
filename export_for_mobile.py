"""Export a complete validated bundle; legacy artifacts are never overwritten."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import onnx
import onnxruntime as ort
import scipy.sparse as sp
from onnxmltools import convert_xgboost, convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from src.artifacts import digest, write_json, json_hash, environment
from src.featurize import FeatureSpec

def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])

def export_bundle(model_directory, output, validation_X):
    source,output = Path(model_directory),Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    meta = json.loads((source/"model_manifest.json").read_text())
    spec = FeatureSpec.from_dict(meta["feature_spec"])
    if spec.schema_id != meta["feature_schema_id"] or len(meta["models"]) != len(meta["labels"]):
        raise ValueError("Model manifest is inconsistent")
    X = np.asarray(validation_X,dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != spec.n_features or not len(X) or not np.isfinite(X).all():
        raise ValueError("Invalid parity features")
    if len(set(meta["labels"])) != len(meta["labels"]) or [m["label"] for m in meta["models"]] != meta["labels"]:
        raise ValueError("Model/label order differs")
    output.mkdir(parents=True)
    entries, parity = [], []
    for j,item in enumerate(meta["models"]):
        if Path(item["filename"]).name != item["filename"] or digest(source/item["filename"]) != item["sha256"]:
            raise ValueError("Native model identity mismatch")
        native = joblib.load(source/item["filename"])
        initial_types = [("features",FloatTensorType([None,spec.n_features]))]
        if meta["algorithm"] == "xgb":
            graph = convert_xgboost(native,initial_types=initial_types,target_opset=15)
        elif meta["algorithm"] == "lgbm":
            graph = convert_lightgbm(native,initial_types=initial_types,target_opset=15,zipmap=False)
        else:
            raise ValueError("Unsupported algorithm")
        onnx.checker.check_model(graph)
        filename = f"{meta['algorithm']}_{j:03d}.onnx"
        target = output/filename
        target.write_bytes(graph.SerializeToString())
        runtime = session(target)
        names = runtime.get_outputs()
        probabilities_name = next((o.name for o in names if o.type == "tensor(float)" and len(o.shape)==2 and o.shape[-1]==2),None)
        if probabilities_name is None:
            raise ValueError("Export must provide a dense [batch,2] probability tensor")
        predicted = runtime.run([probabilities_name],{runtime.get_inputs()[0].name:X})[0][:,1]
        expected = native.predict_proba(X)[:,1]
        threshold = float(item["threshold"])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid model threshold")
        np.testing.assert_allclose(predicted,expected,rtol=1e-5,atol=1e-6)
        crossings = int(np.count_nonzero((predicted >= threshold) != (expected >= threshold)))
        if crossings:
            raise ValueError(f"ONNX conversion changes threshold decisions for {item['label']}")
        parity.append({"label":item["label"],"n_samples":len(X),"max_abs_error":float(np.max(np.abs(predicted-expected))),"threshold_disagreements":crossings})
        entries.append({"label":item["label"],"filename":filename,"sha256":digest(target),"threshold":threshold,
                        "input_name":runtime.get_inputs()[0].name,"probability_output":probabilities_name,"positive_index":1})
        del native,runtime
    bundle = {"version":1,"algorithm":meta["algorithm"],"dataset_id":meta["dataset_id"],
              "feature_spec":spec.to_dict(),"feature_schema_id":spec.schema_id,"models":entries,
              "environment":environment(),"parity":parity,"size_bytes":sum((output/e["filename"]).stat().st_size for e in entries)}
    bundle["bundle_id"] = json_hash({"feature_schema_id":spec.schema_id,"models":entries})
    write_json(output/"feature_spec.json",spec.to_dict())
    write_json(output/"manifest.json",bundle) # Last write marks a complete bundle.
    return bundle

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--models",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--features",required=True,help="Development X_train.npz, never test data")
    args=p.parse_args()
    if Path(args.features).name != "X_train.npz":
        p.error("Use the explicitly named development X_train.npz file")
    features=sp.load_npz(args.features).toarray().astype(np.float32)
    print(export_bundle(args.models,args.output,features[:128])["bundle_id"])
