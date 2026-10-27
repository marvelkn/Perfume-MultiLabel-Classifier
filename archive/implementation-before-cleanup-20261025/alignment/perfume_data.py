"""Five Pyrfume archives -> audited, single-molecule, dynamic-label dataset."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from src.build_dataset import _attach_smiles, _attach_sigma, _cas_bridge, _truthy
from src.label_harmonization import normalize
from .data import ROOT, digest, write_json
from .grouped_splits import POLICY, structure_feature_groups, outer_split, inner_folds, validate_grouped_split

CONFIG = ROOT / "alignment/perfume_config.json"
OUTPUT = ROOT / "data/builds/perfume-five-grouped-v2"
REQUIRED_SOURCES = {"goodscents", "ifra", "leffingwell", "arctander", "sigma"}
IFRA_COLUMNS = ["Descriptor 1", "Descriptor 2", "Descriptor 3"]
# Only explicit lexical equivalents; no fuzzy guesses or taste-to-odor conversion.
ALIASES = {
    "fruit": "fruity", "fruits": "fruity", "flower": "floral", "flowers": "floral",
    "floral notes": "floral", "wood": "woody", "woods": "woody", "herb": "herbal",
    "herbaceous": "herbal", "green notes": "green", "citrus notes": "citrus",
    "spice": "spicy", "spicy notes": "spicy", "sweet notes": "sweet",
    "cream": "creamy", "butter": "buttery", "nut": "nutty", "smoke": "smoky",
    "smokey": "smoky", "earth": "earthy", "mint": "minty", "oil": "oily",
    "musk": "musk like", "animal": "animal like", "animalic": "animal like",
    "jasmine": "jasmin", "savory": "savoury", "tropical": "tropical fruit",
    "sulphurous": "sulfurous", "balsam": "balsamic", "powder": "powdery",
}


def load_config(path=CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config["sources"]) != REQUIRED_SOURCES:
        raise ValueError("Exactly GoodScents, IFRA 2019, Leffingwell, Arctander 1960 and Sigma 2014 are required.")
    if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Use a full pinned commit SHA.")
    if config["labels"]["max_labels"] is not None:
        raise ValueError("This protocol does not impose a top-N label cap.")
    if config["split"].get("group_policy") != POLICY:
        raise ValueError("The active pipeline requires structure/feature grouped splitting.")
    return config


def load_sources(config, *, offline=False):
    """Read only configured files; verify cached bytes rather than silently trusting them."""
    revision = config["revision"]
    cache = ROOT / "data/snapshots/perfume-five-v1" / revision
    old_cache = ROOT / "data/snapshots" / revision
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "manifest.json"
    recorded = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    old_manifest_path = old_cache / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8")) if old_manifest_path.exists() else {}
    tables, manifest = {}, {}
    for source, files in config["sources"].items():
        for kind, rel in files.items():
            target = cache / rel
            url = f"https://raw.githubusercontent.com/pyrfume/pyrfume-data/{revision}/{rel}"
            if target.exists():
                content = target.read_bytes()
                if rel not in recorded or hashlib.sha256(content).hexdigest() != recorded[rel]["sha256"]:
                    raise ValueError(f"Unverified or modified source cache: {rel}")
            elif (old_cache / rel).exists() and rel in old_manifest:
                content = (old_cache / rel).read_bytes()
                if hashlib.sha256(content).hexdigest() != old_manifest[rel]["sha256"]:
                    raise ValueError(f"Historical cache hash mismatch: {rel}")
            else:
                if offline:
                    raise FileNotFoundError(f"Missing offline source: {rel}. Run once without --offline.")
                last_error = None
                for _ in range(3):
                    try:
                        response = requests.get(url, timeout=(10, 30))
                        response.raise_for_status()
                        content = response.content
                        last_error = None
                        break
                    except requests.RequestException as exc:
                        last_error = exc
                if last_error:
                    raise RuntimeError(f"Cannot retrieve {rel}") from last_error
            frame = pd.read_csv(io.BytesIO(content), dtype=str)
            if frame.empty:
                raise ValueError(f"Empty source table: {rel}")
            entry = {"url": url, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            if rel in recorded and recorded[rel] != entry:
                raise ValueError(f"Pinned source changed: {rel}")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            if recorded.get(rel) != entry:
                recorded[rel] = entry
                write_json(manifest_path, recorded)
            manifest[rel] = entry
            tables[f"{source}_{kind}"] = frame
    return tables, manifest


def taxonomy_from_ifra(behavior):
    if not set(IFRA_COLUMNS).issubset(behavior):
        raise ValueError("IFRA must provide its three descriptor columns.")
    return sorted({normalize(value) for value in behavior[IFRA_COLUMNS].stack()
                   if isinstance(value, str) and normalize(value)})


def map_labels(tokens, taxonomy, source, unmapped, mappings):
    accepted = set()
    for token in tokens:
        if not isinstance(token, str) or not token.strip():
            continue
        term = normalize(token)
        mapped = term if term in taxonomy else ALIASES.get(term)
        if mapped in taxonomy:
            accepted.add(mapped)
            mappings[(source, term, mapped, "exact" if term == mapped else "alias")] += 1
        else:
            unmapped[(source, term)] += 1
    return accepted


def tokens_for(source, row, behavior_columns):
    if source == "ifra":
        return [row.get(name) for name in IFRA_COLUMNS]
    if source in ("leffingwell", "sigma"):
        columns = [c for c in behavior_columns if normalize(c) not in {"stimulus", "descriptors", "labels", "cid", "cas"}]
        invalid = [(c, row[c]) for c in columns if pd.notna(row[c]) and str(row[c]).strip().lower()
                   not in {"", "0", "0.0", "1", "1.0", "false", "true"}]
        if invalid:
            raise ValueError(f"Invalid binary behavior values: {invalid[:3]}")
        return [c for c in columns if _truthy(row[c])]
    column = "Labels" if source == "arctander" else "Descriptors"
    value = row.get(column)
    return value.split(";") if isinstance(value, str) else []


def merge_sources(tables, config):
    RDLogger.DisableLog("rdApp.warning")
    taxonomy = taxonomy_from_ifra(tables["ifra_behavior"])
    tax = set(taxonomy)
    frames = {}
    for source in config["sources"]:
        if source != "sigma":
            frames[source] = _attach_smiles(tables[f"{source}_behavior"],
                                           tables[f"{source}_stimuli"], tables[f"{source}_molecules"])
    cas_map, ambiguous = _cas_bridge(tables, frames)
    frames["sigma"], sigma_exclusions = _attach_sigma(tables["sigma_behavior"], tables["sigma_stimuli"], cas_map)
    merged, origins = defaultdict(set), defaultdict(set)
    audit, rejected, provenance = {}, [], []
    unmapped, mappings = Counter(), Counter()
    for source in config["sources"]:
        frame = frames[source]
        expected = IFRA_COLUMNS if source == "ifra" else ["Labels"] if source == "arctander" else ["Descriptors"] if source == "goodscents" else []
        if not set(expected).issubset(frame):
            raise ValueError(f"Unexpected behavior schema: {source}")
        counts = {"rows": len(frame), "missing_identity": 0, "invalid_smiles": 0,
                  "multifragment": 0, "empty_labels": 0, "unmapped_labels": 0,
                  "invalid_behavior": 0, "accepted": 0}
        for row_number, (_, row) in enumerate(frame.iterrows()):
            base = {"source": source, "row": row_number, "stimulus": str(row["Stimulus"])}
            smiles = row["__smiles__"]
            reason = None
            if not isinstance(smiles, str) or not smiles.strip():
                reason = "missing_identity"
            else:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None or mol.GetNumAtoms() == 0:
                    reason = "invalid_smiles"
                elif len(Chem.GetMolFrags(mol)) != 1:
                    reason = "multifragment"
            if reason:
                counts[reason] += 1
                rejected.append(dict(base, reason=reason))
                continue
            try:
                tokens = tokens_for(source, row, tables[f"{source}_behavior"].columns)
            except ValueError as exc:
                counts["invalid_behavior"] += 1
                rejected.append(dict(base, reason="invalid_behavior", detail=str(exc)))
                continue
            labels = map_labels(tokens, tax, source, unmapped, mappings)
            if not labels:
                reason = "empty_labels" if not any(isinstance(t,str) and t.strip() for t in tokens) else "unmapped_labels"
                counts[reason] += 1
                rejected.append(dict(base, reason=reason))
                continue
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            merged[canonical].update(labels)
            origins[canonical].add(source)
            provenance.append(dict(base, smiles=canonical, labels=sorted(labels)))
            counts["accepted"] += 1
        if sum(value for key, value in counts.items() if key != "rows") != counts["rows"]:
            raise AssertionError("Source attrition does not reconcile.")
        if counts["accepted"] == 0:
            raise ValueError(f"No usable records from {source}; inspect its schema and identity mapping.")
        audit[source] = counts
    records = pd.DataFrame([{"source_row": i, "smiles": s, "labels": sorted(merged[s]),
                             "sources": sorted(origins[s])} for i,s in enumerate(sorted(merged))])
    unmap = pd.DataFrame([{"source": s,"descriptor": t,"count":n} for (s,t),n in unmapped.most_common()],
                        columns=["source","descriptor","count"])
    mapped = pd.DataFrame([{"source":s,"raw_descriptor":a,"label":b,"method":m,"count":n}
                           for (s,a,b,m),n in sorted(mappings.items())])
    return records, taxonomy, {"sources":audit,"ambiguous_cas":ambiguous,
                              "sigma_identity_exclusions":sigma_exclusions}, rejected, provenance, unmap, mapped


def encode_and_split(records, taxonomy, config, *, morgan=None):
    y = np.asarray([[int(label in row) for label in taxonomy] for row in records.labels], dtype=np.uint8)
    if morgan is None:
        morgan, _ = extract_features(records, config)
    groups = structure_feature_groups(records.smiles.tolist(), morgan)
    train, test = outer_split(groups, config["split"]["test_size"], config["split"]["seed"])
    positive = y[train].sum(axis=0)
    minimum = config["labels"]["min_train_positive"]
    chosen = [j for j in range(len(taxonomy)) if positive[j] >= minimum and len(train)-positive[j] >= 2]
    if not chosen:
        raise ValueError("No labels have enough training examples.")
    labels, target = [taxonomy[j] for j in chosen], y[:, chosen]
    folds, support = {}, []
    for j, label in enumerate(labels):
        pos = int(target[train,j].sum()); neg = len(train)-pos
        try:
            folds[label] = inner_folds(train, target[:,j], groups,
                                       config["split"]["max_folds"], config["split"]["seed"])
        except ValueError as exc:
            raise ValueError(f"Label {label}: {exc}") from exc
        support.append({"label":label,"train_positive":pos,"train_negative":neg,
                        "folds":len(folds[label]),
                        "positive_groups":len(set(groups[train][target[train,j] == 1])),
                        "negative_groups":len(set(groups[train][target[train,j] == 0]))})
    split = {"train":train.tolist(),"test":test.tolist(),"folds":folds,"eligible_labels":labels,
             "seed":config["split"]["seed"],"group_policy":POLICY,"groups":groups.tolist(),
             "outer_method":"GroupShuffleSplit","inner_method":"StratifiedGroupKFold",
             "test_size_unit":"groups"}
    validate_grouped_split(split, groups)
    selection = [{"label":label,"train_positive":int(positive[j]),"selected":j in chosen}
                 for j,label in enumerate(taxonomy)]
    return target, labels, split, pd.DataFrame(support), pd.DataFrame(selection)


def extract_features(records, config):
    spec = config["features"]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=spec["radius"], fpSize=spec["n_bits"],
                                                     includeChirality=spec["include_chirality"])
    x = np.empty((len(records), spec["n_bits"]), dtype=np.uint8)
    desc = np.empty((len(records), len(spec["descriptors"])), dtype=np.float32)
    for i,s in enumerate(records.smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError("Invalid molecule survived preprocessing.")
        x[i] = gen.GetFingerprintAsNumPy(mol)
        desc[i] = [getattr(Descriptors, name)(mol) for name in spec["descriptors"]]
    if not np.isfinite(desc).all():
        raise ValueError("Nonfinite RDKit features; do not silently impute.")
    return x, desc


def prepare(output=OUTPUT, *, config_path=CONFIG, offline=False):
    config = load_config(config_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Build already exists. Use --output with a new directory; never overwrite a dataset.")
    tables, sources = load_sources(config, offline=offline)
    records, taxonomy, audit, rejected, provenance, unmapped, mappings = merge_sources(tables, config)
    x, descriptors = extract_features(records, config)
    y, labels, split, support, selection = encode_and_split(records, taxonomy, config, morgan=x)
    audit.update(validate_grouped_split(split, np.asarray(split["groups"])))
    audit.update({"record_count":len(records),"train_rows":len(split["train"]),"test_rows":len(split["test"]),
                  "taxonomy_size":len(taxonomy),"label_count":len(labels),
                  "training_all_zero_target_rows":int((y[split["train"]].sum(axis=1)==0).sum()),
                  "canonical_overlap":0,"morgan_overlap_train_test":0,"morgan_plus_5_overlap_train_test":0,
                  "scope":"Selected fragrance-relevant sources, not verified perfume-exclusive records."})
    output.mkdir(parents=True)
    for part in ("train","test"):
        idx=split[part]
        np.savez_compressed(output/f"{part}.npz", row_index=np.asarray(idx),
                            source_row=records.source_row.to_numpy()[idx], morgan=x[idx],
                            descriptors=descriptors[idx], truth=y[idx])
    serial=records.copy()
    for c in ("labels","sources"):
        serial[c]=serial[c].map(json.dumps)
    serial.to_csv(output/"records.csv",index=False)
    pd.DataFrame({"source_row":records.source_row,"smiles":records.smiles,
                  "group_id":split["groups"]}).to_csv(output/"groups.csv",index=False)
    support.to_csv(output/"train_label_support.csv",index=False)
    selection.to_csv(output/"label_selection.csv",index=False)
    unmapped.to_csv(output/"unmapped_descriptors.csv",index=False)
    mappings.to_csv(output/"label_mappings.csv",index=False)
    for filename,value in [("sources.json",sources),("splits.json",split),("audit.json",audit),
                           ("rejected_records.json",rejected),("provenance.json",provenance),("config.json",config)]:
        write_json(output/filename,value)
    feature_spec=dict(config["features"],rdkit_version=rdBase.rdkitVersion,morgan_source="canonical SMILES",
                      descriptor_source="same canonical SMILES",
                      dimensions={"A_C":x.shape[1],"B_D":x.shape[1]+descriptors.shape[1]})
    meta={"protocol":config["id"],"labels":labels,"summary_labels":labels,"feature_spec":feature_spec,
          "taxonomy":taxonomy,"source_names":list(config["sources"]),"source_revision":config["revision"],
          "label_selection":"all labels with >=30 training positives; no top-N cap",
          "metric":"macro Average Precision over all selected labels",
          "split_policy":"GroupShuffleSplit outer; StratifiedGroupKFold per-label inner; structure-or-Morgan components",
          "group_policy":POLICY,
          "training_started":False,"test_evaluated":False,"not_a_suh_dataset_reproduction":True,
          "files":{p.name:digest(p) for p in output.iterdir() if p.is_file()}}
    write_json(output/"dataset_manifest.json",meta)
    environment={n:importlib.metadata.version(n) for n in
                 ("numpy","pandas","scikit-learn","rdkit","xgboost","lightgbm","optuna","joblib")}
    protocol={"id":config["id"],"created_at":datetime.now(timezone.utc).isoformat(),
              "dataset_manifest_sha256":digest(output/"dataset_manifest.json"),"environment":environment,
              "tuning":config["tuning"],"features":feature_spec,"primary_labels":labels,
              "label_count":len(labels),"conditions":["A","B","C","D"],
              "objective":"macro Average Precision over all selected training-supported labels",
              "source_restriction":list(config["sources"]),"test_access":"after all eight candidates are frozen",
              "no_direct_paper_score_comparison":True,"group_policy":POLICY}
    write_json(output/"experiment_protocol.json",protocol)
    return audit


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--config",type=Path,default=CONFIG)
    parser.add_argument("--offline",action="store_true")
    args=parser.parse_args()
    result = prepare(args.output, config_path=args.config, offline=args.offline)
    print(json.dumps({k: result[k] for k in ("record_count", "train_rows", "test_rows",
                                           "taxonomy_size", "label_count", "canonical_overlap")}, indent=2))
