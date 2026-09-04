"""Build a new immutable dataset. Never overwrite historical processed artifacts."""
import argparse
import copy
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .config import CONFIG
from .data_acquisition import load_all
from .featurize import FeatureSpec, canonical_smiles, fingerprint_matrix
from .label_harmonization import LabelHarmonizer, normalize
from .splits import split_indices, scaffold_groups, group_split_indices
from .artifacts import write_json, digest, json_hash, environment

METADATA = {"stimulus", "cid", "new_cid", "cas", "descriptors", "labels", "name", "molecularweight", "iupacname", "isomericsmiles"}
def _columns(frame):
    if frame.index.name and frame.index.name not in frame:
        frame = frame.reset_index()
    return frame.copy()
def _key(series):
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True).replace("", pd.NA)
def _unique(frame, key, value):
    frame = frame[[key, value]].dropna().copy()
    frame[key] = _key(frame[key])
    frame = frame.dropna().drop_duplicates()
    if frame[key].duplicated().any():
        raise ValueError(f"Conflicting {key} -> {value} mapping")
    return frame
def _attach_smiles(beh, sti, mol):
    beh, sti, mol = map(_columns, (beh, sti, mol))
    cid = "CID" if "CID" in sti else "new_CID"
    if cid not in sti or "Stimulus" not in beh or "Stimulus" not in sti:
        raise ValueError("Explicit Stimulus -> CID bridge is required")
    smiles_column = next((c for c in ("IsomericSMILES", "SMILES", "CanonicalSMILES") if c in mol), None)
    if smiles_column is None or "CID" not in mol:
        raise ValueError("Explicit CID -> SMILES mapping is required")
    beh["Stimulus"], sti["Stimulus"] = _key(beh["Stimulus"]), _key(sti["Stimulus"])
    sti[cid], mol["CID"] = _key(sti[cid]), _key(mol["CID"])
    bridge = _unique(sti, "Stimulus", cid).rename(columns={cid: "__cid__"})
    lookup = _unique(mol, "CID", smiles_column).rename(columns={"CID": "__cid__", smiles_column: "__smiles__"})
    # Null identifiers are excluded from mappings; pandas must not match null keys.
    frame = beh.merge(bridge, on="Stimulus", how="left", validate="many_to_one")
    return frame.merge(lookup, on="__cid__", how="left", validate="many_to_one")

def _truthy(value):
    return str(value).strip().lower() in {"1", "1.0", "true"}

def _attach_sigma(beh, sti, cas_map):
    """Require every listed CAS alias to resolve to one validated identity."""
    beh, sti = map(_columns, (beh, sti))
    beh["Stimulus"], sti["Stimulus"] = _key(beh["Stimulus"]), _key(sti["Stimulus"])
    mapping, excluded = [], []
    for stimulus, rows in sti.dropna(subset=["Stimulus"]).groupby("Stimulus"):
        aliases = rows["CAS"].dropna().astype(str).str.strip().unique().tolist()
        identities = {cas_map.get(cas) for cas in aliases}
        if aliases and None not in identities and len(identities) == 1:
            mapping.append({"Stimulus": stimulus, "__smiles__": next(iter(identities))})
        else:
            excluded.append({"stimulus": stimulus, "cas": aliases,
                             "reason": "unresolved_or_conflicting_cas_aliases"})
    bridge = pd.DataFrame(mapping, columns=["Stimulus", "__smiles__"])
    return beh.merge(bridge, on="Stimulus", how="left", validate="many_to_one"), excluded

def _cas_bridge(dfs, frames):
    identities = defaultdict(set)
    for source, frame in frames.items():
        sti = _columns(dfs[f"{source}_stimuli"])
        cas_col = next((c for c in sti if c.lower() == "cas"), None)
        if source == "goodscents":
            pairs = frame[["Stimulus", "__smiles__"]].rename(columns={"Stimulus": "__cas__"})
        elif cas_col:
            sti["Stimulus"] = _key(sti["Stimulus"])
            mapping = _unique(sti, "Stimulus", cas_col).rename(columns={cas_col: "__cas__"})
            pairs = frame[["Stimulus", "__smiles__"]].merge(mapping, on="Stimulus", validate="many_to_one")
        else:
            continue
        for cas, smiles in pairs[["__cas__", "__smiles__"]].itertuples(index=False, name=None):
            canonical = canonical_smiles(smiles)
            if isinstance(cas, str) and canonical:
                identities[cas.strip()].add(canonical)
    return {k: next(iter(v)) for k, v in identities.items() if len(v) == 1}, sorted(k for k,v in identities.items() if len(v)>1)

def build(output, config=None):
    cfg = config or CONFIG
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new dataset directory; existing artifacts are immutable")
    dfs, sources = load_all(config=cfg)
    taxonomy = sorted({normalize(c) for c in dfs["leffingwell_behavior"] if c.lower() not in METADATA})
    frames = {s: _attach_smiles(dfs[f"{s}_behavior"], dfs[f"{s}_stimuli"], dfs[f"{s}_molecules"])
              for s in cfg["sources"] if s != "sigma"}
    cas_map, ambiguous_cas = _cas_bridge(dfs, frames)
    sigma_excluded = []
    if "sigma" in cfg["sources"]:
        frames["sigma"], sigma_excluded = _attach_sigma(dfs["sigma_behavior"], dfs["sigma_stimuli"], cas_map)
    harmonizer = LabelHarmonizer(taxonomy)
    merged, provenance, audit, rejected = defaultdict(set), [], {}, []
    for source, frame in frames.items():
        counts = {"rows": len(frame), "unmapped_identity": 0, "invalid_or_multifragment": 0, "accepted": 0}
        for _, row in frame.iterrows():
            smiles = row["__smiles__"]
            if not isinstance(smiles, str):
                counts["unmapped_identity"] += 1
                rejected.append({"source": source, "stimulus": str(row["Stimulus"]), "reason": "unmapped_identity"})
                continue
            canonical = canonical_smiles(smiles)
            if canonical is None:
                counts["invalid_or_multifragment"] += 1
                rejected.append({"source": source, "stimulus": str(row["Stimulus"]), "reason": "invalid_or_multifragment"})
                continue
            if source in ("leffingwell", "sigma"):
                labels = {harmonizer.map_descriptor(c) for c in frame if c.lower() not in METADATA and not c.startswith("__") and _truthy(row[c])}
                labels.discard(None)
            else:
                desc = "Labels" if source == "arctander" else "Descriptors"
                labels = harmonizer.map_row(row.get(desc, ""))
            if labels:
                merged[canonical].update(labels)
                provenance.append({"source": source, "stimulus": str(row["Stimulus"]), "smiles": canonical, "labels": sorted(labels)})
                counts["accepted"] += 1
        audit[source] = counts
        if counts["accepted"] == 0:
            raise ValueError(f"No validated labeled records from {source}; inspect source identifiers")
    smiles = sorted(merged)
    Y = np.asarray([[int(label in merged[s]) for label in taxonomy] for s in smiles], dtype=np.uint8)
    strategy=cfg["split"].get("strategy","multilabel_random")
    if strategy == "scaffold":
        train,test=group_split_indices(Y,scaffold_groups(smiles),cfg["split"]["test_size"],cfg["seed"])
    elif strategy == "multilabel_random":
        train, test = split_indices(Y, cfg["split"]["test_size"], cfg["seed"])
    else:
        raise ValueError("Unknown split strategy")
    # Label frequency selection uses development data only. Keep test negatives too.
    counts = Y[train].sum(axis=0)
    blocked = set(cfg["labels"].get("blocklist", []))
    candidates = [i for i, l in enumerate(taxonomy) if counts[i] >= cfg["labels"]["min_positive_count"] and l not in blocked]
    chosen = sorted(sorted(candidates, key=lambda i: (-int(counts[i]), taxonomy[i]))[:cfg["labels"]["max_labels"]])
    if not chosen:
        raise ValueError("No labels have sufficient development support")
    labels, Y = [taxonomy[i] for i in chosen], Y[:, chosen]
    spec = FeatureSpec.from_dict(cfg["fingerprint"])
    X, valid = fingerprint_matrix(smiles, spec)
    if not valid.all():
        raise ValueError("Canonicalized molecules failed feature extraction")
    output.mkdir(parents=True)
    for suffix, indices in (("train", train), ("test", test)):
        sp.save_npz(output / f"X_{suffix}.npz", sp.csr_matrix(X[indices]))
        sp.save_npz(output / f"Y_{suffix}.npz", sp.csr_matrix(Y[indices]))
        (output / f"smiles_{suffix}.txt").write_text("\n".join(smiles[i] for i in indices), encoding="utf-8")
    write_json(output / "label_names.json", labels)
    write_json(output / "feature_spec.json", spec.to_dict())
    write_json(output / "provenance.json", provenance)
    write_json(output / "rejected_records.json", rejected)
    harmonizer.unmapped_report().to_csv(output / "unmapped_descriptors.csv", index=False)
    files = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    manifest = {"protocol_version": 2, "identity_join_validated": True, "sources": sources, "source_revision": cfg["data_revision"],
                "source_audit": audit, "ambiguous_cas_excluded": ambiguous_cas, "sigma_identity_exclusions": sigma_excluded,
                "feature_schema_id": spec.schema_id, "feature_spec": spec.to_dict(), "labels": labels,
                "split": {"strategy":strategy,"seed": cfg["seed"], "test_size": cfg["split"]["test_size"], "train": train.tolist(), "test": test.tolist()},
                "label_selection": "development_frequency_only",
                "label_missingness": "Unrecorded descriptors are treated as negatives; absence is not experimentally verified.",
                "evaluation_status": "New split of partly historical molecules; not an independent external test.",
                "n_train": len(train), "n_test": len(test), "train_positives": Y[train].sum(axis=0).tolist(),
                "files": files, "environment": environment()}
    manifest["dataset_id"] = json_hash(files)
    write_json(output / "dataset_manifest.json", manifest)
    print({"output": str(output), "n_train": len(train), "n_test": len(test), "labels": len(labels), "source_audit": audit})
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed",type=int,default=CONFIG["seed"])
    parser.add_argument("--split",choices=["multilabel_random","scaffold"],default="multilabel_random")
    args=parser.parse_args();cfg=copy.deepcopy(CONFIG)
    cfg["seed"]=args.seed;cfg["split"]["strategy"]=args.split
    build(args.output,cfg)
