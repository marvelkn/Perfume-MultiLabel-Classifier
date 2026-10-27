"""Cleaning lima sumber dari snapshot lokal; anggota kohor/split tetap beku."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from .runtime import ROOT, read, digest
from .source_identity import _attach_smiles, _attach_sigma, _cas_bridge, _truthy, normalize
from .splits import POLICY
from .dataset import DATASET, verify, encode_frozen_records
CONFIG = ROOT / "research/config/sources.json"
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

def load_sources(config):
    """Baca snapshot yang sudah ada. Tidak mengunduh atau mengubah cache."""
    cache = ROOT / "data/snapshots/perfume-five-v1" / config["revision"]
    manifest = read(cache / "manifest.json")
    tables, selected = {}, {}
    for source, files in config["sources"].items():
        for kind, relative in files.items():
            path = cache / relative
            if relative not in manifest or digest(path) != manifest[relative]["sha256"]:
                raise ValueError(f"Source snapshot changed: {relative}")
            frame = pd.read_csv(path, dtype=str)
            if frame.empty:
                raise ValueError(f"Empty table: {relative}")
            tables[f"{source}_{kind}"] = frame
            selected[relative] = manifest[relative]
    return tables, selected


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

def review_frozen_preprocessing():
    """Ulangi cleaning di memori untuk audit; tidak membuat dataset atau split baru."""
    config = load_config()
    tables, sources = load_sources(config)
    records, taxonomy, audit, rejected, provenance, unmapped, mappings = merge_sources(tables, config)
    checked = verify()
    target, labels, split = encode_frozen_records(records, verified=checked)
    return {"records": records, "target": target, "labels": labels, "split": split,
            "taxonomy": taxonomy, "audit": audit, "rejected": rejected,
            "provenance": provenance, "unmapped": unmapped, "mappings": mappings,
            "sources": sources}
