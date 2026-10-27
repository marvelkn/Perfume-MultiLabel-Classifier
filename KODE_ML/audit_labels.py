"""Audit label dari snapshot beku; tidak mengubah alias, target, atau data training."""
from collections import Counter
import json

import numpy as np
import pandas as pd

from research.preprocessing import (
    ALIASES, load_config, load_sources, map_labels, merge_sources,
    tokens_for,
)
from research.source_identity import normalize
from research.dataset import DATASET, verify, encode_frozen_records


def audit_labels():
    """Kembalikan tabel audit di memori; semua hitungan memakai kode produksi."""
    config = load_config()
    tables, snapshot = load_sources(config)
    records, taxonomy, attrition, rejected, provenance, unmapped, mappings = merge_sources(tables, config)
    verified = verify()
    target, labels, split = encode_frozen_records(records, verified=verified)
    tax, retained = set(taxonomy), set(labels)
    rejected_by_row = {(r["source"], r["row"]): r["reason"] for r in rejected}

    # Bagian ini mengamati semua token, termasuk baris yang tidak mencapai mapper.
    events, raw_rows = [], []
    for source in config["sources"]:
        behavior = tables[f"{source}_behavior"]
        for row_number, (_, row) in enumerate(behavior.iterrows()):
            rejection = rejected_by_row.get((source, row_number), "accepted")
            try:
                tokens = tokens_for(source, row, behavior.columns)
            except ValueError:
                raw_rows.append({"source": source, "row": row_number, "reason": "invalid_behavior"})
                continue
            usable_tokens = [t for t in tokens if isinstance(t, str) and t.strip()]
            raw_rows.append({"source": source, "row": row_number,
                             "reason": rejection, "raw_token_count": len(usable_tokens)})
            for token in usable_tokens:
                normal = normalize(token)
                accepted, not_mapped = Counter(), Counter()
                found = map_labels([token], tax, source, not_mapped, accepted)
                mapped = next(iter(found)) if found else ""
                method = "exact" if mapped == normal and mapped else "alias" if mapped else "unmapped"
                reached = rejection not in {"missing_identity", "invalid_smiles", "multifragment", "invalid_behavior"}
                if not reached:
                    reason = "row_" + rejection
                elif not mapped:
                    reason = "not_in_taxonomy_or_valid_alias"
                elif mapped not in retained:
                    reason = "mapped_but_below_train_support"
                else:
                    reason = "retained_label"
                events.append({"source": source, "row": row_number,
                               "raw_token": token, "normalized_token": normal,
                               "normalization_changed": token != normal,
                               "mapped_label": mapped, "mapping_method": method,
                               "reached_production_mapper": reached,
                               "row_outcome": rejection, "final_use_reason": reason})
    events = pd.DataFrame(events)
    reached = events[events.reached_production_mapper].copy()
    raw_rows = pd.DataFrame(raw_rows)
    # Reconcile individual observed tokens against the historical mapper counters.
    actual_unmapped = reached[reached.mapping_method.eq("unmapped")].groupby(
        ["source", "normalized_token"]).size().to_dict()
    expected_unmapped = {(r.source, r.descriptor): int(r.count) for r in unmapped.itertuples()}
    if actual_unmapped != expected_unmapped:
        raise AssertionError("Token audit does not reconcile with merge_sources unmapped counters.")
    actual_mapped = reached[reached.mapping_method.ne("unmapped")].groupby(
        ["source", "normalized_token", "mapped_label", "mapping_method"]).size().to_dict()
    expected_mapped = {(r.source, r.raw_descriptor, r.label, r.method): int(r.count) for r in mappings.itertuples()}
    if actual_mapped != expected_mapped:
        raise AssertionError("Token audit does not reconcile with merge_sources mapped counters.")
    stored_unmapped = pd.read_csv(DATASET / "unmapped_descriptors.csv")
    stored_mappings = pd.read_csv(DATASET / "label_mappings.csv")
    pd.testing.assert_frame_equal(unmapped.reset_index(drop=True), stored_unmapped, check_dtype=False)
    pd.testing.assert_frame_equal(mappings.reset_index(drop=True), stored_mappings, check_dtype=False)

    # Ulangi aturan seleksi asli, tetapi gunakan indeks train yang sudah dibekukan.
    all_target = np.asarray([[int(name in row) for name in taxonomy]
                             for row in records.labels], dtype=np.uint8)
    train_positive = all_target[split["train"]].sum(axis=0)
    train_negative = len(split["train"]) - train_positive
    selected = (train_positive >= config["labels"]["min_train_positive"]) & (train_negative >= 2)
    selection = pd.DataFrame({"label": taxonomy, "train_positive": train_positive,
                              "train_negative": train_negative, "selected": selected,
                              "selection_reason": np.where(selected, "retained",
                                  np.where(train_positive < config["labels"]["min_train_positive"],
                                           "train_positive_below_30", "train_negative_below_2"))})
    if selection.loc[selection.selected, "label"].tolist() != labels:
        raise AssertionError("Recomputed label selection differs from frozen 109 labels.")
    stored_selection = pd.read_csv(DATASET / "label_selection.csv")
    pd.testing.assert_frame_equal(selection[list(stored_selection.columns)], stored_selection, check_dtype=False)
    if not np.array_equal(all_target[:, selected], target):
        raise AssertionError("Selected targets differ from the frozen targets.")

    alias_rows = []
    for alias, destination in ALIASES.items():
        observed = reached[reached.normalized_token.eq(alias)]
        alias_rows.append({"alias": alias, "destination": destination,
                           "destination_in_taxonomy": destination in tax,
                           "alias_itself_in_taxonomy": alias in tax,
                           "observed_tokens_at_mapper": len(observed),
                           "tokens_using_alias": int(observed.mapping_method.eq("alias").sum()),
                           "status": "exact_takes_priority" if alias in tax else
                                     "destination_outside_taxonomy" if destination not in tax else
                                     "used" if len(observed) else "not_observed"})
    aliases = pd.DataFrame(alias_rows)
    token_inventory = events.groupby(
        ["source", "raw_token", "normalized_token", "mapped_label", "mapping_method",
         "reached_production_mapper", "row_outcome", "final_use_reason"], dropna=False
    ).size().rename("count").reset_index().sort_values(["count", "source"], ascending=[False, True])
    unmatched_inventory = reached[reached.mapping_method.eq("unmapped")].groupby(
        ["normalized_token"], dropna=False).agg(
            token_occurrences=("row", "size"), sources=("source", lambda values: ";".join(sorted(set(values)))),
            raw_spellings=("raw_token", lambda values: json.dumps(sorted(set(values)), ensure_ascii=False))
        ).reset_index().sort_values("token_occurrences", ascending=False)
    per_source = []
    for source in config["sources"]:
        raw = events[events.source.eq(source)]
        used = reached[reached.source.eq(source)]
        accepted_rows_with_unknown = used[used.mapping_method.eq("unmapped") & used.row_outcome.eq("accepted")].row.nunique()
        per_source.append({"source": source, **attrition["sources"][source],
                           "raw_token_occurrences": len(raw),
                           "raw_normalized_terms": raw.normalized_token.nunique(),
                           "mapper_token_occurrences": len(used),
                           "exact_tokens": int(used.mapping_method.eq("exact").sum()),
                           "alias_tokens": int(used.mapping_method.eq("alias").sum()),
                           "unmapped_tokens": int(used.mapping_method.eq("unmapped").sum()),
                           "unmapped_terms": used[used.mapping_method.eq("unmapped")].normalized_token.nunique(),
                           "accepted_rows_with_unmapped_tokens": int(accepted_rows_with_unknown)})
    per_source = pd.DataFrame(per_source)
    summary = {
        "snapshot_revision": config["revision"], "sources": list(config["sources"]),
        "raw_behavior_rows": int(sum(len(tables[f"{s}_behavior"]) for s in config["sources"])),
        "accepted_source_rows_before_merge": int(sum(v["accepted"] for v in attrition["sources"].values())),
        "rejected_source_rows": len(rejected), "unique_molecules": len(records),
        "taxonomy_terms_from_ifra": len(taxonomy), "manual_alias_entries": len(ALIASES),
        "selected_labels": len(labels), "excluded_taxonomy_terms": int((~selected).sum()),
        "min_train_positive": config["labels"]["min_train_positive"], "min_train_negative": 2,
        "all_zero_retained_target_rows_train": int((target[split["train"]].sum(axis=1) == 0).sum()),
        "all_zero_retained_target_rows_test": int((target[split["test"]].sum(axis=1) == 0).sum()),
        "raw_token_occurrences_all_rows": len(events),
        "normalized_terms_all_rows": int(events.normalized_token.nunique()),
        "tokens_reaching_mapper": len(reached),
        "normalized_terms_reaching_mapper": int(reached.normalized_token.nunique()),
        "exact_token_occurrences": int(reached.mapping_method.eq("exact").sum()),
        "alias_token_occurrences": int(reached.mapping_method.eq("alias").sum()),
        "unmapped_token_occurrences": int(reached.mapping_method.eq("unmapped").sum()),
        "unmapped_normalized_terms": len(unmatched_inventory),
        "mapped_normalized_terms": int(reached[reached.mapping_method.ne("unmapped")].normalized_token.nunique()),
        "mapped_distinct_taxonomy_labels": int(reached.loc[reached.mapping_method.ne("unmapped"), "mapped_label"].nunique()),
        "normalization_changed_token_occurrences": int(reached.normalization_changed.sum()),
        "retained_rows_with_unmapped_tokens": int(reached[reached.mapping_method.eq("unmapped") & reached.row_outcome.eq("accepted")][["source", "row"]].drop_duplicates().shape[0]),
        "selection_equals_frozen": True, "targets_equal_frozen": True,
        "mapping_counters_equal_original": True,
        "mapping_tables_equal_saved": True,
        "unmapped_audit_scope": "Only tokens on valid single-molecule rows reach the original mapper. All-row token audit is additional diagnostics, not training data.",
        "semantic_accuracy_validated": False,
    }
    return {"summary": summary, "per_source": per_source, "taxonomy_selection": selection,
            "aliases": aliases, "token_inventory": token_inventory,
            "unmatched_terms": unmatched_inventory, "mappings": mappings,
            "unmapped_by_source": unmapped, "rejected_rows": pd.DataFrame(rejected),
            "raw_row_status": raw_rows, "records": records, "target": target,
            "labels": labels, "taxonomy": taxonomy, "snapshot": snapshot}
