"""Reproduce archived Suh checkpoints and audit v7 features. Never train models."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "v8/protocol.json"
LOCK = ROOT / "v8/source_lock.json"
SLOTS = ["ifra1", "ifra2", "ifra3"]
D8 = ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA",
      "NumRotatableBonds", "HeavyAtomCount", "RingCount"]
AUTHOR_D8 = ["MolWt", "NumHDonors", "NumHAcceptors", "TPSA", "MolLogP",
             "NumRotatableBonds", "HeavyAtomCount", "RingCount"]
SOURCE_NAMES = {
    "arctander": "Arctander 1960", "aromadb": "AromaDB",
    "flavordb(odor)": "FlavorDB (odor)", "flavornet": "Flavornet",
    "tgsc": "GoodScents / TGSC", "ifra": "IFRA 2019",
    "leffingwell": "Leffingwell", "sharma_a": "Sharma 2021a",
    "sharma_b": "Sharma 2021b", "sigma": "Sigma 2014",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def environment():
    return {k: version(k) for k in read_json(CONFIG)["expected_environment"]}


def validate_environment():
    actual = environment()
    expected = read_json(CONFIG)["expected_environment"]
    if actual != expected:
        raise ValueError(f"Dependency mismatch. Expected {expected}; actual {actual}.")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Use Python 3.12 for this frozen V8 preparation.")
    return actual


def verify_reference(reference):
    actual = {}
    for name, expected in read_json(LOCK)["files"].items():
        path = Path(reference) / name
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Reference differs from pinned OSF version: {name}")
        actual[name] = expected["sha256"]
    return actual


def workbook_audit(reference, raw):
    """Compare archived checkpoints; do not invent an expert mapping algorithm."""
    path = Path(reference) / "Dataset1_data_preparation.xlsx"
    with pd.ExcelFile(path, engine="openpyxl") as book:
        sheets = list(book.sheet_names)
        final = pd.read_excel(book, sheet_name="6. Final_3_descriptors")
        source = pd.read_excel(book, sheet_name="2. pyrfume", header=1,
                               usecols=["smiles", "cid", "reference"])
    counts = source["reference"].value_counts(dropna=False)
    observed = set(counts.index)
    if observed != set(SOURCE_NAMES):
        raise ValueError(f"Unexpected archived source names: {observed}")
    inventory = pd.DataFrame([
        {"source_key": k, "source_name": SOURCE_NAMES[k], "archived_rows": int(counts[k])}
        for k in SOURCE_NAMES
    ])
    columns = ["smiles", "cid", *SLOTS]
    left = final[columns].fillna("").astype(str)
    right = raw[columns].fillna("").astype(str)
    left_set = set(left.itertuples(index=False, name=None))
    right_set = set(right.itertuples(index=False, name=None))
    left_dedup = left.drop_duplicates("smiles").reset_index(drop=True)
    right_dedup = right.drop_duplicates("smiles").reset_index(drop=True)
    # Exact row content, membership, multiplicity, and order are different checks.
    audit = {
        "sheets": sheets, "pyrfume_rows": len(source),
        "pyrfume_unique_cids": int(source.cid.nunique()),
        "pyrfume_nonpositive_cid_rows": int((source.cid <= 0).sum()),
        "final_excel_rows": len(final), "final_csv_rows": len(raw),
        "excel_extra_duplicate_full_rows": int(left.duplicated().sum()),
        "csv_extra_duplicate_full_rows_without_molblock": int(right.duplicated().sum()),
        "excel_unique_records_missing_in_csv": len(left_set - right_set),
        "csv_unique_records_absent_in_excel": len(right_set - left_set),
        "same_unique_record_content": left_set == right_set,
        "same_row_count_and_order": left.equals(right),
        "same_after_author_smiles_deduplication": left_dedup.equals(right_dedup),
        "original_github_commit": None,
        "expert_mapping_recomputed": False,
        "raw_to_curated_status": "Archived checkpoints verified; manual curation not reimplemented.",
        "ifra_glossary": "Workbook sheet 1. IFRA2020 is also a curation reference.",
    }
    return inventory, audit


def prepare_checkpoint(raw):
    required = {"smiles", "cid", "molblock", *SLOTS}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing source columns: {required - set(raw.columns)}")
    f = raw.copy()
    f.insert(0, "source_row", np.arange(len(f), dtype=np.int64))
    exclusions = []
    duplicate = f.smiles.duplicated(keep="first")
    for idx in f.loc[duplicate, "source_row"]:
        exclusions.append({"source_row": int(idx), "reason": "duplicate_smiles_keep_first"})
    unique = f.loc[~duplicate].copy()
    missing = unique[["smiles", "molblock"]].isna().any(axis=1)
    for idx in unique.loc[missing, "source_row"]:
        exclusions.append({"source_row": int(idx), "reason": "missing_smiles_or_molblock"})
    complete = unique.loc[~missing].copy()
    complete[SLOTS] = complete[SLOTS].apply(lambda x: x.str.upper())
    labels = sorted(set(complete[SLOTS].stack()))
    unlabeled = complete[SLOTS].isna().all(axis=1)
    for idx in complete.loc[unlabeled, "source_row"]:
        exclusions.append({"source_row": int(idx), "reason": "all_label_slots_missing"})
    frame = complete.loc[~unlabeled].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No labeled records.")
    frame.insert(0, "row_index", np.arange(len(frame), dtype=np.int64))
    frame["stratify_label"] = frame[SLOTS].apply(lambda x: min(x.dropna()), axis=1)
    lookup = {label: j for j, label in enumerate(labels)}
    truth = np.zeros((len(frame), len(labels)), dtype=np.uint8)
    for i, row in enumerate(frame[SLOTS].itertuples(index=False, name=None)):
        for label in row:
            if pd.notna(label):
                truth[i, lookup[label]] = 1
    slot_counts = frame[SLOTS].stack().value_counts()
    summary_labels = sorted(slot_counts[slot_counts >= 30].index.tolist())
    support = pd.DataFrame({
        "label": labels,
        "record_positive_count": truth.sum(axis=0).astype(int),
        "slot_occurrence_count": [int(slot_counts.get(x, 0)) for x in labels],
        "author_summary_ge30_slots": [x in summary_labels for x in labels],
    })
    stages = pd.DataFrame([
        {"stage": "Author Dataset2 CSV", "rows": len(raw)},
        {"stage": "Exact SMILES deduplication", "rows": len(unique)},
        {"stage": "Required SMILES/MolBlock present", "rows": len(complete)},
        {"stage": "At least one label: pre-split checkpoint", "rows": len(frame)},
    ])
    return frame, truth, labels, summary_labels, support, stages, pd.DataFrame(exclusions)


def author_helpers(reference):
    """Load only six reviewed pre-ML helpers, never the notebook's training code."""
    verify_reference(reference)
    nb = read_json(Path(reference) / "author_code.ipynb")
    tree = ast.parse("".join(nb["cells"][2]["source"]))
    names = {"functional_groups_to_vector", "smiles_to_descriptors",
             "molblock_to_fingerprint", "prepare_fg_df", "prepare_md_df", "prepare_st_df"}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "functional_groups" for t in node.targets):
            selected.append(node)
    if len(selected) != 7:
        raise ValueError("Pinned author helper layout changed.")
    env = {"pd": pd, "np": np, "Chem": Chem, "DataStructs": DataStructs,
           "Descriptors": Descriptors, "AllChem": AllChem}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "reviewed_author_helpers", "exec"), env)
    return env


def feature_audit(frame, truth, labels, reference):
    helpers = author_helpers(reference)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False)
    n = len(frame)
    morgan = np.empty((n, 2048), np.uint8)
    desc = np.empty((n, 8), np.float32)
    canonical, connected, issues = [], [], []
    blocks = []
    with rdBase.BlockLogs():
        for i, r in enumerate(frame.itertuples(index=False)):
            mol = Chem.MolFromSmiles(r.smiles)
            block = Chem.MolFromMolBlock(r.molblock)
            if mol is None or block is None:
                raise ValueError(f"Invalid structure source_row={r.source_row}; no zero-fill or removal.")
            morgan[i] = generator.GetFingerprintAsNumPy(mol)
            desc[i] = [getattr(Descriptors, name)(mol) for name in D8]
            canonical.append(Chem.MolToSmiles(mol, isomericSmiles=True))
            connected.append(len(Chem.GetMolFrags(mol)) == 1)
            blocks.append(Chem.MolToSmiles(block, isomericSmiles=False))
            if Chem.MolToSmiles(mol, isomericSmiles=False) != blocks[-1]:
                issues.append({"source_row": int(r.source_row), "cid": int(r.cid),
                               "smiles": r.smiles, "parsed_molblock_smiles": Chem.MolToSmiles(block),
                               "action": "reported_only_no_repair"})
        authored = frame.rename(columns=dict(zip(SLOTS, ["fragrance1", "fragrance2", "fragrance3"])))
        st, ys = helpers["prepare_st_df"](authored, dict.fromkeys(labels))
        md, ym = helpers["prepare_md_df"](authored, labels)
        fg, yf = helpers["prepare_fg_df"](authored, labels)
    for y in [ys, ym, yf]:
        if not np.array_equal(truth, y.to_numpy(dtype=np.uint8)):
            raise AssertionError("Label encoding differs from reviewed author helpers.")
    ref_md_v7order = md.to_numpy()[:, [AUTHOR_D8.index(x) for x in D8]].astype(np.float32)
    if not np.array_equal(ref_md_v7order, desc):
        raise AssertionError("Eight descriptor values differ from the author helpers after reordering.")
    if not np.isfinite(desc).all() or not np.isfinite(md.to_numpy()).all():
        raise ValueError("Nonfinite descriptors; no imputation permitted.")
    ref_st = st.to_numpy(dtype=np.uint8)
    folded = np.maximum(morgan[:, :1024], morgan[:, 1024:])
    mismatched = np.any(ref_st != folded, axis=1)
    source_rows = frame.source_row.to_numpy()
    audit = {
        "v8_shape": [n, 2056], "author_st_shape": list(ref_st.shape),
        "author_md_shape": list(md.shape), "author_fg_shape": list(fg.shape),
        "labels_shape": list(truth.shape),
        "label_encoding_matches_author_helpers": True,
        "d8_matches_author_after_order_and_float32_conversion": True,
        "morgan_fold_to_reference_mismatched_rows": int(mismatched.sum()),
        "morgan_fold_mismatch_source_rows": source_rows[mismatched].astype(int).tolist(),
        "smiles_molblock_structure_differences": issues,
        "multicomponent_records": int(n - sum(connected)),
        "multicomponent_source_rows": source_rows[~np.asarray(connected)].astype(int).tolist(),
        "canonical_duplicate_extra_rows": len(canonical) - len(set(canonical)),
        "v8_identical_fingerprint_extra_rows": n - len({x.tobytes() for x in morgan}),
        "author_identical_fingerprint_extra_rows": n - len({x.tobytes() for x in ref_st}),
        "all_v8_features_finite": bool(np.isfinite(desc).all()),
        "author_library_versions": None,
        "original_molblocks_regenerated": False,
        "feature_parity_claim": "Intentionally different: 2048+8 from SMILES versus separate ST1024 from MolBlock and MD8.",
    }
    matrices = {
        "source_row": source_rows, "cid": frame.cid.to_numpy(dtype=np.int64),
        "morgan_2048": morgan, "descriptors_8": desc,
        "X_v8_2056": np.concatenate([morgan, desc], axis=1).astype(np.float32),
        "truth_200": truth, "connected_structure": np.asarray(connected, dtype=bool),
        "labels": np.asarray(labels), "descriptor_names": np.asarray(D8),
        "reference_morgan_1024": ref_st,
        "reference_md_8": md.to_numpy(dtype=np.float64),
        "reference_md_names": np.asarray(AUTHOR_D8),
        "reference_fg_20": fg.to_numpy(dtype=np.uint8),
        "reference_fg_names": np.asarray(fg.columns, dtype=str),
    }
    if any(a.dtype.hasobject for a in matrices.values()):
        raise TypeError("Object arrays are not permitted in portable data.")
    return matrices, audit, canonical, blocks


def split_diagnostic(frame, truth, matrices, canonical, blocks):
    """Expose the original positional bug; do not export a trainable split."""
    primary = frame.stratify_label
    keep = primary.map(primary.value_counts()).to_numpy() >= 2
    intended = np.flatnonzero(keep)
    actual = np.arange(len(intended))
    tr, te = train_test_split(actual, test_size=0.2, random_state=42,
                              stratify=primary.iloc[intended])
    roles = np.full(len(actual), "train", dtype="<U5")
    roles[te] = "test"
    table = pd.DataFrame({
        "split_position": actual, "reference_partition_diagnostic": roles,
        "intended_preml_row": intended, "actual_feature_preml_row": actual,
        "intended_source_row": frame.source_row.to_numpy()[intended],
        "actual_feature_source_row": frame.source_row.to_numpy()[actual],
        "actual_label_source_row": frame.source_row.to_numpy()[actual],
        "intended_primary_label": primary.iloc[intended].to_numpy(),
        "actual_primary_label": primary.iloc[actual].to_numpy(),
    })
    def overlap(values):
        train_keys = {values[i] for i in tr}
        test_keys = {values[i] for i in te}
        return {"shared_keys": len(train_keys & test_keys),
                "test_rows_with_training_key": sum(values[i] in train_keys for i in te)}
    ref_keys = [r.tobytes() for r in matrices["reference_morgan_1024"]]
    v8_keys = [r.tobytes() for r in matrices["morgan_2048"]]
    support = truth[actual].sum(axis=0)
    summary = {
        "intended_rows": len(intended), "actual_feature_rows": len(actual),
        "removed_singleton_strata": int((~keep).sum()),
        "mismatched_source_row_positions": int((actual != intended).sum()),
        "mismatched_stratification_label_positions": int(
            (primary.iloc[actual].to_numpy() != primary.iloc[intended].to_numpy()).sum()),
        "actual_cohort_rows_outside_intended": len(set(actual) - set(intended)),
        "intended_cohort_rows_missing_from_actual": len(set(intended) - set(actual)),
        "train_rows_diagnostic": len(tr), "test_rows_diagnostic": len(te),
        "reference_fingerprint_overlap": overlap(ref_keys),
        "v8_fingerprint_overlap": overlap(v8_keys),
        "canonical_smiles_overlap": overlap(canonical),
        "nonstereo_molblock_overlap": overlap(blocks),
        "actual_cohort_zero_support_labels": np.asarray(matrices["labels"])[support == 0].tolist(),
        "note": "X and Y remain mutually aligned. The bug is the reset filtered metadata index versus pre-filter X/Y.",
        "split_adopted_for_training": False, "index_bug_corrected": False,
    }
    return table, summary, keep


def comparison_rows(audit):
    n = audit["stages"][-1]["rows"]
    s = audit["split_diagnostic"]
    return [
        {"component": "Dataset sumber", "suh": "10 sources, curated OSF artifacts",
         "v8": "Same pinned workbook, CSV and notebook; 3 checksums verified", "status": "Sama pada arsip"},
        {"component": "Raw -> curated preparation", "suh": "CID merge, IFRA priority, expert harmonization",
         "v8": "Uses archived expert output; complete automated curation unavailable", "status": "Belum direproduksi penuh"},
        {"component": "CSV -> pre-split checkpoint", "suh": "8700 rows after exact-SMILES dedup and required values",
         "v8": f"{n} rows, same row IDs and author label encoding", "status": "Sama pada checkpoint"},
        {"component": "Features", "suh": "ST1024 (MolBlock), MD8 and FG20 separately",
         "v8": "SMILES -> Morgan2048 + MD8 = 2056, v7 descriptor order", "status": "Berbeda disengaja"},
        {"component": "Targets", "suh": "200 observed labels; summary over 100 labels with >=30 slot occurrences",
         "v8": "Preserves all 200 and the fixed 100-label reference list", "status": "Sama pada checkpoint"},
        {"component": "After singleton filter / split", "suh": "8681 reported; filtered index applied to earlier X/Y",
         "v8": f"Diagnostic only: {s['mismatched_source_row_positions']} positions differ; no training split",
         "status": "STOP untuk ditinjau"},
        {"component": "Model", "suh": "RF, XGBoost, LightGBM",
         "v8": "Planned: XGBoost and LightGBM only; no fitting", "status": "Subset model acuan"},
        {"component": "Hyperparameter", "suh": "100 estimators; defaults + shown arguments, historic versions unknown",
         "v8": "Not frozen; baseline/tuned must use common feature data", "status": "Belum dijalankan"},
        {"component": "Tuning", "suh": "No Optuna search in supplied notebook",
         "v8": "Optuna planned only after pre-ML review", "status": "Ditunda"},
        {"component": "Evaluation", "suh": "Accuracy, AUROC, AP called AUPRC, specificity, precision, recall",
         "v8": "Same definitions planned; split and evaluation protocol pending", "status": "Belum dijalankan"},
    ]


def md_table(rows, keys):
    def value(x):
        return str(x).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
                     + ["| " + " | ".join(value(r[k]) for k in keys) + " |" for r in rows])


def write_report(output, audit, comparisons):
    w, f, s = audit["workbook"], audit["features"], audit["split_diagnostic"]
    text = f"""# Hasil audit V8: Suh 10 sumber, Morgan 2048 + 8
Status: **PREML_AUDIT_COMPLETE_ML_BLOCKED**. Tidak ada training atau tuning.

## Yang berhasil diverifikasi
- Ketiga berkas asli cocok dengan SHA-256 OSF versi 1.
- Workbook memuat {w['pyrfume_rows']:,} baris sumber dari 10 sumber.
- CSV final {w['final_csv_rows']:,} baris; workbook final {w['final_excel_rows']:,}.
- Isi rekaman unik keduanya sama: {w['same_unique_record_content']}.
- Sesudah deduplikasi SMILES, isi dan urutan keduanya sama:
  {w['same_after_author_smiles_deduplication']}.
- Checkpoint sebelum split: {f['v8_shape'][0]:,} baris, 200 label.
- V8: 2.056 fitur = Morgan 2048 + 8 deskriptor, dihitung dari SMILES.
- Delapan deskriptor cocok dengan helper Suh setelah penyamaan urutan/float32.
- Encoding label cocok dengan ketiga helper fitur Suh.
- 100 label ringkasan ditentukan oleh frekuensi slot >=30 sesuai kode acuan;
  bukan 100 label satu-satunya yang tersimpan.

## Perbandingan
{md_table(comparisons, ['component', 'suh', 'v8', 'status'])}

## Masalah yang menghentikan kelanjutan ke ML
1. **Indeks split acuan tidak konsisten.** Setelah menyaring 19 baris, metadata
   di-reset, tetapi matriks fitur/label lama tidak ikut disaring.
   {s['mismatched_source_row_positions']:,} posisi menunjuk molekul berbeda dari metadata
   yang dimaksud; {s['mismatched_stratification_label_positions']:,} label stratifikasi
   tidak cocok. Ada {s['actual_cohort_rows_outside_intended']} anggota kohor yang berbeda.
   X dan Y masih berpasangan satu sama lain; jangan menyebut semua pasangan X/Y salah.
2. **Kemiripan identik melintasi split acuan.** Diagnosis split kode acuan menemukan
   {s['reference_fingerprint_overlap']['shared_keys']} fingerprint 1024 bersama,
   mencakup {s['reference_fingerprint_overlap']['test_rows_with_training_key']} baris test.
   Dengan fingerprint V8, ada {s['v8_fingerprint_overlap']['shared_keys']} fingerprint
   bersama. Tidak dibuat split baru atau koreksi otomatis.
3. **Kebijakan molekul berbeda.** {f['multicomponent_records']} rekaman mempunyai
   beberapa fragmen. Semuanya dipertahankan untuk audit Suh; pipeline v7 sebelumnya
   hanya menerima satu komponen. Ini bukan persetujuan training/prediksi campuran parfum.
4. **Satu representasi struktur berbeda.** SMILES dan MolBlock berbeda pada
   {len(f['smiles_molblock_structure_differences'])} rekaman; OR-fold Morgan2048 ke 1024
   berbeda dari referensi pada {f['morgan_fold_to_reference_mismatched_rows']} rekaman.
   Detail ada di audit.json. Tidak diperbaiki/dihapus.
5. **Reproduksi kurasi awal belum penuh.** Tersedia workbook hasil tahapan, tetapi
   kode lengkap pemetaan pakar, commit unduhan Pyrfume, versi RDKit asli, serta
   seed EmbedMolecule asli tidak tersedia. MolBlock tersimpan dipertahankan.
   Perhitungan ulang fitur yang cocok pada runtime ini bukan bukti bitwise
   identik dengan runtime historis yang tidak disertakan.

## Yang memang berbeda atas instruksi pengguna
Representasi akhir mengikuti v7: SMILES, Morgan radius 2, 2048 bit, tanpa
kiralitas, delapan deskriptor berurutan {', '.join(D8)}, float32.
Suh memisahkan ST1024, MD8, dan FG20. Karena itu tidak tepat mengklaim seluruh
pre-ML sama. Kesetaraan yang terverifikasi adalah arsip data dan checkpoint
pembersihan/encoding; fitur adalah modifikasi yang disengaja.

Untuk membuktikan efek tuning kelak, baseline XGBoost, tuned XGBoost,
baseline LightGBM, dan tuned LightGBM harus menerima **data, 2056 fitur,
target, split, serta aturan evaluasi yang sama**. Selisih dengan skor publik
Suh bukan efek tuning murni karena representasi fitur berbeda.

## Kualitas label dan ruang lingkup
0 berarti tidak tercatat pada tiga slot, bukan bukti aroma tidak ada.
ODORLESS adalah label eksplisit tersendiri. Data tetap odor molekuler umum;
sepuluh sumber tidak berarti seluruh rekaman eksklusif parfum.
Tidak ada trimming, imputasi label, oversampling, pseudo-labeling, normalisasi,
atau penggantian label berdasarkan model.

## Cara menafsirkan hasil
Audit pre-ML dapat dijalankan ulang secara offline setelah dependency tersedia.
Berkas PREML_AUDIT_ONLY.npz sengaja belum menjadi train.npz/test.npz.
Exit code 2 berarti audit selesai dan **STOP sebelum ML**. Ini bukan training gagal.
Baca V8_PREML_PLAN.md sebelum memutuskan perbaikan indeks, split, dan fragmen.

Sumber: [artikel Suh](https://www.nature.com/articles/s42004-025-01651-7),
[arsip asli OSF](https://osf.io/vfru6/).
"""
    (output / "HASIL_AUDIT_V8.md").write_text(text, encoding="utf-8")


def prepare(reference=None, output=None):
    config = read_json(CONFIG)
    reference = Path(reference) if reference else ROOT / config["reference"]
    output = Path(output) if output else ROOT / config["output"]
    validate_environment()
    verified = verify_reference(reference)
    if output.exists():
        if (output / "manifest.json").exists():
            return verify(output, reference)
        raise FileExistsError("Incomplete/existing output: inspect it and use a new --output; never overwrite.")
    print("1/5 Verify original data and archived workbook...", flush=True)
    raw = pd.read_csv(reference / "Dataset2_final_w_molblock.csv")
    inventory, workbook = workbook_audit(reference, raw)
    frame, truth, labels, summary_labels, support, stages, excluded = prepare_checkpoint(raw)
    if (len(frame), len(labels), len(summary_labels)) != (8700, 200, 100):
        raise ValueError("Unexpected reference checkpoint counts; stop and investigate.")
    if not workbook["same_unique_record_content"] or not workbook["same_after_author_smiles_deduplication"]:
        raise ValueError("Unexplained workbook/CSV content drift; stop before feature generation.")
    print("2/5 Calculate 2048+8 and compare reviewed reference helpers...", flush=True)
    matrices, features, canonical, blocks = feature_audit(frame, truth, labels, reference)
    print("3/5 Diagnose reference split without adopting or fixing it...", flush=True)
    split_table, split_audit, keep = split_diagnostic(frame, truth, matrices, canonical, blocks)
    observed_counts = truth.sum(axis=0)
    audit = {
        "status": "PREML_AUDIT_COMPLETE_ML_BLOCKED",
        "training_started": False, "tuning_started": False, "ready_for_ml": False,
        "reference_hashes": verified, "workbook": workbook,
        "stages": stages.to_dict("records"), "features": features,
        "split_diagnostic": split_audit,
        "quality": {
            "raw_missing": {k: int(v) for k, v in raw.isna().sum().items()},
            "label_density": float(truth.mean()),
            "labels_per_record_mean": float(truth.sum(axis=1).mean()),
            "labels_per_record_min": int(truth.sum(axis=1).min()),
            "labels_per_record_max": int(truth.sum(axis=1).max()),
            "minimum_label_support": int(observed_counts.min()),
            "maximum_label_support": int(observed_counts.max()),
            "median_label_support": float(np.median(observed_counts)),
            "zero_means": "Not recorded in curated top-three slots; not confirmed sensory absence.",
            "perfume_exclusive": False,
        },
        "unresolved": [
            "author_metadata_feature_index_mismatch", "reference_split_identical_feature_overlap",
            "multicomponent_policy_conflicts_with_v7", "smiles_molblock_structure_difference",
            "manual_raw_to_curated_provenance_and_historical_environment_incomplete",
        ],
        "deliberate_deviations": ["Morgan2048+8 concatenation from SMILES as requested; not Suh ST1024."],
        "environment": environment(), "python": platform.python_version(),
    }
    comparisons = comparison_rows(audit)
    output.mkdir(parents=True, exist_ok=False)
    print("4/5 Save audit-only matrices, identities, and comparison...", flush=True)
    np.savez_compressed(output / "PREML_AUDIT_ONLY.npz", **matrices)
    records = frame.drop(columns="molblock").copy()
    records["canonical_smiles_audit"] = canonical
    records["connected_structure"] = matrices["connected_structure"]
    records["retained_by_intended_singleton_filter"] = keep
    records["labels_json"] = [json.dumps([labels[j] for j in np.flatnonzero(y)]) for y in truth]
    records.to_csv(output / "records.csv", index=False)
    excluded.to_csv(output / "author_exclusions.csv", index=False)
    stages.to_csv(output / "preprocessing_stages.csv", index=False)
    support.to_csv(output / "label_support.csv", index=False)
    inventory.to_csv(output / "source_inventory.csv", index=False)
    split_table.to_csv(output / "REFERENCE_SPLIT_DIAGNOSTIC_ONLY.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "comparison_with_suh.csv", index=False)
    descriptor_frame = pd.DataFrame(matrices["descriptors_8"], columns=D8)
    descriptor_frame.describe(percentiles=[.01, .25, .5, .75, .99]).to_csv(
        output / "descriptor_statistics.csv", index_label="statistic")
    write_json(output / "labels.json", {"all_labels": labels, "author_summary_labels": summary_labels,
                                        "summary_rule": ">=30 slot occurrences before singleton filtering"})
    write_json(output / "audit.json", audit)
    write_report(output, audit, comparisons)
    files = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    manifest = {
        "id": config["id"], "audit_completed": True, "ready_for_ml": False,
        "training_started": False, "tuning_started": False, "cohort": "author_pre_split_8700",
        "reference_hashes": verified,
        "code_sha256": digest(__file__), "protocol_sha256": digest(CONFIG),
        "source_lock_sha256": digest(LOCK), "requirements_sha256": digest(ROOT / "requirements_preml_v8.txt"),
        "feature_spec": config["feature_spec"], "environment": environment(),
        "python_major_minor": [3, 12],
        "arrays": {k: {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": array_digest(a)}
                   for k, a in matrices.items()},
        "files": files,
    }
    write_json(output / "manifest.json", manifest)
    verify(output, reference)
    print("5/5 Audit complete. STOP before ML; read HASIL_AUDIT_V8.md.", flush=True)
    return manifest


def verify(output=None, reference=None):
    config = read_json(CONFIG)
    output = Path(output) if output else ROOT / config["output"]
    reference = Path(reference) if reference else ROOT / config["reference"]
    validate_environment()
    hashes = verify_reference(reference)
    m = read_json(output / "manifest.json")
    if (m["id"] != config["id"] or m["reference_hashes"] != hashes
            or m["code_sha256"] != digest(__file__) or m["protocol_sha256"] != digest(CONFIG)
            or m["source_lock_sha256"] != digest(LOCK)
            or m["requirements_sha256"] != digest(ROOT / "requirements_preml_v8.txt")
            or m["environment"] != environment()
            or m["feature_spec"] != config["feature_spec"]):
        raise ValueError("Build identity changed; do not reuse or overwrite it.")
    if m["ready_for_ml"] or m["training_started"] or m["tuning_started"]:
        raise ValueError("Audit-only manifest was improperly promoted to ML.")
    for name, h in m["files"].items():
        if digest(output / name) != h:
            raise ValueError(f"Build artifact changed: {name}")
    with np.load(output / "PREML_AUDIT_ONLY.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != set(m["arrays"]):
            raise ValueError("Unexpected matrix keys.")
        for name in arrays.files:
            a = arrays[name]
            if array_digest(a) != m["arrays"][name]["sha256"]:
                raise ValueError(f"Array changed: {name}")
        if not np.array_equal(arrays["X_v8_2056"][:, :2048], arrays["morgan_2048"]):
            raise ValueError("Morgan order changed.")
        if not np.array_equal(arrays["X_v8_2056"][:, 2048:], arrays["descriptors_8"]):
            raise ValueError("Descriptor order changed.")
        records = pd.read_csv(output / "records.csv")
        if not np.array_equal(records.source_row, arrays["source_row"]):
            raise ValueError("Record/matrix row identities differ.")
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "verify"], nargs="?", default="run")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = (prepare(args.reference, args.output) if args.action == "run"
              else verify(args.output, args.reference))
    print(json.dumps({"audit_completed": result["audit_completed"],
                      "ready_for_ml": result["ready_for_ml"],
                      "feature_dimensions": 2056,
                      "status": "STOP_BEFORE_ML_REVIEW_FINDINGS"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
