# 11. Peta source dan cuplikan aktual
Ini penunjuk file dan fungsi aktif pada 27 Oktober 2026. Semua lokasi di tabel dihitung langsung dari AST source. Cuplikan di bawah dibaca dari file tersebut, bukan ditulis ulang sebagai algoritma lain. Nomor baris dan SHA256 membantu mencocokkan versi; keduanya bukan parameter ML.
| Source di research | Fungsi | Baris | Peran singkat |
|---|---|---|---|
| source_identity.py | normalize | 32-36 | Identitas dan istilah |
| source_identity.py | _attach_smiles | 57-71 | Identitas dan istilah |
| source_identity.py | _attach_sigma | 78-92 | Identitas dan istilah |
| preprocessing.py | load_sources | 42-57 | Baca dan bersihkan lima sumber |
| preprocessing.py | map_labels | 67-79 | Baca dan bersihkan lima sumber |
| preprocessing.py | tokens_for | 82-94 | Baca dan bersihkan lima sumber |
| preprocessing.py | merge_sources | 97-164 | Baca dan bersihkan lima sumber |
| preprocessing.py | review_frozen_preprocessing | 166-176 | Baca dan bersihkan lima sumber |
| features.py | calculate | 19-26 | Struktur menjadi angka |
| features.py | calculate_morgan | 29-33 | Struktur menjadi angka |
| features.py | features | 41-45 | Struktur menjadi angka |
| dataset.py | verify | 14-79 | Kontrak data beku |
| dataset.py | load_partition | 82-93 | Kontrak data beku |
| dataset.py | encode_frozen_records | 96-109 | Kontrak data beku |
| splits.py | inner_folds | 6-18 | Grup dan fold |
| splits.py | validate_grouped_split | 21-52 | Grup dan fold |
| training.py | make_model | 9-18 | Classifier dan fitting |
| training.py | balanced_sample_weights | 21-30 | Classifier dan fitting |
| training.py | fit | 33-53 | Classifier dan fitting |
| training.py | fit_final_models | 55-66 | Classifier dan fitting |
| tuning.py | atomic_dump | 11-15 | Pencarian pengaturan |
| tuning.py | event | 17-23 | Pencarian pengaturan |
| tuning.py | suggest | 25-36 | Pencarian pengaturan |
| tuning.py | initialize | 38-59 | Pencarian pengaturan |
| tuning.py | tune | 72-134 | Pencarian pengaturan |
| validation.py | cross_validate | 8-66 | CV dan OOF |
| validation.py | f1_from_metrics | 69-71 | CV dan OOF |
| validation.py | prediction_diagnostics | 74-86 | CV dan OOF |
| validation.py | select_validation_thresholds | 89-116 | CV dan OOF |
| validation.py | repeated_grouped_cv | 119-180 | CV dan OOF |
| experiment.py | verify_files | 32-34 | Urutan eksperimen |
| experiment.py | preflight | 36-63 | Urutan eksperimen |
| experiment.py | select_candidates | 66-71 | Urutan eksperimen |
| experiment.py | evaluate_test | 73-109 | Urutan eksperimen |
| experiment.py | paired_cv | 111-137 | Urutan eksperimen |
| experiment.py | summaries | 140-219 | Urutan eksperimen |
| experiment.py | reject_historical_run | 222-229 | Urutan eksperimen |
| experiment.py | run_all | 232-300 | Urutan eksperimen |
| experiment.py | main | 302-313 | Urutan eksperimen |
| metrics.py | binary_metrics | 10-30 | Evaluasi |
| metrics.py | summarize | 33-44 | Evaluasi |
| metrics.py | compare | 47-54 | Evaluasi |
| inference.py | predict | 16-58 | Prediksi tersimpan |
| inference.py | main | 61-67 | Prediksi tersimpan |
| runtime.py | digest | 19-20 | I/O, hash dan penjagaan eksekusi |
| runtime.py | read | 22-23 | I/O, hash dan penjagaan eksekusi |
| runtime.py | write | 26-31 | I/O, hash dan penjagaan eksekusi |
| runtime.py | environment | 34-36 | I/O, hash dan penjagaan eksekusi |
| runtime.py | compute_profile | 39-46 | I/O, hash dan penjagaan eksekusi |
| runtime.py | check_host | 49-54 | I/O, hash dan penjagaan eksekusi |
| runtime.py | run_lock | 58-72 | I/O, hash dan penjagaan eksekusi |
| runtime.py | save_npz | 93-99 | I/O, hash dan penjagaan eksekusi |
| results.py | main_run | 5-13 | Alamat hasil utama |

## Cuplikan inti yang bisa kamu ikuti
Baca ini setelah bab 01-10. Source lengkap tetap ada di research dan ditampilkan notebook lewat tampilkan_sumber. Penjelasan input, output dan alasannya ada di bab terkait.

## features.py: calculate_morgan

Baris 29-33. SHA256 file: `8a1774f10afe12c65985b1e326903f5509ad68814a61f6cb2d7df086f14564d0`.

```python
def calculate_morgan(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, fpSize=N_BITS, includeChirality=False)
    return np.asarray([generator.GetFingerprintAsNumPy(molecule(smi, i))
                       for i, smi in enumerate(smiles)], dtype=np.uint8)
```

## features.py: features

Baris 41-45. SHA256 file: `8a1774f10afe12c65985b1e326903f5509ad68814a61f6cb2d7df086f14564d0`.

```python
def features(key, morgan, descriptors):
    if (morgan.ndim != 2 or morgan.shape[1] != N_BITS or descriptors.shape != (len(morgan), 8)
            or not np.isfinite(descriptors).all()):
        raise ValueError("Expected Morgan 2048 and eight ordered descriptors per molecular row.")
    return np.column_stack([morgan, descriptors[:, :descriptor_count(key)]]).astype(np.float32)
```

## training.py: balanced_sample_weights

Baris 21-30. SHA256 file: `4b8f34ca70a8263b3a8a25e7c0e2b3e46bc1e8767be2149ec7cf4c804024e184`.

```python
def balanced_sample_weights(y):
    """Use the same fold-local weighting rule for XGBoost and LightGBM."""
    values = np.asarray(y, dtype=np.int8)
    if values.ndim != 1 or len(values) == 0 or not np.isin(values, [0, 1]).all():
        raise ValueError("Expected a non-empty binary target vector.")
    counts = np.bincount(values, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Balanced weighting requires both classes.")
    per_class = len(values) / (2.0 * counts)
    return per_class[values]
```

## training.py: make_model

Baris 9-18. SHA256 file: `4b8f34ca70a8263b3a8a25e7c0e2b3e46bc1e8767be2149ec7cf4c804024e184`.

```python
def make_model(algorithm, parameters):
    params = dict(parameters)
    profile = compute_profile()
    threads = profile["threads_per_model"]
    if algorithm == "xgb":
        return XGBClassifier(n_jobs=threads, random_state=42, eval_metric="logloss",
                             tree_method=profile["xgboost_tree_method"], **params)
    if algorithm == "lgbm":
        return LGBMClassifier(n_jobs=threads, random_state=42, verbosity=-1, **params)
    raise ValueError("Unknown algorithm.")
```

## inference.py: predict

Baris 16-58. SHA256 file: `a9cdd7d49ed7a308134c9d6b82b47f1cbfcd720a9c58d02511ecf78a9b432a4a`.

```python
def predict(run, smiles):
    run = Path(run)
    bundle = core.read(run/"inference_bundle.json")
    completion = core.read(run/"completion_manifest.json")["files"]
    if digest(run/"inference_bundle.json") != completion["inference_bundle.json"]:
        raise ValueError("Inference bundle changed.")
    if bundle["environment"] != core.environment():
        raise ValueError("Use the pinned training environment.")
    spec = bundle["feature_spec"]
    names = spec["descriptors"]
    if (names not in ([], list(DESCRIPTORS8)) or spec["n_bits"] != N_BITS
            or bool(names) != bundle["use_descriptors"]):
        raise ValueError("Unsupported descriptor order.")
    if spec.get("rdkit_version", rdBase.rdkitVersion) != rdBase.rdkitVersion:
        raise ValueError("RDKit version differs.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Input must be one valid connected molecule.")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=spec["radius"], fpSize=spec["n_bits"], includeChirality=spec["include_chirality"])
    x = np.concatenate([generator.GetFingerprintAsNumPy(mol),
                        [getattr(Descriptors,name)(mol) for name in names]]).astype(np.float32)[None,:]
    if not np.isfinite(x).all() or x.shape[1] != spec["dimensions"]["input"]:
        raise ValueError("Feature schema mismatch.")
    output = []
    for label in bundle["labels"]:
        entry = bundle["models"][label]
        path = run/bundle["models_directory"]/entry["file"]
        if digest(path) != entry["sha256"]:
            raise ValueError("Model checksum changed.")
        model = joblib.load(path)
        if model.n_features_in_ != x.shape[1]:
            raise ValueError("Model expects a different feature count.")
        # The ndarray feature order is frozen and checked above.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names", category=UserWarning)
            score = float(model.predict_proba(x)[0,1])
        output.append({"label":label, "score":score, "predicted":score>=bundle["primary_threshold"]})
    return {"smiles":Chem.MolToSmiles(mol), "candidate":bundle["selected_candidate"],
            "descriptors":len(names), "threshold":bundle["primary_threshold"],
            "score_meaning":"uncalibrated model score, not aroma intensity",
            "labels":sorted(output,key=lambda row:row["score"],reverse=True)}
```
