# 7 Optuna validasi dan training final

## Yang perlu kamu pahami

Targetnya: bisa membedakan tuning, validasi, dan final fit.

**Optuna** adalah library pencari pengaturan model. **Trial** berarti satu percobaan satu paket pengaturan. Contohnya jumlah pohon, learning rate, dan kedalaman pohon. Dalam project ini, satu trial mencoba **satu paket pengaturan bersama untuk 109 label**, bukan 109 paket pengaturan terpisah.

## Urutan kerja penelitian

| Tahap | Apa yang dilakukan | Hasil yang diteruskan |
|---|---|---|
| Skenario A/B | Baseline n_estimators=100 (target 100 iterasi boosting), tanpa Optuna | Resep baseline |
| Skenario C/D | Mencoba resep lewat Optuna | Resep terbaik berdasarkan validasi |
| Repeated grouped CV | Mengecek delapan kandidat dengan lima seed | Nilai validasi dan prediksi OOF |
| Pembekuan keputusan | Mengunci resep, threshold, dan pilihan kandidat | `selection_frozen.json` |
| Final fit | Belajar ulang pada seluruh 5.353 baris train | 109 model final tiap kandidat |
| Test | Memakai 1.333 baris test yang ditetapkan | Hasil evaluasi akhir |

A/C memakai Morgan 2.048 saja; B/D memakai Morgan 2.048 ditambah delapan deskriptor. Empat study adalah xgb_C, xgb_D, lgbm_C, dan lgbm_D. Masing-masing punya anggaran aktif delapan jam: total anggaran tuning 32 jam. Ini bukan total waktu seluruh penelitian, karena CV dan final fit juga membutuhkan waktu.

## Baca satu trial dahulu

Sumber: `KODE_ML/03_Kode/tuning.py`, fungsi `tune`, mulai baris 72. Potongan asli baris 98–105:

```python
                    params=suggest(trial,algorithm,protocol)
                    trial.set_user_attr('resolved_parameters',params)
                    atomic_dump(study.sampler,folder/'sampler.joblib')
                    event(run,'trial_started',candidate=key,trial=trial.number)
                    values=cross_validate(algorithm,params,x,y,meta['labels'],split['folds'],positions,limits,meta['summary_labels'])
                    score=summarize(values,meta['summary_labels'])['metrics']['auprc']
                    study.tell(trial,score)
                    event(run,'trial_completed',candidate=key,trial=trial.number,macro_ap=score,best_macro_ap=study.best_value)
```

`suggest` memilih pengaturan; `cross_validate` melatih dan memvalidasi; `summarize` merangkum label; `study.tell` mengirim skor ke Optuna. **Skor lebih besar lebih baik.** TPE memakai hasil trial selesai untuk usulan berikutnya. Fase startup berlangsung sampai sepuluh trial selesai: resep awal diantrekan, usulan sisanya masih acak. Trial terbaik XGBoost adalah trial 2, masih fase awal. Jadi jangan mengklaim TPE adaptif terbukti mengungguli pencarian acak.

Nama kunci `auprc` di sini dihitung sebagai **Average Precision atau AP**, bukan integrasi trapezoid kurva. Objective trial adalah rata-rata AP antar-fold untuk setiap label, lalu rata-rata 109 label. Nilai logloss tidak menjadi skor pemilihan trial. Logloss berkaitan dengan kesalahan probabilitas ketika classifier belajar; AP dipakai di luar classifier untuk menilai kualitas peringkat hasilnya. Protokol tidak memakai early stopping atau pruning trial.

**Input:** matriks fitur train `(5353, 2048)` atau `(5353, 2056)`, target `(5353, 109)`, daftar fold, dan batas komputasi. **Output:** satu angka objective per trial; akhirnya satu resep terbaik per study. Trial yang terputus karena anggaran habis ditandai gagal. Waktunya tetap dihitung.

## Fold itu apa dan apa yang terjadi di kode

**Fold** adalah bagian data untuk latihan dan pengecekan bergantian. Pada satu putaran, sebagian baris menjadi train fold dan sebagian menjadi validation fold. Model hanya belajar dari train fold, lalu menebak validation fold yang belum dipakai belajar pada putaran tersebut.

Lihat `KODE_ML/03_Kode/validation.py`, fungsi `cross_validate`, baris 34–41:

```python
        for fold_index, fold in enumerate(folds[name]):
            tr = np.array([positions[i] for i in fold["train"]], dtype=int)
            va = np.array([positions[i] for i in fold["validation"]], dtype=int)
            model = fit(algorithm, params, x[tr], y[tr, j], limits)
            probabilities = model.predict_proba(x[va])[:, 1]
            decisions = (probabilities >= 0.5).astype(np.int8)
            metrics = binary_metrics(y[va, j], probabilities, decisions)
            fold_values.append(metrics)
```

`tr` menyimpan posisi baris latihan; `va` posisi validasi. `x[tr]` berukuran `(jumlah_train_fold, jumlah_fitur)`; `y[tr, j]` hanya satu label. `predict_proba` menghasilkan dua skor per baris; `[:, 1]` mengambil skor positif. Batas 0,5 menghasilkan keputusan 0/1. Jumlah baris fold mengikuti pembagian kelompok sehingga tidak harus persis sama.

Pembagian dibuat **per label** dengan menjaga anggota kelompok molekul tetap bersama. Jadi jangan menganggap seluruh label pasti memakai satu daftar validation fold yang identik. Aturan ini menjaga anggota grup historis tetap bersama. Seluruh kemiripan struktur atau scaffold belum dijamin terpisah.

## Seed dan OOF

`repeated_grouped_cv`, mulai baris 119, mengulang CV memakai seed **7, 21, 42, 84, 100**. Seed adalah angka pengendali pengacakan. Lima angka ini mengubah pembagian CV. Sementara itu `make_model` tetap memakai `random_state=42`. Jadi seed pembagian data dan seed classifier punya tugas berbeda.

**OOF** berarti prediksi untuk baris train dari model yang tidak melatih baris itu pada fold bersangkutan. Satu seed menghasilkan matriks OOF `(5353, 109)`. Lima hasil dirata-ratakan menjadi matriks berukuran sama. Kode memakainya untuk mencoba threshold per label yang memaksimalkan F1. Threshold ini berasal dari validasi, bukan test. Pelaporan utama dan inference bundle tetap memakai **0,5**.

Pemilihan memakai rata-rata macro AP kelima seed. Nilainya bisa berbeda dari AP atas rata-rata skor OOF: merata-ratakan metrik berbeda dengan merata-ratakan skor.

## Model final dan batas kesimpulan

`KODE_ML/03_Kode/experiment.py`, `run_all`, baris 232–301 mengatur urutan ini. Setelah keputusan dibekukan, `fit_final_models` belajar ulang memakai seluruh train. Model final ini berbeda dari model sementara setiap fold. Test tidak ikut final fit.

Hasil tersimpan memilih **xgb_D** dalam XGBoost dan **lgbm_D** dalam LightGBM. Pilihan keseluruhan adalah **lgbm_D**. Bukti dibaca dari `complete.json`, melalui lokasi hasil yang ditetapkan `results.main_run()`, bukan menebak folder dengan angka versi terbesar.

Batasnya: repeated CV ini **bukan nested CV**. Data pengembangan turut dipakai mencari dan menilai resep sehingga hasil CV bersifat deskriptif. Protokol juga mencatat test historis pernah diperiksa; penelitian bersifat eksploratif. Jangan menyebutnya pembuktian konfirmatori tanpa batasan.

## Bekal sidang

**“Optuna mencari model atau label?”** “Optuna mencari paket hyperparameter. Satu paket diuji pada 109 classifier label dan dinilai dengan macro AP validasi.”

**“Kenapa final training setelah CV?”** “CV membantu memilih resep tanpa melatih validation fold pada putarannya. Setelah resep dikunci, model final belajar dari seluruh train agar semua data latih dimanfaatkan.”

## Cek pemahaman

- [ ] Aku bisa membedakan trial, fold, seed, dan kandidat.
- [ ] Aku tahu target Optuna adalah AP, bukan logloss.
- [ ] Aku tahu test tidak dipakai untuk mencari threshold dalam run ini.
- [ ] Aku bisa menjelaskan batas repeated CV yang tidak nested.
