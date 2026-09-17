# Rancangan v6 sebelum training

Pertanyaan: apakah perluasan batas pencarian memperbaiki kandidat C/D, dan apakah
pengaruh fitur/tuning tetap terlihat ketika semua A/B/C/D dinilai dengan repeated
grouped CV yang sama?

Dasar: resep terbaik D v5 pada kedua algoritma mencapai 800 pohon (maksimum lama),
learning rate sekitar 0,0184/0,0133, serta colsample sekitar 0,504/0,505 (dekat
minimum lama 0,5). Itu alasan menguji rentang lebih luas, bukan bukti bahwa pohon
lebih banyak pasti menaikkan skor. Rentang baru dan budget merupakan keputusan
eksperimen yang dinyatakan sebelum run.

Empat studi baru masing-masing 8 jam: xgb_C, xgb_D, lgbm_C, lgbm_D. Resep lama
yang dipilih lewat validasi masuk sebagai trial pertama dan dinilai ulang dalam
budget. Riwayat/skor trial lama tidak diimpor; jangan menafsirkan ini sebagai
pencarian tanpa pengetahuan terdahulu. A/B tetap 100 pohon. Semua delapan kandidat
menjalani 5 seed × 5 fold per label, kemudian fitting dan test setelah seleksi.

TPE tetap seed 42, n_startup_trials 10, n_ei_candidates 24, multivariate false.
Perubahan hanya pada tiga batas search space; tidak menambahkan pruning atau early
stopping yang akan mengubah penggunaan fold dan memerlukan validasi tersendiri.
Seed CV berbeda dari seed pencarian Optuna; ini bukan lima optimasi independen.

Hasil wajib: comparison_all.csv, ablation_deltas.csv (B−A, C−A, D−B, D−C, D−A),
comparison_history.csv, RESULTS_SUMMARY.md, per-label metrics, prediksi test,
OOF/fold, SQLite dan sampler, model joblib per label, serta inference_bundle.json.
Model terpilih ditentukan oleh rata-rata AP validasi; exact tie dalam algoritma
memilih urutan A/B/C/D. Riwayat v3–v5 hanya pembanding, bukan test untuk memilih v6.

Validitas: pemisahan struktur identik/fingerprint tidak membuktikan semua kemiripan
scaffold hilang. Label negatif berarti anotasi tidak tercatat. Test set sudah
diperiksa pada run sebelumnya; v6 tetap eksploratori. CV dipakai memilih/tuning
kandidat sehingga bukan estimasi nested yang bebas bias seleksi. Interval seed
dan bootstrap label bersifat deskriptif; label dan pengulangan tidak independen.
Tidak ada training pseudo-label atau data tambahan di v6.

Referensi teknis:
- [Optuna TPE 4.5](https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [Nested vs non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)

Lihat validation.json untuk pemeriksaan persiapan. Bukti pemeriksaan ZIP final
ada di package_validation.json pada repository asal (di luar ZIP agar checksum
paket tidak merujuk dirinya sendiri). Keduanya bukan hasil training skala penelitian.
