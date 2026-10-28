# Salinan hasil untuk belajar

Ini salinan byte-identik dari hasil utama yang dikunci. Bukan training baru. Buka **comparison_all.csv** dulu: setiap baris adalah satu kandidat A-D dari XGBoost/LightGBM. Pemilihan utama **lgbm_D** berasal dari AP validasi, bukan pencarian skor test tertinggi.

| File | Yang kamu cek |
|---|---|
| comparison_all.csv | Ringkasan delapan kandidat |
| selection_frozen.json | Keputusan dan identitas artefak sebelum final fit/test |
| inference_bundle.json | Kandidat, urutan fitur/label dan ambang utama |
| xgb_A sampai lgbm_D/recipe.json | Parameter yang benar-benar digunakan |
| */trials.csv dan tuning_summary.json | Trial kandidat C/D dan hasil pencarian |
| */test_summary.json | Metrik pada ambang utama 0,5 |
| */test_per_label.json | Label mana yang mudah/sulit menurut evaluasi |
| */test_predictions.npz | Target, skor dan keputusan test asli |
| */test_summary_validation_thresholds.json | Evaluasi tambahan dengan ambang OOF |
| robustness/*/robustness_summary.json | Ringkasan lima seed |
| robustness/*/thresholds.json | Ambang tambahan masing-masing label |
| robustness/lgbm_D/pooled_oof_predictions.npz | Prediksi train ketika menjadi validation |
| audit_label_summary.json | Rekonsiliasi mapping istilah |
| audit_27_oktober/README.md | Audit dan keterbatasan terbaru |

Nama file arsip bukan nama metode. Hasil tambahan V8/V9 tidak dimasukkan ke sini. Notebook tetap membaca artefak asli melalui 03_Kode/results.py. Model .joblib tidak diduplikasi; notebook08 otomatis memuatnya dari hasil utama.

Daftar asal, checksum dan ukuran setiap salinan tercatat di **../ASAL_BAHAN_BELAJAR.json**. Jika hasil asli berubah pada masa mendatang, ini tetap snapshot tanggal 27 Oktober 2026. Cocokkan kembali sebelum memakai untuk laporan baru.
