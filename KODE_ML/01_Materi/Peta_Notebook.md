# Peta notebook sesuai PDF

Mulai dari **dasar ilmiah dan gap penelitian** pada halaman 2-3, lalu **alasan pemilihan XGBoost-LightGBM dan batas tanpa RF** pada halaman 5-6 PDF. Saat membuka notebook mandiri, mulai dari **Persiapan**, lalu cari judul berikut dengan Ctrl+F. Baca penjelasan, source yang ditampilkan, dan output setelahnya. Nomor halaman merujuk panduan revisi 22 halaman, 3 November 2026.


## Halaman 7: 01_Sumber_Data.ipynb

[Buka notebook](../01_Sumber_Data.ipynb)

1. **Konfigurasi sumber**
2. **Baca snapshot dan ukur tabel**
3. **Ikuti contoh join GoodScents**


## Halaman 8: 02_Preprocessing_dan_Audit_Label.ipynb

[Buka notebook](../02_Preprocessing_dan_Audit_Label.ipynb)

1. **normalize**
2. **taxonomy_from_ifra**
3. **tokens_for**
4. **map_labels**
5. **merge_sources**
6. **Buktikan asal angka 109**
7. **Rekonsiliasi jumlah baris dan token**


## Halaman 9: 03_Dataset_dan_Fitur.ipynb

[Buka notebook](../03_Dataset_dan_Fitur.ipynb)

1. **calculate_morgan**
2. **calculate**
3. **features**
4. **Lihat satu molekul yang memang ada dalam data**


## Halaman 10: 04_Grup_Split_dan_Fold.ipynb

[Buka notebook](../04_Grup_Split_dan_Fold.ipynb)

1. **Bedakan pembentukan historis dengan pemakaian saat ini**
2. **structure_feature_groups**
3. **outer_split**
4. **inner_folds**
5. **Contoh pembagian satu label**


## Halaman 11: 05_Model_dan_Bobot.ipynb

[Buka notebook](../05_Model_dan_Bobot.ipynb)

1. **training.make_model**
2. **Periksa delapan recipe aktual**
3. **Periksa model final yang sudah ada**


## Halaman 12: 05_Model_dan_Bobot.ipynb

[Buka notebook](../05_Model_dan_Bobot.ipynb)

1. **training.balanced_sample_weights**
2. **Buktikan rumus bobot pada label aktual**
3. **training.fit**


## Halaman 13: 06_Optuna_CV_dan_OOF.ipynb

[Buka notebook](../06_Optuna_CV_dan_OOF.ipynb)

1. **Baca kebijakan tuning**
2. **tuning.suggest**
3. **tuning.tune**
4. **Baca hasil empat studi sebenarnya**


## Halaman 14: 06_Optuna_CV_dan_OOF.ipynb

[Buka notebook](../06_Optuna_CV_dan_OOF.ipynb)

1. **validation.cross_validate**
2. **validation.repeated_grouped_cv**


## Halaman 15: 06_Optuna_CV_dan_OOF.ipynb

[Buka notebook](../06_Optuna_CV_dan_OOF.ipynb)

1. **validation.repeated_grouped_cv**
2. **Hitung ulang ambang dari OOF tersimpan**


## Halaman 16: 07_Evaluasi_dan_Pemilihan.ipynb

[Buka notebook](../07_Evaluasi_dan_Pemilihan.ipynb)

1. **metrics.binary_metrics**
2. **metrics.summarize**
3. **Hitung ulang metrik delapan kandidat**


## Halaman 17: 07_Evaluasi_dan_Pemilihan.ipynb

[Buka notebook](../07_Evaluasi_dan_Pemilihan.ipynb)

1. **metrics.binary_metrics**
2. **Hitung ulang metrik delapan kandidat**


## Halaman 18: 07_Evaluasi_dan_Pemilihan.ipynb

[Buka notebook](../07_Evaluasi_dan_Pemilihan.ipynb)

1. **metrics.binary_metrics**
2. **metrics.summarize**


## Halaman 19: 06_Optuna_CV_dan_OOF.ipynb

[Buka notebook](../06_Optuna_CV_dan_OOF.ipynb)

1. **validation.f1_from_metrics**
2. **validation.select_validation_thresholds**
3. **Hitung ulang ambang dari OOF tersimpan**


## Halaman 20: 07_Evaluasi_dan_Pemilihan.ipynb

[Buka notebook](../07_Evaluasi_dan_Pemilihan.ipynb)

1. **Baca delapan kandidat dan hasil utama**
2. **Reproduksi pilihan dari CV tersimpan**
3. **Baca delta dan cek kesamaan cakupan**


## Halaman 21: 08_Prediksi_Molekul.ipynb

[Buka notebook](../08_Prediksi_Molekul.ipynb)

1. **Temukan hasil utama dan kontrak model**
2. **inference.predict**
3. **Jalankan prediksi nyata**
4. **Lihat keputusan positif**


Urutan ini untuk belajar, bukan ketergantungan eksekusi antar-notebook. Tidak mengaktifkan gerbang eksperimen baru.
