# Catatan eksperimen grouped-v4

Eksperimen ini adalah analisis lanjutan setelah grouped-v3. Tujuannya menambah
budget tuning yang sama untuk XGBoost dan LightGBM tanpa mengubah dataset, split,
fitur, ruang pencarian, objective, bobot kelas, dan metrik.

- Warm-start XGBoost: 10 trial lengkap untuk C dan 10 untuk D.
- Warm-start LightGBM: 43 trial lengkap untuk C dan 52 untuk D.
- Tambahan: 7.200 detik per studi, total 28.800 detik aktif.
- Seleksi: mean macro AUPRC dari 5 repeated grouped CV.
- Pembanding: xgb_D dan lgbm_D grouped-v3 yang sudah dibekukan.
- Analisis: perbandingan berpasangan antarseed dan bootstrap 10.000 kali per label.
- Status: eksploratori; grouped-v3 tetap hasil utama.
