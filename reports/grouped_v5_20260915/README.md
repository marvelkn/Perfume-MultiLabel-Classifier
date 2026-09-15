# Grouped-v5: final D-only tuning audit

V5 memperbaiki kehilangan budget `lgbm_D` pada grouped-v4. Hanya kondisi D
diteruskan karena repeated-CV AUPRC D mengungguli C sebesar 0,03005 pada XGBoost
dan 0,02335 pada LightGBM.

- XGBoost D: tambahan baru 14.400 detik.
- LightGBM D: tambahan baru 14.400 detik.
- Ruang pencarian, objective, bobot kelas, dataset, dan split tidak berubah.
- Budget disimpan setiap 15 detik agar interupsi tidak menghabiskan seluruh sisa waktu.
- Seleksi memakai lima repeated grouped CV sebelum test dibuka.
- Grouped-v3 tetap hasil utama; v5 adalah audit eksploratori terakhir.
