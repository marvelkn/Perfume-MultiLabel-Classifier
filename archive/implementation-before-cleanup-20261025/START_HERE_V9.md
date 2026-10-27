# Mulai V9 di komputer kampus

**Paket ini menjalankan training lengkap**, bukan audit V8.

- Windows: pastikan Python **3.12 64-bit** terpasang, lalu klik dua kali **RUN_GROUPED_V9.cmd**.
- Linux: buka terminal di folder ini, lalu `bash RUN_GROUPED_V9.sh`.
- Internet hanya diperlukan untuk memasang dependency pertama kali. Data sudah lengkap.
- Otomatis: baseline, 4 studi Optuna, validasi 5 seed, fitting final, evaluasi 8 kandidat, ZIP hasil.
- **64 jam adalah total tuning saja.** Baseline, validasi, analisis snapshot dan fitting final menambah waktu; sediakan beberapa hari.
- Hasil akhir: ZIP di **campus-results/**. Bawa ZIP tersebut kembali untuk dianalisis.

V9 menguji **Morgan 1024 + 8** versus **2048 + 8**, pada XGBoost dan LightGBM.
Dataset tetap **5 sumber / 6686 molekul / 109 label**, train 5353 dan test 1333.
Semua model memakai CPU 6 thread, dijalankan berurutan. Tidak memerlukan CUDA
atau aplikasi sensor suhu. GTX 1080 Ti belum diaktifkan pada eksperimen ini.

**V7 tetap hasil utama skripsi yang dikunci. V9 adalah eksperimen tambahan.**
Panduan: [Windows](WINDOWS_GUIDE_V9.md), [Linux](LINUX_GUIDE_V9.md),
[rancangan dan batas klaim](V9_EXPERIMENT_PLAN.md).
