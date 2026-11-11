# Konteks proyek — 7 November 2026

Jalur penelitian aktif ada di `KODE_ML/03_Kode/`; panduan belajar di `KODE_ML/`. Lima sumber Pyrfume, 6.686 molekul, 109 label. A/C Morgan2048 radius2 tanpa chirality; B/D ditambah delapan deskriptor. Dua algoritma XGBoost/LightGBM, A–D, empat studi TPE masing-masing8jam, repeated grouped CV lima seed dan lima fold per label.

Seleksi dari mean macro AP validasi. Ambang utama0,5; ambang tambahan dari rata-rata prediksi OOF lima seed pada train. Hasil utama selesai/dikunci sebagai V7; V9 tambahan. LightGBM D terpilih dengan mean CV AP0,299731 dan test AP0,310531. Test pernah dilihat; CV bukan nested CV.

Konfigurasi aktif `KODE_ML/03_Kode/config/experiment.json`. Target/grup/split beku di `data/builds/perfume-five-grouped-v2`; fitur utama di `data/builds/perfume-m2048-d8-v7/features.npz`. Loader tidak memakai array fitur1024+5 lama. Grup tetap historis: struktur tanpa stereo atau Morgan1024 identik. Verifikasi baru tidak mengklaim membentuk grup dari2048.

Source dan hasil penelitian lama tetap dipertahankan; source baru tidak boleh resume run lama. Aplikasi aktif memakai **RDKit + model LightGBM D asli (joblib) pada komputer lokal**, diakses Android melalui USB. Implementasi dan API aktif berada di repo `Essenza_Frontend/feature_api/`, dengan entrypoint `feature_api.app:app` serta endpoint `/predict`. Android menampilkan skor dan menyimpan Riwayat/Koleksi; tidak menjalankan estimator. `src/`, API root backend, dan rilis v6 adalah kompatibilitas lama 1.029 fitur, bukan model/API aktif v7. Tidak ada training ulang atau perubahan angka evaluasi untuk migrasi runtime aplikasi.

Lihat README.md, KODE_ML/README.md, dan reports/research_cleanup_20261025/REFACTOR_REPORT.md. Perapian ini bukan eksperimen baru; tidak menjalankan training, tuning atau unduhan dataset.

Delapan notebook 01–08 dalam KODE_ML menjadi panduan utama. Kode yang ditampilkan berasal dari fungsi lokal yang diimpor. Audit menemukan 31 alias, 184 istilah IFRA, 109 target terpilih, dan 498 istilah/6.494 kemunculan yang gagal dipetakan. Makna semua istilah belum tervalidasi; aturan label tetap dipertahankan. Lihat KODE_ML/HASIL_VERIFIKASI.md.

Setup aplikasi mengikuti `../Essenza_Frontend/docs/LOCAL_DEVELOPMENT.md`; dependensi penelitian mengikuti `requirements_research.txt`. `requirements_lock.txt`, protokol, dan hasil bertanggal adalah provenance historis, bukan daftar instalasi API baru. Pengujian terkini harus mengacu pada bukti migrasi aplikasi; riwayat pengujian arsitektur lama tidak dianggap otomatis berlaku.
