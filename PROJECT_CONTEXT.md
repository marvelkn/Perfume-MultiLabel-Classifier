# Konteks proyek — 26 Oktober 2026

Jalur penelitian aktif ada di `KODE_ML/research/`; panduan belajar di `KODE_ML/`. Lima sumber Pyrfume, 6.686 molekul, 109 label. A/C Morgan2048 radius2 tanpa chirality; B/D ditambah delapan deskriptor. Dua algoritma XGBoost/LightGBM, A–D, empat studi TPE masing-masing8jam, repeated grouped CV lima seed dan lima fold per label.

Seleksi dari mean macro AP validasi. Ambang utama0,5; ambang tambahan dari rata-rata prediksi OOF lima seed pada train. Hasil utama selesai/dikunci sebagai V7; V9 tambahan. LightGBM D terpilih dengan mean CV AP0,299731 dan test AP0,310531. Test pernah dilihat; CV bukan nested CV.

Konfigurasi aktif `KODE_ML/research/config/experiment.json`. Target/grup/split beku di `data/builds/perfume-five-grouped-v2`; fitur utama di `data/builds/perfume-m2048-d8-v7/features.npz`. Loader tidak memakai array fitur1024+5 lama. Grup tetap historis: struktur tanpa stereo atau Morgan1024 identik. Verifikasi baru tidak mengklaim membentuk grup dari2048.

Source lama diarsipkan, hasil lama tidak dimodifikasi. Source baru tidak boleh resume run lama. Implementasi ONNX/Android model baru ditunda. API/prototipe lama dan src tetap di lokasi semula untuk kompatibilitas; keduanya belum memakai model utama.

Lihat README.md, KODE_ML/README.md, dan reports/research_cleanup_20261025/REFACTOR_REPORT.md. Perapian ini bukan eksperimen baru; tidak menjalankan training, tuning atau unduhan dataset.

Delapan notebook 01–08 dalam KODE_ML menjadi panduan utama. Kode yang ditampilkan berasal dari fungsi lokal yang diimpor. Audit menemukan 31 alias, 184 istilah IFRA, 109 target terpilih, dan 498 istilah/6.494 kemunculan yang gagal dipetakan. Makna semua istilah belum tervalidasi; aturan label tetap dipertahankan. Lihat KODE_ML/HASIL_VERIFIKASI.md.
