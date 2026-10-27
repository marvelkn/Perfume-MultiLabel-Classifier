# Arsip source sebelum perapian

Folder `implementation-before-cleanup-20261025` menyimpan alignment, eksplorasi V8, notebook duplikat, konfigurasi dan panduan/launcher eksperimen versi lama. Alamat relatif asli dipertahankan. Ini riwayat; jalur penelitian aktif ada di ../research dan ../KODE_ML.

Data mentah, dataset beku, model, release dan laporan bukti tidak dipindah/dihapus. Link source pada laporan historis dapat ditelusuri dengan menambahkan prefix folder arsip ini.

## Pemulihan

Lihat ARCHIVE_MANIFEST.json untuk daftar asal dan SHA-256. Backup ZIP lengkap source sebelum edit, termasuk notebook yang sudah memiliki perubahan lokal, ada di:

C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\.local-backups\research-cleanup-20261025-221854

Ekstrak source-before.zip ke folder pemulihan BARU. Jangan timpa source aktif. Backup mempertahankan layout asli, tetapi tidak menggandakan dataset/model. Untuk menjalankan audit lama, sediakan salinan artefak yang dirujuk. Jangan menjalankan launcher training hanya untuk membaca hasil.

Source sebelum perapian tidak otomatis identik dengan source PC kampus saat training. Source di paket kampus dan identity.json hasil tetap menjadi bukti utama. Arsip ini mencatat kondisi lokal tepat sebelum refactor.

## Kompatibilitas aplikasi lama

src, API, tests lama, Dockerfile, Procfile dan config aplikasi tetap di lokasi semula karena masih saling dipakai. Pemindahan luasnya ditolak review otomatis karena risiko memutus caller. Jalur research tidak mengimpor src. Kontrak aplikasi tidak diganti menjadi 2048+8. Integrasi ONNX/Android model utama masih ditunda.
