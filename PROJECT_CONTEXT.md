# Project Context — V7 dikunci; V9 eksplorasi

## Status 23 Oktober 2026: V7 utama, V9 selesai sebagai eksperimen tambahan

**V7 tetap hasil utama penelitian yang dikunci**: lima sumber Pyrfume,
6.686 molekul, 109 label, Morgan 2048 bit + delapan deskriptor.
**V9 telah selesai dan diaudit**: 1024/2048 bit + delapan deskriptor,
8 kandidat XGBoost/LightGBM, empat studi Optuna masing-masing 16 jam.
Kandidat V9 terpilih dari AP validasi adalah LightGBM 2048+8 tuned
(CV AP 0,306549; test AP 0,316227). Hasil V9 bersifat eksploratori
dan tidak otomatis mengganti V7 atau rilis aplikasi.
Bukti: [HASIL_ANALISIS_V9.md](reports/grouped_v9_analysis_20261021/HASIL_ANALISIS_V9.md).

Untuk belajar preprocessing, fitur, Optuna, training dan evaluasi, buka
[MULAI_BACA_KODE.md](MULAI_BACA_KODE.md).
ONNX dan Android untuk model baru masih ditunda; inferensi Python rilis
lama dan API fingerprint belum memakai model penelitian V7.

**Bagian bertanggal lebih lama di bawah adalah riwayat.**

## Status penelitian: V7 dikunci; V8 eksplorasi — 1 Oktober 2026

V7 telah selesai; hasilnya tersimpan di [analisis v7](reports/grouped_v7_analysis_20260928/HASIL_ANALISIS_V7.md).
Keputusan 1 Oktober 2026: **V7 dikunci sebagai hasil utama penelitian** (lima sumber Pyrfume; Morgan 2048 dan delapan deskriptor). **V8 hanya eksplorasi/audit** atas sepuluh sumber Suh; hasilnya tidak mengganti v7 dan belum menjadi dasar klaim performa skripsi.
Eksperimen lanjutan ditahan atas instruksi pengguna. Fokus sekarang adalah
**audit pre-ML V8 memakai arsip Suh (10 sumber) dan fitur v7: Morgan 2048 + 8 deskriptor = 2056**.

Ketiga berkas referensi cocok dengan checksum OSF. Checkpoint audit berisi
8700 baris dan 200 label; 100 label ringkasan acuan disimpan terpisah.
**V8 belum siap training/tuning:** ada masalah indeks split acuan, overlap
fingerprint, kebijakan multikomponen, dan batas reproduksi kurasi awal.
Fitur sengaja berbeda dari Suh; jangan mengklaim seluruh pre-ML identik.

Mulai dari [rencana dan perbedaan V8](V8_PREML_PLAN.md),
[panduan Windows](WINDOWS_GUIDE_V8.md), atau [panduan Linux](LINUX_GUIDE_V8.md).
Paket: .local-tools/campus-transfer-full-20260909/perfume-campus-v8-preml-m2048-d8.zip.
Runner hanya audit; status STOP sebelum ML merupakan hasil yang disengaja.
Data dan hasil v7 dipertahankan. ONNX, Android, serta revisi hasil LaTeX V8 belum dijalankan.

**Bagian versi sebelumnya di bawah adalah riwayat; keputusan penelitian mengikuti status V7 di atas.**

## V7 revisi 2: Morgan 2048 disiapkan - 25 September 2026

Pengguna menetapkan **Morgan 2048 bit**; paket tambahan dipilih **8 deskriptor RDKit**.
XGBoost dan LightGBM masing-masing menjalankan A-D baru: A/C Morgan saja (2048 fitur),
B/D Morgan + 8 (2056 fitur). A/B baseline 100; C/D empat studi Optuna 8 jam/studi
(**32 jam tuning**, belum termasuk validasi dan final fit). Eksekusi di **PC kampus/PC lain**.

Delapan jenis deskriptor mengikuti acuan Suh; pemilihan ini bukan bukti paket terbaik.
Gabungan fitur dan ukuran 2048 adalah eksperimen proyek, bukan resep gabungan asli Suh.
Data 6686 molekul, 109 label, split 5353/1333 dan kelompok lama tetap.
Fitur baru terpisah di data/builds/perfume-m2048-d8-v7; hasil v6 dipertahankan.

**Training penelitian v7 belum dijalankan.** Rilis aktif tetap LightGBM D v6.
Seleksi dari mean macro AP repeated grouped CV; ambang tambahan dari prediksi OOF.
B-A menguji tambahan 8 pada baseline tetap; D-C membandingkan hasil tuning masing-masing.
V6-v7 merupakan sejarah eksperimen; B/D mengubah panjang Morgan dan paket deskriptor.
Test pernah dilihat, sehingga hasil tetap eksploratori.

Paket terbaru: **perfume-campus-grouped-v7-m2048-d8.zip**; ekstrak ke folder baru.
Run ID: perfume-five-grouped-v7-m2048-d8.
Rancangan/paket v7 lama 1024 (B/D/E/F, 5 vs 8) digantikan revisi ini dan hanya menjadi riwayat.
Panduan: [Windows](WINDOWS_GUIDE_V7.md) / [Linux](LINUX_GUIDE_V7.md).
Rencana lengkap: [DESCRIPTOR_COMPARISON_V7_PLAN.md](DESCRIPTOR_COMPARISON_V7_PLAN.md).

**Setelah hasil v7 diaudit, agenda berikutnya merombak laporan sesuai metode/hasil terbaru.**
Catatan permanen: [AFTER_V7_REPORT_UPDATE.md](AFTER_V7_REPORT_UPDATE.md).
LaTeX belum direvisi untuk hasil yang belum tersedia. ONNX/Android tetap ditunda.

## Hasil penelitian v6 yang sudah tersedia

- **Eksperimen v6 selesai.** Delapan kandidat XGBoost/LightGBM A–D telah dievaluasi; model terpilih adalah **LightGBM D**, berdasarkan macro AP validasi berulang **0,302232**.
- Lima sumber Pyrfume: GoodScents, IFRA 2019, Leffingwell, Arctander 1960, Sigma-Aldrich 2014. Dataset: **6.686 molekul**, **109 label**, **5.353 train / 1.333 test**, **5.363 kelompok**.
- Masukan pengguna berupa SMILES **satu molekul terhubung**. Model menerima fitur numerik, lalu mengeluarkan skor dan keputusan per label aroma.
- RDKit menghitung Morgan radius 2, 1.024 bit, tanpa kiralitas. A/C memakai Morgan; B/D menambahkan MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA sehingga menjadi 1.029 fitur.
- Binary Relevance memakai satu pengklasifikasi biner per label. A/B adalah baseline 100 pohon; C/D memakai Optuna TPE. Bobot seimbang dihitung dari bagian training masing-masing fold.
- Empat studi C/D mendapat anggaran **8 jam per studi / 32 jam total tuning**. Validasi berulang dan fitting final dihitung terpisah. Training v7 disiapkan sesuai revisi di atas.

## Validasi, ambang, dan batas klaim

- Group split mencegah anggota kelompok struktur nonstereokimia identik atau fingerprint identik tersebar antarpartisi. Ini bukan scaffold split dan bukan jaminan bebas seluruh jenis bias/kebocoran.
- Label dipilih menggunakan dukungan training. Tuning memakai grouped CV; seleksi delapan kandidat memakai **lima pengulangan grouped CV**, seed 7, 21, 42, 84, 100, masing-masing lima fold per label pada dataset ini.
- **Ambang utama hasil dan rilis adalah 0,5.** Ambang tambahan per label dipilih untuk F1 dari **rata-rata prediksi out-of-fold lima pengulangan**, grid 0,05–0,95 langkah 0,05. Bukan threshold holdout terpisah dan bukan hasil tuning pada test.
- Test sudah diperiksa pada eksperimen pendahuluan. Hasil tetap **eksploratori**; repeated CV bukan nested CV. Penguncian v6 tidak menghapus riwayat tersebut.
- Tidak ada klaim mengungguli Suh secara langsung, semua molekul eksklusif parfum, atau skor sebagai intensitas/komposisi/probabilitas yang sudah terkalibrasi.

## Backend dan aplikasi

- Rilis aktif: `releases/grouped-v6-lgbm-d/`, berisi 109 model dengan skema 1.029 fitur. Inferensi Python lokal menerima SMILES dan menghasilkan skor serta label.
- Feature API `/fingerprint` menyediakan fitur RDKit dengan skema berversi; API ini belum menjadi layanan prediksi model terpilih.
- **Implementasi model baru melalui ONNX, integrasi Android Essenza, dan deployment ditunda dan belum terverifikasi.** ONNX lama 25 label sudah dihapus.
- Arsitektur akhir mobile belum dikunci. Jika ekstraksi SMILES–fitur memakai server, alur tetap memerlukan jaringan walaupun inferensi ONNX nantinya lokal. Jangan menjanjikan seluruh alur offline.
- Frontend hanya referensi read-only pada sesi backend/laporan. Katalog atau taksonomi aplikasi bukan pengganti urutan 109 label model.

## Laporan dan pekerjaan tersisa

- Laporan sudah menggunakan hasil **v6**, diperbaiki melalui audit ilmiah 22 September, dan diagramnya diselaraskan dengan delapan kandidat pada 23 September 2026.
- Revisi narasi tidak berarti validasi pakar harmonisasi, evaluasi independen, kalibrasi, atau analisis ketidakpastian baru sudah dilakukan. Detail ada pada rencana audit.
- Training ulang, pseudo-labeling, dan inferensi massal data yang ditolak tidak dijalankan otomatis.

## Sumber kerja

- [Ringkasan dan tabel v6](PROJECT_OVERVIEW.md)
- [Aturan/status aktif](SAFE_EXECUTION_PLAN.md), [status implementasi](IMPLEMENTATION_STATUS.md)
- [Panduan inferensi](INFERENCE_GUIDE.md)
- [Rencana audit dan status F01–F18](SCIENTIFIC_AUDIT_FIX_PLAN_20260921.md)
- [Bukti kesetaraan backend](reports/backend_alignment_20260921/verification.json)
- [Bukti audit ilmiah](reports/scientific_audit_20260922/EVIDENCE_REVIEW.md)
