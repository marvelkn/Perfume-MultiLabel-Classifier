# Project Overview — hasil utama V7 dan eksplorasi V9

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

## 1. Fokus dan status hasil v6

**Judul laporan:** Komparasi XGBoost dan LightGBM dengan Optimasi Optuna untuk Klasifikasi Multi-Label Aroma Molekul dalam Konteks Parfum.

Eksperimen v6 selesai. Penelitian menguji tambahan paket lima deskriptor molekuler dan tuning pada XGBoost/LightGBM, dengan baseline internal A–D pada dataset serta pembagian yang sama. Suh menjadi acuan struktur–aroma; penelitian ini bukan reproduksi identik atau bukti mengungguli skor Suh pada dataset berbeda.

**LightGBM D dipilih melalui macro AP validasi**, bukan dari peringkat test. Inferensi Python tersedia; ONNX baru, integrasi Android Essenza, dan deployment **ditunda**. Satu molekul menghasilkan beberapa skor label aroma; tidak memprediksi campuran, intensitas sensori, atau komposisi parfum.

## 2. Dataset dan representasi

| Komponen | Nilai/kebijakan v6 |
|---|---|
| Sumber Pyrfume | GoodScents, IFRA 2019, Leffingwell, Arctander 1960, Sigma-Aldrich 2014 |
| Baris mentah | 13.337 |
| Baris ditolak / diterima | 1.560 / 11.777 |
| Konsolidasi rekaman | 5.091 baris; bukan 5.091 molekul tak berlabel yang dibuang |
| Molekul unik / label | 6.686 / 109 |
| Training / test | 5.353 / 1.333 molekul |
| Kelompok | 5.363; terbesar 38 molekul |
| Morgan | Radius 2, 1.024 bit, tanpa kiralitas |
| Lima deskriptor berurutan | MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA |
| Dimensi | A/C: 1.024; B/D: 1.029 |
| Keluaran | Skor kelas positif dan keputusan untuk masing-masing dari 109 label |

Kelompok dibentuk dari hubungan transitif struktur nonstereokimia identik atau fingerprint identik. Pemisahan ini bukan scaffold split dan tidak menjamin semua molekul yang mirip terpisah. Label dipilih dari training. Label tidak tercatat diperlakukan sebagai nol anotasi, bukan bukti sensori negatif. Relevansi sumber dengan parfum tidak membuktikan setiap molekul eksklusif digunakan dalam parfum.

## 3. Delapan kandidat dan alur evaluasi

| Kandidat | Fitur | Penyiapan parameter | Evaluasi kandidat |
|---|---|---|---|
| XGBoost A | Morgan | Baseline 100 pohon | Lima pengulangan grouped CV |
| XGBoost B | Morgan + lima deskriptor | Baseline 100 pohon | Lima pengulangan grouped CV |
| XGBoost C | Morgan | Optuna, 8 jam/studi | Lima pengulangan grouped CV |
| XGBoost D | Morgan + lima deskriptor | Optuna, 8 jam/studi | Lima pengulangan grouped CV |
| LightGBM A | Morgan | Baseline 100 pohon | Lima pengulangan grouped CV |
| LightGBM B | Morgan + lima deskriptor | Baseline 100 pohon | Lima pengulangan grouped CV |
| LightGBM C | Morgan | Optuna, 8 jam/studi | Lima pengulangan grouped CV |
| LightGBM D | Morgan + lima deskriptor | Optuna, 8 jam/studi | Lima pengulangan grouped CV |

Setiap kandidat adalah Binary Relevance dengan 109 model biner dan bobot seimbang yang dihitung dari training fold. A/B tidak melalui Optuna. Parameter baseline selain pengaturan eksplisit mengikuti versi pustaka yang didokumentasikan; 100 pohon tidak menyamakan kapasitas kedua algoritma.

Tuning memakai TPE dan grouped CV tersimpan (seed 42); empat studi berjumlah 32 jam anggaran tuning. Satu konfigurasi historis dievaluasi ulang pada setiap studi sebagai trial awal. Nilai terbaik hanya berlaku untuk ruang pencarian dan anggaran yang diuji.

Setelah parameter kandidat ditetapkan, lima pengulangan grouped CV memakai seed 7, 21, 42, 84, 100. Setiap label memperoleh lima fold pada dataset ini. Rata-rata macro AP validasi menentukan pilihan model. Prediksi out-of-fold lima pengulangan dirata-ratakan untuk memilih ambang tambahan per label yang memaksimalkan F1, grid 0,05–0,95 langkah 0,05. Jika seri, pilih terdekat 0,5, lalu lebih besar. Ini **bukan threshold holdout terpisah**. Prediksi validasi tetap terkait dengan seleksi parameter, sehingga tidak bebas bias pemilihan.

Konfigurasi, pilihan kandidat, dan ambang tambahan dibekukan sebelum fitting final/evaluasi v6. **Ambang utama tetap 0,5**, termasuk pada rilis. Seluruh delapan kandidat menjalani fitting final dan evaluasi. Cabang diagram menunjukkan kandidat terpisah; eksekusi komputasi tetap berurutan. Test telah dilihat dalam eksplorasi pendahuluan, sehingga hasil **eksploratori**, bukan evaluasi independen baru.

## 4. Tabel hasil utama v6

Seluruh angka dibaca dari `comparison_all.csv` v6 dan ditampilkan enam desimal. AP adalah Average Precision; nama kolom artefaknya `auprc`. F1 adalah **rata-rata F1 per label**, bukan harmonic mean dari macro precision dan macro recall. AP/AUROC menggunakan skor sehingga tidak berubah karena pergantian ambang keputusan.

| Kandidat | AP validasi | AP test | AUROC test | Macro-F1 test: 0,5 | Macro-F1 test: ambang OOF | Status seleksi |
|---|---:|---:|---:|---:|---:|---|
| XGBoost A | 0,248716 | 0,251634 | 0,808388 | 0,273580 | 0,273988 | — |
| XGBoost B | 0,275842 | 0,288825 | 0,821165 | 0,309399 | 0,303762 | — |
| XGBoost C | 0,271591 | 0,278934 | 0,828769 | 0,276136 | 0,299179 | — |
| XGBoost D | 0,298848 | 0,314531 | 0,838731 | 0,308414 | 0,331776 | Finalis XGBoost |
| LightGBM A | 0,250651 | 0,254001 | 0,807111 | 0,275519 | 0,270552 | — |
| LightGBM B | 0,283486 | 0,297056 | 0,825168 | 0,310900 | 0,308279 | — |
| LightGBM C | 0,273669 | 0,275895 | 0,828951 | 0,255100 | 0,298679 | — |
| LightGBM D | 0,302232 | 0,313921 | 0,843526 | 0,319645 | 0,331041 | **Terpilih** |

**LightGBM D** memiliki AP validasi tertinggi. XGBoost D mempunyai AP test sedikit lebih tinggi; hal itu tidak mengubah keputusan seleksi. Ambang OOF merupakan analisis tambahan, bukan alasan mengganti metrik utama atau kandidat setelah melihat test. Penggunaan ambang OOF juga tidak selalu meningkatkan F1.

### Metrik keputusan pada ambang utama 0,5

| Kandidat | Accuracy | Specificity | Macro precision | Macro recall |
|---|---:|---:|---:|---:|
| XGBoost A | 0,936895 | 0,950730 | 0,231785 | 0,372304 |
| XGBoost B | 0,947150 | 0,962132 | 0,285604 | 0,361971 |
| XGBoost C | 0,923777 | 0,934083 | 0,203916 | 0,480890 |
| XGBoost D | 0,935553 | 0,946463 | 0,237984 | 0,472166 |
| LightGBM A | 0,940260 | 0,954409 | 0,245131 | 0,348771 |
| LightGBM B | 0,948939 | 0,963519 | 0,288986 | 0,357453 |
| LightGBM C | 0,907410 | 0,915797 | 0,181629 | 0,523359 |
| LightGBM D | 0,934651 | 0,945199 | 0,249324 | 0,481812 |

Accuracy adalah ketepatan pasangan molekul–label, bukan persentase molekul yang seluruh labelnya benar. Prediksi semua label negatif mencapai accuracy **0,960970**, dengan recall/F1 nol. Karena itu accuracy tidak menjadi bukti keberhasilan tunggal.

## 5. Tabel analisis modifikasi

| Kontras | Yang diuji | Batas interpretasi |
|---|---|---|
| B–A | Paket lima deskriptor pada baseline | Bukan pengaruh setiap deskriptor secara terpisah |
| C–A | Tuning pada Morgan | Bukan keunggulan TPE atas random/grid search |
| D–B | Tuning pada fitur gabungan | Tidak menjamin semua metrik naik |
| D–C | Dua representasi yang masing-masing dituning | Parameter hasil pencarian berbeda; bukan efek fitur dengan semua parameter tetap |
| D–A | Gabungan fitur tambahan dan tuning | Tidak memisahkan seluruh kontribusi komponen |

Selisih berikut adalah pengurangan skor absolut, **bukan persen kenaikan relatif**. F1 dan recall memakai ambang utama 0,5.

| Algoritma | Kontras | Delta AP test | Delta macro-F1 test | Delta macro recall test |
|---|---|---:|---:|---:|
| XGBoost | B–A | +0,037191 | +0,035818 | -0,010332 |
| XGBoost | C–A | +0,027300 | +0,002556 | +0,108586 |
| XGBoost | D–B | +0,025706 | -0,000985 | +0,110195 |
| XGBoost | D–C | +0,035598 | +0,032278 | -0,008724 |
| XGBoost | D–A | +0,062897 | +0,034834 | +0,099862 |
| LightGBM | B–A | +0,043056 | +0,035381 | +0,008682 |
| LightGBM | C–A | +0,021895 | -0,020419 | +0,174587 |
| LightGBM | D–B | +0,016865 | +0,008745 | +0,124359 |
| LightGBM | D–C | +0,038026 | +0,064545 | -0,041546 |
| LightGBM | D–A | +0,059920 | +0,044126 | +0,133041 |

Paket deskriptor meningkatkan AP pada B–A untuk kedua algoritma; recall XGBoost turun. Tuning meningkatkan AP terhadap baseline representasi masing-masing, tetapi F1 LightGBM C–A dan XGBoost D–B turun. Hasil tidak membuktikan mekanisme biologis, volatilitas, manfaat setiap deskriptor, atau peningkatan semua metrik. Perbandingan versi lama mengubah ruang pencarian dan titik awal, sehingga tidak mengisolasi pengaruh lama tuning saja.

## 6. Tabel status implementasi dan batas bukti

| Bagian | Status setelah v6 | Pekerjaan/batas yang tersisa |
|---|---|---|
| Training dan evaluasi A–D | Selesai | Tetap eksploratori; tidak otomatis training ulang |
| Seleksi rilis | LightGBM D | Tetap berdasarkan AP validasi |
| Fitur/Python inference | Diverifikasi pada cakupan tersimpan | 6.686 vektor fitur; 32 contoh / 3.488 skor; bukan pengukuran akurasi baru |
| Ambang | Utama 0,5; tambahan dari OOF | Bukan holdout terpisah; kalibrasi belum diuji |
| Feature API | Menyediakan fitur RDKit berversi | Belum layanan prediksi model terpilih |
| Laporan | Hasil v6 dan audit ilmiah sudah dimasukkan | Batasan empiris tetap dinyatakan |
| Harmonisasi label | Aturan/provenance terdokumentasi | Kesetaraan semantik belum divalidasi pakar |
| Ketidakpastian | Interval deskriptif tersedia | Label/seed berkorelasi; bukan bukti ekuivalensi atau superioritas populasi |
| ONNX baru / Android / deployment | **Ditunda** | Konversi, parity, ukuran, RAM, latensi, dan uji perangkat belum dilakukan |

Rilis `releases/grouped-v6-lgbm-d/` memuat 109 model, skema fitur, urutan label, versi pustaka, checksum, dan ambang utama. ONNX lama 25 label telah dihapus. Pemrosesan SMILES tetap membutuhkan RDKit; jika memakai Feature API, seluruh alur tidak dapat disebut offline. Arsitektur mobile akhir belum dikunci. Katalog/rekomendasi parfum atau manfaat pengguna tidak dianggap tervalidasi oleh metrik model molekul.

## 7. Acuan presentasi dan pekerjaan berikutnya

Urutan slide yang konsisten: masalah struktur–aroma; posisi Suh dan kontribusi terbatas; lima sumber dan alur jumlah data; fitur; delapan kandidat; grouped CV/OOF dan batas test; hasil v6; kontras A–D; rilis Python; pekerjaan ONNX/Android yang ditunda; batasan; kesimpulan.

Jangan menggunakan keluarga aroma buatan sebagai taksonomi penelitian tanpa dasar dan pemetaan yang tervalidasi. Daftar target mengikuti manifest `release.json` dan spesifikasi fitur pada rilis, bukan keluarga label buatan. Jangan menyebut bebas 100% leakage, test baru dibuka satu kali sepanjang penelitian, Optuna optimal global, atau aplikasi sudah berhasil.

Pekerjaan berikutnya adalah penyiapan implementasi saat dimulai pengguna, dengan menjaga fitur/label/ambang dan menguji kesetaraan. Validasi ahli, evaluasi independen, kalibrasi, serta kontrol/ablasi tambahan tetap berstatus belum dilakukan; keterbatasan ini tidak hilang setelah revisi dokumen. Pseudo-labeling dan training tambahan tidak termasuk scope aktif.

## 8. Sumber angka dan riwayat pembaruan

- Angka utama: `.local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v6/perfume-campus-grouped-v6/runs/perfume-five-grouped-v6/comparison_all.csv`.
- Pendukung: `ablation_deltas.csv`, `selection_frozen.json`, `complete.json`, dan dataset manifest pada paket yang sama.
- Konsistensi Python: [verification.json](reports/backend_alignment_20260921/verification.json).
- Batas ilmiah dan bukti revisi: [rencana audit](SCIENTIFIC_AUDIT_FIX_PLAN_20260921.md), [EVIDENCE_REVIEW.md](reports/scientific_audit_20260922/EVIDENCE_REVIEW.md).
- Panduan aktif: [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md), [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md), [SAFE_EXECUTION_PLAN.md](SAFE_EXECUTION_PLAN.md).
- Sumber laporan utama tetap berada di `C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael`.
- 23 September 2026: ringkasan lama diganti; seluruh tabel hasil memakai v6 dan definisi metrik aktual; threshold OOF dijelaskan; diagram delapan cabang diselaraskan. Tidak mengubah data, model, split, hasil eksperimen, atau frontend.
