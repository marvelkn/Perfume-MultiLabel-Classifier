# Analisis hasil v7 — 28 September 2026

**Hasil lengkap dan angka utama terverifikasi. LightGBM D menjadi kandidat pilihan sesuai aturan macro AP validasi.**
Temuan utamanya: tambahan delapan deskriptor membantu kedua algoritma secara konsisten pada lima seed; Optuna menaikkan AP, tetapi tidak semua metrik ikut naik.

## Data dan kelengkapan

- Sumber run: [perfume-five-grouped-v7-m2048-d8](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/.local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v7-m2048-d8/perfume-campus-grouped-v7-m2048-d8/runs/perfume-five-grouped-v7-m2048-d8/RESULTS_SUMMARY.md>).
- 6686 molekul; 5353 training, 1333 test; 109 label. Morgan 2048; A/C 2048 fitur, B/D 2056 fitur.
- Semua A-D XGBoost/LightGBM selesai: 8 kandidat × 109 model per label = 872 model.
- 1199 berkas pada completion manifest cocok dengan checksum; source, data, identitas fitur dan freeze ikut diperiksa.
- Seluruh label memakai lima fold pada kelima seed. Tidak ditemukan kelompok yang melintasi train-test atau train-validasi; fold sama pada seluruh kandidat.
- Skor test dihitung ulang dari prediksi mentah untuk seluruh kandidat dan kedua mode ambang. AP dari OOF mentah kedua finalis dihitung ulang; rerata/SD CV semua kandidat juga cocok.
- Pilihan model dibekukan sebelum final fit/evaluasi. Pemeriksaan ini tidak memulai training, mengubah pipeline, mengganti rilis aktif, atau mengedit LaTeX.
- Waktu dari event awal sampai complete: 47.78 jam, sekitar 47 jam 47 menit; 32 jam merupakan budget tuning.

## Hasil utama v7

A: Morgan; B: Morgan+8; C: Morgan+Optuna; D: Morgan+8+Optuna. AP merupakan Average Precision, bukan persentase prediksi benar.

| Kandidat | AP validasi ± SD | AP test | AUROC test | Macro F1, 0,5 | Macro F1, ambang OOF |
|---|---:|---:|---:|---:|---:|
| xgb_A | 0.249589 ± 0.002441 | 0.249043 | 0.809232 | 0.274899 | 0.275833 |
| xgb_B | 0.277412 ± 0.001425 | 0.288125 | 0.822636 | 0.311835 | 0.309717 |
| xgb_C | 0.267049 ± 0.001463 | 0.273029 | 0.825034 | 0.270369 | 0.293362 |
| xgb_D | 0.296959 ± 0.001128 | 0.308827 | 0.838546 | 0.305902 | 0.327241 |
| lgbm_A | 0.251681 ± 0.003295 | 0.252837 | 0.809417 | 0.277184 | 0.276695 |
| lgbm_B | 0.283211 ± 0.001496 | 0.296380 | 0.825813 | 0.320721 | 0.315920 |
| lgbm_C | 0.269114 ± 0.002343 | 0.276461 | 0.827912 | 0.252220 | 0.300816 |
| lgbm_D | 0.299731 ± 0.001360 | 0.310531 | 0.842985 | 0.293156 | 0.330419 |

LightGBM D unggul atas XGBoost D pada kelima seed, tetapi selisih mean AP validasi hanya 0,002772 dan AP test 0,001704. Tulis 'unggul tipis dalam eksperimen ini'; belum ada bukti signifikansi atau keunggulan universal.

## Pengaruh fitur dan tuning

| Perbandingan | XGBoost: delta AP validasi | LightGBM: delta AP validasi |
|---|---:|---:|
| B-A | +0.027823 (5/5 seed positif) | +0.031530 (5/5 seed positif) |
| D-C | +0.029909 (5/5 seed positif) | +0.030617 (5/5 seed positif) |
| C-A | +0.017460 (5/5 seed positif) | +0.017433 (5/5 seed positif) |
| D-B | +0.019547 (5/5 seed positif) | +0.016520 (5/5 seed positif) |

B-A adalah perbandingan paket delapan deskriptor pada baseline dengan parameter sama. D-C menguji representasi setelah tuning masing-masing; parameter terbaik dapat berbeda.
C-A dan D-B menunjukkan perbaikan AP setelah tuning. Optuna dioptimalkan untuk AP, jadi kenaikan AP tidak menjamin F1, precision, dan accuracy pada ambang 0,5 ikut meningkat.

## Kelemahan praktis pemenang

| Metrik LightGBM D di test | Ambang utama 0,5 | Ambang tambahan dari OOF |
|---|---:|---:|
| Accuracy rata-rata per label | 0.923660 | 0.945002 |
| AUROC | 0.842985 | 0.842985 |
| AP | 0.310531 | 0.310531 |
| Specificity | 0.932611 | 0.957765 |
| Precision rata-rata per label | 0.213862 | 0.303265 |
| Recall rata-rata per label | 0.515089 | 0.411890 |
| Macro F1 | 0.293156 | 0.330419 |
| Rata-rata label yang diprediksi per molekul | 9.355589 | 6.372093 |

Pada ambang 0,5, model mengeluarkan rata-rata 9,36 label per molekul, sedangkan anotasi memuat rata-rata 4,25 label. Banyak prediksi tidak cocok dengan anotasi; pada data ini label yang tidak tercatat diperlakukan sebagai negatif, sehingga itu juga merupakan keterbatasan evaluasi.
Macro F1 LightGBM B pada 0,5 adalah 0,320721, lebih tinggi dari D sebesar 0,293156. D dipilih karena aturan awalnya memakai AP validasi, bukan karena unggul di setiap metrik.
Ambang OOF menaikkan F1 D ke 0,330419 dan precision ke 0,303265, dengan recall turun ke 0,411890. Ini tradeoff yang harus dilaporkan; hasil ambang tambahan tidak menggantikan hasil utama 0,5 diam-diam.
Model yang selalu menjawab semua label 'tidak' menghasilkan accuracy 0.960970, tetapi recall dan F1 nol. Karena itu accuracy 0,923660 bukan bukti akurasi praktis tinggi.

## Contoh hasil per label

| Label | Positif di test | AP | F1 per label, ambang 0,5 |
|---|---:|---:|---:|
| fruity | 511 | 0.794976 | 0.714154 |
| floral | 374 | 0.650814 | 0.633371 |
| sweet | 327 | 0.436889 | 0.459584 |
| woody | 240 | 0.610696 | 0.574545 |
| vanilla | 28 | 0.307348 | 0.265306 |
| musk like | 22 | 0.517758 | 0.464286 |
| tea | 13 | 0.029003 | 0.040000 |
| narcissus | 2 | 0.013459 | 0.000000 |
| popcorn | 3 | 0.916667 | 0.461538 |

Contoh ini dipilih untuk menampilkan aroma umum dan variasi performa, bukan hanya skor tertinggi. Popcorn AP tinggi tetapi hanya memiliki 3 contoh positif; narcissus hanya 2. Skor label dengan dukungan kecil mudah berubah.

## Status Optuna

| Studi | Trial selesai | Trial dihentikan | Trial terbaik (nomor mulai 0) | Trial selesai setelah terbaik |
|---|---:|---:|---:|---:|
| xgb_C | 12 | 1 | 2 | 9 |
| xgb_D | 12 | 1 | 2 | 9 |
| lgbm_C | 104 | 1 | 56 | 47 |
| lgbm_D | 74 | 1 | 73 | 0 |

Keempat trial berstatus FAIL masing-masing dihentikan karena budget 8 jam habis, bukan exception yang membuat run gagal. Jumlah dan status trial cocok antara CSV dan database SQLite.
Budget waktu kedua algoritma setara, tetapi jumlah kombinasi yang sempat dicoba berbeda. XGBoost baru menyelesaikan 12 trial/studi. Protokol memakai 10 startup trials, sehingga pencarian adaptif setelah startup masih terbatas.
LightGBM D menemukan skor terbaik pada trial terakhir yang selesai. Ini tidak membuktikan proses sudah mencapai titik terbaik; juga tidak menjamin penambahan waktu pasti memperbaiki hasil.
XGBoost C/D kebetulan memilih parameter sama (trial 2). Resep awal dan sampler seed sama, sehingga hal ini mungkin terjadi; representasi dan skor kedua studi tetap berbeda.

## Penilaian dan langkah berikutnya

**Layak menjadi hasil eksperimen skripsi, dengan batas klaim yang jelas.** Ada bukti internal yang konsisten bahwa paket delapan deskriptor membantu Morgan 2048 dan tuning membantu AP. Kualitas semua label belum merata, dan hasil belum layak disebut optimum global atau siap prediksi sensori yang pasti benar.

1. Bekukan hasil v7 ini dan gunakan seluruh delapan kandidat dalam Bab 4. Nyatakan alasan pemilihan LightGBM D dari AP validasi.
2. Laporkan tradeoff F1/precision/recall dan hasil ambang OOF sebagai analisis tambahan. Jangan menyamakan AP 0,3105 dengan 31,05% akurasi.
3. Audit ini tidak menunjukkan kegagalan teknis yang mewajibkan training ulang. Eksperimen tambahan bersifat opsional; jika dilakukan, tetapkan pertanyaan dan budget sebelum menjalankan, bukan mengulang sampai test naik.
4. Untuk pemakaian aplikasi, lakukan pekerjaan integrasi/inferensi dan evaluasi perilaku prediksi sebagai tahap terpisah. Jangan menganggap rilis v6 otomatis sudah diganti oleh artefak v7.

## Batas ilmiah dan riwayat

Test historis telah dilihat pada iterasi sebelumnya. Lima seed/fold berkorelasi dan repeated CV ini bukan nested CV; hasil tetap eksploratori, bukan evaluasi konfirmatori independen.
Lima sumber relevan dengan parfum tetapi tidak menjamin semua molekul eksklusif parfum. Tidak adanya anotasi bukan bukti sensori negatif; skor model belum dikalibrasi sebagai probabilitas empiris.
V7 membandingkan Morgan2048 dengan/tanpa8, bukan lima lawan delapan pada panjang Morgan yang sama. Angka publikasi Suh tidak apple-to-apple karena data, label dan pembagiannya berbeda.

Riwayat v6 bukan tabel hasil utama: LightGBM D v6 AP validasi 0,302232 dan AP test 0,313921, sementara v7 0,299731 dan 0,310531. Jadi perubahan ke2048+8 belum menunjukkan kenaikan atas konfigurasi lama. Dua komponen berubah sekaligus dan tuning terpisah, sehingga penyebab perbedaannya tidak dapat dipisahkan.

## Bukti yang dapat diperiksa

- audit_v7.py: skrip pembacaan artefak dan penghitungan ulang.
- audit.json: checksum, kelengkapan, pemisahan kelompok, skor CV dan status Optuna.
- recomputed_metrics.csv: metrik test yang dihitung ulang dari prediksi tersimpan.
- selected_model_per_label.csv: AP, precision, recall, F1 dan dukungan setiap label.
- [Tabel hasil asli](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/.local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v7-m2048-d8/perfume-campus-grouped-v7-m2048-d8/runs/perfume-five-grouped-v7-m2048-d8/comparison_all.csv>).
- [Protokol yang benar-benar dipakai](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/.local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v7-m2048-d8/perfume-campus-grouped-v7-m2048-d8/alignment/protocol_v7.json>).
