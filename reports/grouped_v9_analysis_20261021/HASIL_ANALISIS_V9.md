# Analisis hasil V9 — 21 Oktober 2026

**Kesimpulan: eksperimen selesai dan memberi peningkatan AP, tetapi kenaikannya kecil dan tidak mencakup semua metrik. Rekomendasi: hentikan tuning rutin; simpan V9 sebagai eksperimen tambahan. V7 tetap hasil utama yang dikunci.**

V9 terpilih: **LightGBM, Morgan 2048 bit + 8 deskriptor, tuned 16 jam**, berdasarkan rata-rata macro-AP validasi. Ini adalah pilihan terbaik di antara kandidat V9 menurut aturan yang ditetapkan, bukan bukti konfigurasi terbaik secara global.

## 1. Sumber dan ruang lingkup

Hasil dibaca langsung dari folder hasil kampus yang berada satu tingkat di dalam folder kiriman:

`.local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v9-bits-d8/perfume-campus-grouped-v9-bits-d8/runs/perfume-five-grouped-v9-bits-d8`

Sumber utama: `comparison_all.csv`, `selection_frozen.json`, `completion_manifest.json`, prediksi NPZ, hasil CV per fold/seed, empat SQLite Optuna, dan `budget_comparison.json`. Perbandingan V7 menggunakan hasil asli dalam paket V7, bukan angka ingatan.

Data tetap **6.686 molekul, 109 label; 5.353 train dan 1.333 test**, dari lima sumber Pyrfume. Semua kandidat V9 memakai delapan deskriptor. Yang diuji adalah panjang Morgan (1024/2048), baseline/tuned, dan anggaran tuning (8/16 jam).

Analisis ini tidak menjalankan ulang training, mengubah pipeline, mengganti model rilis, atau mengubah laporan LaTeX.

## 2. Hasil delapan kandidat

Semua AP, AUROC, precision, recall, dan F1 pada tabel adalah macro (rata-rata per label). AP adalah Average Precision, bukan accuracy dan bukan trapezoidal PR-AUC. Skor lebih tinggi lebih baik.

| Model | Morgan | Konfigurasi | CV AP | SD antar-seed | Test AP | Test AUROC | F1 ambang 0,5 | F1 ambang OOF |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 1024 | Baseline | 0,277126 | 0,002111 | 0,286298 | 0,819090 | 0,309725 | 0,307307 |
| XGBoost | 1024 | Tuned | 0,305089 | 0,001156 | 0,314301 | 0,841854 | 0,304766 | 0,325220 |
| XGBoost | 2048 | Baseline | 0,277412 | 0,001425 | 0,288125 | 0,822636 | 0,311835 | 0,309717 |
| XGBoost | 2048 | Tuned | 0,303381 | 0,000943 | 0,315063 | 0,844813 | 0,307948 | **0,341471** |
| LightGBM | 1024 | Baseline | 0,284469 | 0,001395 | 0,298612 | 0,824708 | 0,308498 | 0,308850 |
| LightGBM | 1024 | Tuned | 0,303769 | 0,001013 | 0,314836 | **0,846001** | 0,301772 | 0,331212 |
| LightGBM | 2048 | Baseline | 0,283211 | 0,001496 | 0,296380 | 0,825813 | **0,320721** | 0,315920 |
| **LightGBM** | **2048** | **Tuned — terpilih** | **0,306549** | 0,001875 | **0,316227** | 0,844797 | 0,317183 | 0,327098 |

Ambang utama adalah 0,5. Ambang OOF dipilih hanya dari prediksi validasi pada data train dan dilaporkan sebagai analisis tambahan.

**Tidak ada satu kandidat yang menang di semua metrik.** XGB 2048 tuned memiliki F1 OOF tertinggi di test; LGBM 2048 baseline memiliki F1 utama tertinggi. Keduanya tidak mengubah pemenang protokol, karena kriteria pemilihan yang disepakati adalah CV AP. Memilih ulang pemenang hanya setelah melihat metrik test akan mengubah aturan eksperimen.

## 3. Jawaban tentang 1024 versus 2048

- Pada baseline XGBoost, CV AP 2048 lebih tinggi hanya **0,000286** daripada 1024.
- Pada baseline LightGBM, 1024 lebih tinggi **0,001258**.
- Setelah tuning dengan waktu yang sama, **XGBoost 1024** lebih tinggi **0,001707 CV AP** daripada 2048.
- Setelah tuning dengan waktu yang sama, **LightGBM 2048** lebih tinggi **0,002781 CV AP** daripada 1024.
- Kedua arah perbedaan tuned tersebut konsisten pada lima seed; ini tetap deskripsi pada data yang sama, bukan bukti signifikansi statistik.

**Kesimpulan: tidak ada alasan untuk pindah semua model ke 1024.** Efek jumlah bit bergantung pada algoritma dan hasil tuning. Perbandingan baseline mengendalikan parameter; perbandingan tuned membandingkan dua proses pencarian dengan anggaran waktu sama, sehingga perbedaannya tidak bisa seluruhnya diklaim berasal dari panjang bit.

## 4. Dampak tuning dan tambahan waktu

Dalam V9, tuning meningkatkan CV AP dan test AP pada keempat pasangan baseline–tuned. Namun, **F1 pada ambang 0,5 sedikit turun di keempat pasangan**. F1 dengan ambang OOF meningkat di keempat pasangan. Ini sesuai dengan tujuan tuning yang mengoptimalkan AP, bukan langsung mengoptimalkan F1 pada 0,5.

| Studi | Trial selesai sampai 8 jam | Trial selesai sampai 16 jam | CV AP resep 8 jam | CV AP resep 16 jam | Perubahan |
|---|---:|---:|---:|---:|---:|
| XGB 1024+8 | 18 | 40 | 0,298300 | 0,305089 | +0,006789 |
| XGB 2048+8 | 11 | 19 | 0,296959 | 0,303381 | +0,006423 |
| LGBM 1024+8 | 110 | 170 | 0,300791 | 0,303769 | +0,002978 |
| LGBM 2048+8 | 73 | 148 | 0,300495 | 0,306549 | +0,006054 |

Resep terbaik pada masing-masing batas waktu dievaluasi menggunakan repeated grouped CV yang sama. Snapshot 8 jam tidak dievaluasi di test dan tidak masuk pemilihan delapan kandidat utama.

- **377 trial selesai**, ditambah empat trial FAIL: masing-masing studi berhenti di tengah satu trial karena `budget_exhausted`.
- Ledger mencatat **16 jam per studi, total 64 jam tuning**.
- Rentang waktu seluruh run sekitar **87,04 jam**, termasuk validasi, fitting final, dan kemungkinan jeda/resume.
- Tambahan 8 jam meningkatkan CV AP keempat studi, dengan arah positif pada seluruh lima seed. Kenaikannya sekitar **0,0030–0,0068 AP**, bukan lonjakan besar.
- Fairness di sini adalah **waktu komputasi sama**, bukan jumlah trial sama. Kedua algoritma tetap memakai CPU enam thread.

**Belum boleh bilang Optuna sudah menemukan optimum global.** XGB 2048 hanya menyelesaikan 19 trial; trial terbaiknya adalah trial terakhir selesai, dengan 1.600 pohon dan kedalaman 8 yang menyentuh batas pencarian. Pada LGBM 2048, ada 13 trial selesai setelah trial terbaik. Informasi ini menunjukkan peluang eksplorasi masih ada, tetapi tidak membuktikan tambahan waktu pasti memperbaiki generalisasi. Dokumentasi Optuna mendefinisikan hasil terbaik sebagai hasil terbaik dalam studi tersebut, bukan seluruh kemungkinan konfigurasi. [Optuna 4.5](https://optuna.readthedocs.io/en/v4.5.0/reference/generated/optuna.study.Study.html)

## 5. Perbandingan model terpilih V7 dan V9

Keduanya menggunakan LightGBM dengan Morgan 2048+8, data dan test yang sama.

| Metrik | V7 LGBM D | V9 LGBM 2048 tuned | Perubahan |
|---|---:|---:|---:|
| CV macro-AP | 0,299731 | 0,306549 | +0,006818 |
| Test macro-AP | 0,310531 | 0,316227 | **+0,005696** |
| Test macro-AUROC | 0,842985 | 0,844797 | +0,001812 |
| Test macro-F1, ambang 0,5 | 0,293156 | 0,317183 | **+0,024027** |
| Test macro-F1, ambang OOF | 0,330419 | 0,327098 | **−0,003322** |

AP test bertambah sekitar **0,57 poin persentase** pada skala 0–100, atau 1,83% secara relatif. Jangan menulis “accuracy meningkat 57%”.

Baseline 2048 V9 kedua algoritma menghasilkan probabilitas test persis sama dengan skenario B V7. Ini mendukung bahwa setup dasar tetap konsisten. Namun, untuk menyimpulkan pengaruh waktu tuning secara lebih terkontrol, gunakan snapshot 8/16 jam dalam V9, bukan semata-mata membandingkan versi historis.

Sebanyak **71 dari 109 label** meningkat AP-nya dibanding model terpilih V7, sedangkan **38 menurun**. V9 membaik secara rata-rata, tidak merata di semua label.

## 6. Kondisi prediksi yang masih membatasi

Untuk LGBM 2048 tuned:

| Ukuran | Ambang 0,5 | Ambang OOF |
|---|---:|---:|
| Macro-precision | 0,248767 | 0,309792 |
| Macro-recall | 0,471479 | 0,403730 |
| Macro-F1 | 0,317183 | 0,327098 |
| Rata-rata label diprediksi per molekul | 7,720 | 6,152 |

Rata-rata label yang tercatat pada test adalah **4,254 per molekul**. Model masih mengeluarkan lebih banyak label daripada anotasi. Perlu diingat, label tidak tercatat belum tentu benar-benar tidak beraroma; evaluasi ini membandingkan model dengan anotasi yang tersedia.

Macro-precision adalah rata-rata precision per label, bukan proporsi semua prediksi positif yang benar jika digabung. Jangan menyamakan 0,248767 dengan “tepat 24,88% dari seluruh rekomendasi benar”.

| Label | Positif pada test | AP | F1 pada 0,5 |
|---|---:|---:|---:|
| Fruity | 511 | 0,799395 | 0,712406 |
| Floral | 374 | 0,667276 | 0,636051 |
| Green | 404 | 0,629549 | 0,601504 |
| Woody | 240 | 0,614025 | 0,583012 |
| Sweet | 327 | 0,447189 | 0,486228 |
| Musk like | 22 | 0,548144 | 0,490566 |
| Vanilla | 28 | 0,329810 | 0,320000 |
| Tea | 13 | 0,025741 | 0 |
| Narcissus | 2 | 0,011743 | 0 |

**40 label** memiliki kurang dari 20 positif pada test; **13 label** kurang dari 10. F1 lima label bernilai nol pada ambang 0,5, dan empat label tetap nol dengan ambang OOF. Skor label langka mudah berubah karena jumlah contoh kecil.

Accuracy model terpilih sekitar **93,61%**, tetapi prediktor semua-negatif mencapai **96,10%** pada target yang sama dan tidak mendeteksi label positif apa pun. Karena itu, accuracy tidak boleh menjadi satu-satunya dasar penilaian.

## 7. Batas kesimpulan ilmiah

1. Test historis sudah berkali-kali diperiksa. Walaupun pemilihan kandidat V9 dilakukan sebelum evaluasi test V9, keputusan eksplorasi sebelumnya sudah dipengaruhi hasil lama. V9 bukan konfirmasi independen pada test baru.
2. Repeated CV menggunakan data pengembangan yang sama dengan tuning dan **bukan nested CV**. Nilai CV digunakan untuk pemilihan dan analisis kestabilan; tidak diperlakukan sebagai estimasi bebas bias. [Dokumentasi scikit-learn](https://scikit-learn.org/1.6/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
3. Lima seed saling berkaitan karena menggunakan molekul yang sama. SD antarseed bukan confidence interval; hasil konsisten di lima seed tidak otomatis “signifikan”.
4. Grup berdasarkan kesamaan struktur/fitur yang telah dibekukan bukan pemisahan scaffold. Grup sama tidak melintasi train/test atau train/validasi, tetapi ini tidak menjamin semua molekul mirip sudah terpisah.
5. Nol pada anotasi, ketidakseragaman antarsumber, dan label langka tetap menjadi keterbatasan data. Tuning tidak memperbaiki kualitas label sumber.
6. Hasil tidak membuktikan keunggulan apple-to-apple atas Suh; data, target dan protokol tidak identik.
7. Analisis tidak menguji generalisasi pada sumber baru, kalibrasi skor, keamanan penggunaan bahan, kinerja Android/ONNX, atau kelayakan deployment.

## 8. Rekomendasi keputusan

**Tidak perlu mengulang run V9 untuk memperbaiki kegagalan teknis. Rekomendasi saya: berhenti menambah waktu tuning secara rutin.**

Eksperimen ini sudah menjawab dua pertanyaan penting: 1024 tidak selalu lebih baik, dan penambahan waktu dapat meningkatkan AP tetapi tidak otomatis semua metrik. Besarnya kenaikan saat ini belum memberi alasan kuat untuk terus mengejar angka dengan test historis yang sama.

- Pertahankan **V7 sebagai hasil utama yang dikunci**, sesuai keputusan sebelumnya.
- Simpan **V9 sebagai eksperimen tambahan lengkap**, termasuk kandidat yang menurun.
- Bila membahas kandidat terbaik V9, gunakan **LGBM 2048+8 tuned berdasarkan CV AP**. Jangan memilih XGB 2048 hanya karena melihat F1 test OOF tertinggi.
- Jika kelak ada keputusan menjadikan V9 sebagai hasil utama, jelaskan perubahan protokol dan status eksplorasinya secara transparan; tidak ada penggantian otomatis melalui audit ini.
- Prioritas sesudah analisis: pembahasan hasil yang jujur, batas penggunaan model, lalu validasi implementasi model yang disepakati. Bukan menganggap semua 109 label sudah andal.

Kalimat hasil yang aman untuk laporan:

> Pada eksperimen tambahan V9, penambahan anggaran pencarian dari 8 menjadi 16 jam meningkatkan rata-rata macro-AP validasi pada keempat konfigurasi. LightGBM dengan Morgan Fingerprint 2048 bit dan delapan deskriptor memperoleh macro-AP validasi tertinggi sebesar 0,306549 dan macro-AP test sebesar 0,316227. Peningkatan tidak terjadi pada seluruh metrik, dan hasil diperlakukan sebagai eksplorasi karena menggunakan kembali data test historis.

## 9. Bukti perhitungan

Checksum **1.314 file hasil**, 117 file paket input, 61 file sumber yang tercatat, dan 380 file yang dibekukan sebelum evaluasi test cocok. Delapan kandidat memiliki total **872 model label**. Metrik test dihitung ulang pada kedua aturan ambang; AP validasi dan pemilihan ambang OOF diperiksa pada seluruh 12 konfigurasi × 5 seed. Grup yang sama tidak melintasi train–test maupun train–validasi dalam definisi grup yang tersimpan.

- `audit.json`: cakupan verifikasi checksum, pemisahan grup, metrik dan CV.
- `recomputed_metrics.csv`: hasil penghitungan ulang delapan kandidat pada kedua aturan ambang.
- `selected_model_per_label.csv`: AP, precision, recall, F1 dan hitungan TP/FP/FN/TN per label.
- `v7_comparison.csv`: perbandingan V7–V9 pada fitur 2048+8 yang sama.
- `audit_v9.py`: skrip audit yang hanya membaca hasil eksperimen dan menulis bukti analisis di foldernya sendiri.

Perbedaan sangat kecil antara AP fold tersimpan dan AP yang dihitung ulang dari OOF muncul karena probabilitas OOF disimpan sebagai float32, sedangkan LightGBM menghitung metrik sebelum konversi tersebut. Selisih mutlak maksimum pada AP rata-rata per seed yang diperiksa adalah **0,000001438**. Besarnya dicatat di `audit.json`; tidak mengubah pemeringkatan kandidat atau interpretasi hasil.

