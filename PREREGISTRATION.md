# Praregistrasi eksperimen ML Essenza — v1

**Dikunci secara lokal pada 8 September 2026, sebelum training protokol baru.** ID: `essenza-ml-v1-20260908`. Dokumen ini dan [protocol.json](protocols/essenza_v1_20260908/protocol.json) mencatat keputusan Step 3. Jumlah trial final baru dihitung di Step 5 dari pengukuran resource Step 4. Step 4 belum diizinkan.

Ini praregistrasi lokal setelah audit deskriptif dan pekerjaan historis, bukan registrasi publik atau klaim bahwa dataset belum pernah dilihat. Hash mendeteksi perubahan; hash tidak membuat file mustahil diubah. Amandemen harus memakai versi baru, alasan, waktu, dan hash sebelum/sesudah. Keputusan tidak boleh diubah berdasarkan final test.

## 1. Pertanyaan penelitian dan unit analisis

Bandingkan dua kandidat **Binary Relevance XGBoost dan LightGBM** yang dituning secara terpisah untuk 25 label aroma pada satu molekul. Nilai keberhasilan adalah performa validasi/test sesuai protokol, stabilitas deskriptif, kebutuhan komputasi, serta kelayakan deployment yang diuji pada tahap berikutnya.

Klaim optimal global, peningkatan tuning yang pasti, prediksi campuran parfum, intensitas sensori, atau probabilitas persepsi manusia tidak termasuk. Skor model tidak harus berjumlah 100%.

## 2. Dataset dan partisi yang dibekukan

| Keputusan | Nilai |
| --- | --- |
| Build | `data/builds/audited-20260904-v2` |
| Dataset ID | `5c0b0e7c9a56eb60fbc344fc3b6629a160dfa723fdb7b6ba398e621870f2f558` |
| Feature schema ID | `a1dce64f7591f9b40f5f2332d805b7f1c7f91f24d9af0dcd0964cda7c231c018` |
| Source revision | `8054ea98ed675005ec10e67359902f500e4911b0` |
| Development / test | 5.383 / 1.320 molekul |
| Fitur | Morgan radius 2, 2.048 bit, chirality aktif, 5 deskriptor, float32 |
| RDKit | 2026.03.4; environment lengkap tersimpan dalam protocol.json |
| Target | 25 label pada urutan manifest; pemilihan berdasarkan frekuensi development |
| Seed utama | **42** |
| Holdout threshold | **808** molekul, tidak masuk CV |
| Pool CV/fitting | **4.575** molekul |
| Setiap fold utama | 2.592 train, 458 early stop, 1.525 score; tiga fold |

Pemilihan label, source snapshot, batas development/test, dan 381 baris development tanpa target positif dipertahankan. Tidak ada penyaringan tambahan berdasarkan skor. Nilai nol adalah anotasi yang tidak tercatat; tidak otomatis membuktikan aroma tidak ada.

Indeks [development_partitions.json](reports/step3_20260908/development_partitions.json) cocok persis dengan arsip sebelumnya dan lolos pemeriksaan keterpisahan serta dukungan kedua kelas pada semua label/peran. Inisialisasi Step 5 wajib menghasilkan indeks yang sama. Stratifikasi multilabel mengikuti pendekatan yang dibahas oleh [Sechidis et al. (2011)](https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10).

## 3. Metrik dan aturan seleksi

Objektif trial:

```text
J = mean_folds(mean_25_labels(AP_label))
```

AP menggunakan definisi noninterpolasi `sklearn.metrics.average_precision_score`, bukan luas trapezoid kurva PR. Semua label diberi bobot sama dalam macro AP; label tanpa positif diberi AP nol. Definisi mengikuti [scikit-learn 1.6.1](https://scikit-learn.org/1.6/modules/generated/sklearn.metrics.average_precision_score.html).

Hanya trial **COMPLETE**, bernilai finite, dengan tiga fold dan 25 label lengkap yang menjadi kandidat. Pilih J terbesar; jika sama persis, pilih nomor trial terkecil dan periksa sebelum fitting. Pilih algoritme final berdasarkan J dari dua kandidat tuned tersebut; nilai sama persis memilih XGBoost sebagai aturan administratif deterministik, bukan bukti keunggulan.

Tiga baseline per learner tetap dilaporkan sebagai pembanding terpisah. Baseline tidak dimasukkan secara retrospektif ke pool kandidat tuned. Jika tuning kalah terhadap baseline, laporkan **tidak ada manfaat tuning yang teramati**; jangan menambah trial atau mengubah ruang pencarian untuk mengejar klaim peningkatan.

Fitting akhir memakai pool 4.575 molekul dan median integer best iterations tiap label dari trial utama. Threshold per label memaksimalkan F1 pada 808 molekul holdout: semua skor unik ditambah satu nilai di atas maksimum; jika F1 sama dalam toleransi `1e-12`, pilih threshold lebih tinggi. Keputusan positif memakai `score >= threshold`.

Bekukan hash kedua manifest serta keputusan CV sebelum test. CV yang digunakan memilih trial dapat optimistis; pemisahan pemilihan model dari evaluasi akhir didukung [Cawley dan Talbot (2010)](https://jmlr.org/papers/v11/cawley10a.html). Dataset ini tetap bukan test eksternal independen.

## 4. Baseline, imbalance, dan Optuna

Strategi yang diizinkan: **none, class_weight, random_oversample**. **MLSMOTE nonaktif** pada protokol utama maupun sensitivitas ini. Perubahan memerlukan amandemen eksplisit sebelum eksperimen terkait.

Class weight memakai rasio negatif/positif pada label training terkait. Random oversampling menambah hingga `floor(0,5 × n_train)` baris multilabel utuh: pilih label minoritas berdasarkan imbalance ratio di atas rata-rata, lalu pilih baris positif label tersebut. Semua resampling dilakukan pada train saja dengan seed utama/varian ditambah nomor fold.

| Pengaturan | XGBoost | LightGBM |
| --- | --- | --- |
| Sampler | TPE seed 42, 10 startup trial | Sama |
| Pruner | MedianPruner, startup 10, warmup step 1 | Sama |
| Pelaporan pruning | Rata-rata AP kumulatif setelah semua label dalam fold selesai; step 0,1,2 | Sama |
| Early stopping | Logloss pada holdout stop, patience 50 | Sama |
| Round maksimum | 2.000 | 3.000 |
| Trial bersamaan | 1 | 1 |
| Parameter antarlabel | Satu konfigurasi trial untuk semua label | Sama |
| Profil operasional | safe-training, 2 thread, 20 menit/sesi | Sama |

Ruang numerik tetap mengikuti [snapshot source](protocols/essenza_v1_20260908/search_space_source.py.txt) dari `suggest_parameters` dan `fit_binary`. Snapshot merupakan bagian hash protokol. Baseline memakai override kosong dan default library yang dipin; default tersebut dapat berada di luar ruang tuning.

TPE melakukan pencarian sekuensial yang menggunakan hasil sebelumnya untuk mengusulkan parameter; ini bukan pencarian yang menjamin optimum global. Landasan: [Bergstra et al. (2011)](https://papers.nips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html), [Akiba et al. (2019)](https://arxiv.org/abs/1907.10902), serta dokumentasi [TPESampler](https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.TPESampler.html) dan [MedianPruner](https://optuna.readthedocs.io/en/v4.5.0/reference/generated/optuna.pruners.MedianPruner.html). Optimasi berurutan dan seed tetap mengikuti batas reproducibility pada [FAQ Optuna](https://optuna.readthedocs.io/en/v4.5.0/faq.html#how-can-i-obtain-reproducible-optimization-results).

## 5. Budget berdasarkan resource, bukan skor

Pengguna menetapkan **14.400 detik / 4 jam komputasi aktif per model** untuk tuning. Waktu ini mencakup overhead sesi tuning, trial complete/pruned/failed, dan interupsi. Cooldown, baseline, smoke, fitting akhir, dan sensitivitas berada di luar budget tuning ini. Perubahan budget memerlukan keputusan baru; memilih budget bukan izin menjalankan training sekarang.

Smoke Step 4, setelah kondisi termal memenuhi syarat, dibatasi **total 300 detik aktif** untuk kedua learner secara berurutan: XGBoost lalu LightGBM. Profil safe-smoke memakai satu thread. Probe memakai fold 0, semua fitur, serta **musty, citrus, fruity** (peringkat dukungan development 1, 13, 25, diurutkan berdasarkan jumlah lalu nama). Masing-masing memakai ketiga strategi imbalance dan parameter probe tetap dalam protocol.json. Full round ceiling dan early stopping 50 dipertahankan; ukur waktu fit dan predict tanpa memakai AP/F1 smoke untuk seleksi.

Diperlukan sembilan label-fit selesai per learner agar estimasi valid. Probe terputus/censored, sensor bermasalah, atau suhu tidak aman menghasilkan STOP; jangan mengekstrapolasi seolah fit selesai. Gunakan nilai maksimum load, resampling, serta fit+predict dari probe:

```text
t_hat[m] = 1,5 × (max_load[m] + 3 × (max_resample[m] + 25 × max_fit_predict[m]))
N = min(50, floor(0,8 × 14.400 / max(t_hat[xgb], t_hat[lgbm])))
```

Satuan waktu detik. Kedua learner menerima target **N yang sama**, agar jumlah kesempatan pencarian direncanakan sama; waktunya tidak harus sama. Tidak diasumsikan bahwa dua thread memberi percepatan 2×. Faktor 1,5, cadangan 20%, floor 20 trial, dan cap 50 adalah aturan perencanaan rekayasa, bukan konstanta optimal dari jurnal.

Step 5 GO hanya jika kedua estimasi valid, smoke termal GO, **N ≥20**, dan setiap `t_hat ≤960 detik` agar satu trial diperkirakan menyisakan ruang dalam sesi 20 menit. Jika tidak, STOP dan dokumentasikan penyesuaian resource sebelum tuning. Dua puluh percobaan menyediakan ruang di luar 10 startup; tetap tidak membuktikan konvergensi.

Hitung N sebagai seluruh percobaan COMPLETE/PRUNED/FAIL, bukan hanya yang berhasil. Trial RUNNING akibat kill harus diperiksa dan diselesaikan statusnya tanpa menghapus riwayat. Hentikan pada N percobaan atau 14.400 detik aktif, mana yang lebih dulu. Tidak ada penggantian failure yang otomatis menambah N/waktu. Dua interupsi resource berturut-turut menghentikan tahap untuk investigasi. Mulai satu trial per sesi; sesi berikutnya setelah cooldown. Batch maksimal dua hanya setelah review resource.

Jika cap waktu mencegah salah satu learner mencapai N, laporkan perbandingan terpotong resource, bukan eksperimen equal-attempt lengkap. Total aktif kumulatif harus diperiksa sebelum/saat setiap sesi; batas 20 menit pada CLI saat ini belum menegakkan cap lintas sesi secara otomatis.

Pelaporan kecukupan tuning mencakup trajectory best-so-far terhadap trial/waktu, status seluruh trial, skor antar-fold, selisih baseline, dan parameter yang menyentuh batas. Plateau atau parameter terbaik di batas adalah informasi keterbatasan, bukan dasar menambah budget berdasarkan skor.


## 6. Sensitivitas seed dan scaffold

Semua sensitivitas memakai **parameter/imbalance terpilih dari trial utama dan rounds per label yang dibekukan**, tanpa retuning atau early stopping ulang. Batas development/test, 25 label, fitur, serta holdout threshold tetap. Holdout threshold tidak dipakai untuk fitting maupun skor sensitivitas.

| Varian per learner | Seed | Training per fold | Score per fold |
| --- | ---: | --- | --- |
| Kontrol random dengan rounds tetap | 42 | 3.050 | 1.525 |
| Sensitivitas random | 123 | 3.050 | 1.525 |
| Sensitivitas random | 2026 | 3.050 | 1.525 |
| Scaffold | 42 | 2.073 / 3.035 / 4.042 | 2.502 / 1.540 / 533 |

Partisi final: [sensitivity_execution_partitions.json](reports/step3_20260908/sensitivity_execution_partitions.json). Semua training/score memiliki kedua kelas untuk seluruh label dan saling lepas. Seed varian berlaku untuk learner; resampling memakai seed varian + fold. Jalankan berurutan sesuai tabel, terpisah per learner, dengan resume per unit dan batas resource plan.

**Alasan rounds tetap:** audit rancangan awal mendapati holdout early stopping scaffold fold 1 tidak memiliki kedua kelas pada apple dan winey. Menggabungkan train+stop untuk semua sensitivitas, memakai rounds utama yang dibekukan, mengatasi kebutuhan early-stop tersebut tanpa membuang label atau mengganti seed. Kontrol random seed 42 harus difit ulang dengan aturan ini; jangan langsung memakai skor CV utama yang menggunakan early stopping sebagai kontrol.

Scaffold memakai Murcko dengan chirality dan GroupKFold tiga fold, shuffle=True, seed 42. Seluruh molekul tanpa cincin tetap satu kelompok. Terdapat 606 kelompok development; kelompok kosong terbesar memuat 2.303 molekul (42,78%), sehingga jumlah molekul antar-fold tidak seimbang. Dasar memakai pemisahan struktur: [MoleculeNet](https://arxiv.org/html/1703.00564v3); perilaku kelompok mengikuti [GroupKFold 1.6.1](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GroupKFold.html). Desain spesifik proyek ini tidak diklaim identik dengan seluruh protokol MoleculeNet.

Laporkan AP setiap fold, mean macro AP antar-fold, dan pooled out-of-fold macro AP pada 4.575 molekul yang sama. AP pooled berbeda dari rata-rata AP fold, terutama ketika ukuran fold scaffold berbeda. Untuk random seeds, laporkan tiga nilai, mean, sample SD, dan selisih kedua learner secara deskriptif. Bandingkan scaffold dengan kontrol random-42 rounds tetap.

Sensitivitas ini **bersyarat pada resep yang sudah dipilih**, bukan evaluasi nested CV yang tidak bias. Data pool yang sama pernah dipakai untuk memilih resep utama. Jangan menafsirkannya sebagai bukti performa eksternal pada scaffold baru atau signifikansi statistik dari tiga seed. Varian gagal/terputus tetap dilaporkan sebelum test, tanpa aggregate parsial yang diberi label hasil lengkap. Jangan memilih seed/label yang menguntungkan.

## 7. Evaluasi akhir dan batas klaim

Kedua kandidat tuned selesai dan manifest/seleksi dibekukan sebelum evaluasi test satu kali pada Step 12. Laporkan macro/micro AP, macro/micro F1, macro precision/recall, Hamming loss, subset accuracy, macro ROC-AUC dengan jumlah label valid, serta metrik/support per label. AUC tak terdefinisi memakai null. Metrik sekunder tidak menggantikan aturan seleksi utama.

Protokol ini menggunakan variasi fold/seed secara deskriptif; tidak merencanakan p-value yang menganggap fold CV saling independen. Keunggulan numerik pada split ini tidak otomatis berarti keunggulan signifikan atau berlaku pada populasi lain.

Keterbatasan yang wajib masuk pembahasan:

- Anotasi tidak tercatat dianggap negatif; 381/5.383 development (7,08%) tidak mempunyai positif dalam 25 target terpilih.
- Dataset menggabungkan sumber dengan cakupan dan taxonomy berbeda. Acceptance source adalah hitungan record, bukan kontribusi molekul unik.
- Rentang massa molekul development 4,003–1.297,128; contoh domain mencakup molekul tanpa target positif seperti `[He]` dan molekul besar. Dataset tidak dibatasi khusus pada bahan parfum organik/volatil. Tidak dilakukan filter baru setelah melihat skor.
- Audit historis menunjukkan 1.231/1.320 test (93,26%) berbagi scaffold dengan development; angka ini tidak dihitung ulang dengan membuka test pada Step 3.
- Sebagian molekul sudah ada pada eksperimen terdahulu; final test baru bukan validasi eksternal independen.
- Tuning terbatas pada ruang, sampler, seed, dan resource yang telah dipilih. Tidak ada jaminan global optimum atau peningkatan dari baseline.

## 8. Gate pelaksanaan dan keselarasan laporan

Sebelum Step 4: periksa ulang sensor dan baseline termal; gunakan batas mulai/stop pada [RUNTIME_SAFETY.md](RUNTIME_SAFETY.md). Hasil panas Step 1 tetap alasan readiness STOP sampai ada pengukuran valid baru. Pengaman kooperatif belum dibuktikan pada beban nyata.

Sebelum Step 5/8/9: cocokkan seluruh hash protokol, dataset, konfigurasi, dependency, dan primary split; buat budget freeze serta ledger aktif kumulatif. CLI saat ini menyimpan batas per sesi, bukan cap 4 jam lintas sesi. Ledger/runner harus memeriksa sisa budget sebelum/saat melanjutkan dan tidak menambah N untuk mengganti failure. Jangan membuat run baru sekadar untuk menghindari cap.

Sebelum Step 11: siapkan runner khusus development untuk partisi sensitivitas yang dibekukan dan rounds tetap; jangan memakai `build_dataset --seed/--split` untuk membangun ulang outer test, atau memakai `init` biasa untuk varian karena seed saat ini diwarisi dari manifest utama. Runner belum dibuat/dijalankan pada Step 3.

Jika implementasi perlu perubahan, simpan hash source baru, bukti tes, dan alasan perubahan mekanis dalam amandemen sebelum run terkait. Jangan mengubah metrik, seed, target, atau aturan berdasarkan hasil test. Snapshot source Step 3 tidak diklaim sebagai versi semua tooling masa depan.

Bab III masih menyebut runtime 30 menit dan 90°C/30 detik, budget belum terkunci, dan scaffold belum spesifik. [alignment.md](reports/step3_20260908/alignment.md) mencatat pembaruan yang perlu diterapkan pada tahap penyelarasan laporan. Isi LaTeX tidak diubah oleh Step 3.

Frontend asli tetap main dan belum memuat layanan inference terbaru di worktree audit. Schema root/API historis memakai chirality=false, sedangkan protokol baru true. Rekonsiliasi frontend/deployment dilakukan pada tahap integrasi terpisah; tidak ada frontend atau bundle yang diganti di sesi ini.

## 9. Bukti dan cara memeriksa

- [Laporan audit Step 3](reports/step3_20260908/README.md): temuan dan batas interpretasi.
- [Notebook audit](reports/step3_20260908/audit_notebook.ipynb): tujuh sel kode dieksekusi berurutan di Python proyek.
- [Runner notebook](reports/step3_20260908/execute_audit.py): tidak memerlukan state kernel interaktif; tidak menjalankan model.
- [Protocol JSON](protocols/essenza_v1_20260908/protocol.json) dan [freeze_manifest.json](protocols/essenza_v1_20260908/freeze_manifest.json): aturan dan hash artefak.
- [Verifier praregistrasi](protocols/essenza_v1_20260908/verify_protocol.py): memeriksa hash, keputusan, dan partisi tanpa membuka matriks test.

Dari root ML asli, pemeriksaan freeze:

```powershell
.\.venv\Scripts\python.exe protocols\essenza_v1_20260908\verify_protocol.py
```

Eksekusi ulang notebook audit, bila diperlukan:

```powershell
.\.venv\Scripts\python.exe reports\step3_20260908\execute_audit.py
```

Notebook tersimpan sebagai bukti eksekusi. Menjalankannya ulang akan memperbarui metadata/output notebook; jangan memperbarui freeze diam-diam. Simpan revisi bukti baru secara terpisah bila ingin mempertahankan checksum praregistrasi.

