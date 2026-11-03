# 08 — Cara baca hasil: model kita bagusnya di bagian mana?

**Target belajar:** hitung metrik, baca hasil asli, jelaskan pemilihan model. Angka contoh di awal adalah **ilustrasi**, bukan hasil penelitian.

## Mulai dari satu aroma dulu

Bayangkan kita mengecek label `sweet` pada enam molekul. Target **1** berarti label tercatat di dataset; **0** berarti tidak tercatat. Nol belum membuktikan molekul benar-benar tidak beraroma sweet.

Model menghasilkan **skor**, lalu aturan ambang mengubahnya jadi keputusan. Dengan ambang 0,5, skor minimal 0,5 menjadi prediksi 1.

| Molekul ilustrasi | Target | Skor sweet | Prediksi | Nama hasil |
|---|---:|---:|---:|---|
| A | 1 | 0,9 | 1 | TP |
| B | 0 | 0,8 | 1 | FP |
| C | 1 | 0,7 | 1 | TP |
| D | 0 | 0,6 | 1 | FP |
| E | 1 | 0,4 | 0 | FN |
| F | 0 | 0,1 | 0 | TN |

**TP**: prediksi 1 dan target 1. **FP**: prediksi 1 tetapi target 0. **FN**: prediksi 0 tetapi target 1. **TN**: prediksi 0 dan target 0. Kita dapat TP=2, FP=2, FN=1, TN=1.

## Hitung metriknya pelan-pelan

Huruf pada rumus adalah jumlah kejadian tadi.

| Metrik | Rumus | Hitungan ilustrasi | Pertanyaan yang dijawab |
|---|---|---|---|
| Precision | TP/(TP+FP) | 2/4 = **0,50** | Dari yang diprediksi sweet, berapa bagian yang cocok dengan dataset? |
| Recall | TP/(TP+FN) | 2/3 = **0,667** | Dari semua sweet tercatat, berapa bagian yang tertangkap? |
| F1 | 2TP/(2TP+FP+FN) | 4/7 = **0,571** | Bagaimana gabungan precision dan recall? |
| Specificity | TN/(TN+FP) | 1/3 = **0,333** | Dari target 0, berapa bagian yang tetap diprediksi 0? |
| Accuracy | (TP+TN)/(TP+FP+FN+TN) | 3/6 = **0,50** | Berapa bagian seluruh keputusan yang cocok? |

F1 juga bisa dihitung sebagai **2PR/(P+R)**. P berarti precision dan R berarti recall untuk **label yang sama**. Jangan menghitung macro F1 dari macro precision dan macro recall: hitung F1 setiap label dulu, baru rata-ratakan.

Di kode, [metrics.py baris 10](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/metrics.py:10>) menerima tiga vektor: `truth` = target asli, `probabilities` = skor, `predicted` = keputusan. Untuk satu label test, masing-masing panjangnya 1.333. Implementasi precision dan recall memakai scikit-learn; kasus pembagi nol dikembalikan sebagai nol melalui `zero_division=0`.

Potongan asli [validation.py baris 69](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/validation.py:69>):

```python
def f1_from_metrics(value):
    precision, recall = value["precision"], value["recall"]
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
```

`precision` adalah P dan `recall` adalah R. Kondisi terakhir mencegah pembagian nol.

## AP dan AUROC melihat urutan skor

**AP, Average Precision**, mengecek apakah target positif muncul di peringkat atas. Pakai tabel ilustrasi yang sudah diurutkan dari skor terbesar. Ada tiga positif, di peringkat 1, 3, dan 5. Precision sampai posisi itu adalah 1/1, 2/3, dan 3/5.

**AP = (1 + 2/3 + 3/5)/3 = 0,756.**

Pembagi 3 adalah jumlah positif. Rumus sederhana ini berlaku untuk contoh tanpa skor kembar. Secara umum AP menjumlahkan precision yang dibobot kenaikan recall. Kode benar-benar memanggil `average_precision_score(y, p)` di [metrics.py baris 23](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/metrics.py:23>). Kolom `auprc` dalam artefak berarti **AP dari fungsi tersebut**, bukan integrasi trapesium kurva precision–recall.

**AUROC** bisa dibayangkan dengan memasangkan setiap positif melawan setiap negatif. Positif punya skor 0,9; 0,7; 0,4. Negatif punya 0,8; 0,6; 0,1. Ada 3×3=9 pasangan. Skor positif menang masing-masing 3, 2, dan 1 kali. Jadi **AUROC = 6/9 = 0,667**. Skor sama mendapat setengah kemenangan. Implementasinya `roc_auc_score(y, p)` di baris 26.

**AP dan AUROC memakai skor, bukan keputusan ambang 0,5.** Mengganti ambang saja tidak mengubah keduanya. Contoh tadi juga menunjukkan AP dan AUROC bukan angka yang sama.

## Dari satu label ke 109 label

Kode menghitung metrik **per label**, lalu `summarize()` di [metrics.py baris 33](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/metrics.py:33>) mengambil rata-ratanya. Itulah **macro**. Secara umum helper mengabaikan nilai metrik None jika suatu kelas tidak tersedia. Pada test utama, seluruh 109 label mempunyai kedua kelas sehingga rata-rata memakai 109 label. Contoh: AP sweet 0,6 dan woody 0,2 menghasilkan macro AP (0,6+0,2)/2=0,4. Setiap label mendapat bobot sama, meskipun jumlah contohnya berbeda.

Untuk seleksi, urutannya: **AP setiap fold → rata-rata fold tiap label → rata-rata label → rata-rata lima seed**. Seed yang dipakai 7, 21, 42, 84, 100; fold maksimal lima per label. Lihat `cross_validate()` baris 8 dan `repeated_grouped_cv()` baris 119 pada [validation.py](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/validation.py:119>).

Ini beda dengan menggabungkan seluruh skor **OOF** lalu menghitung AP sekali. OOF berarti skor suatu baris dibuat saat baris itu menjadi validasi. Kode merata-ratakan skor OOF antarseed untuk memilih ambang tambahan. Karena AP bergantung peringkat, **rata-rata AP fold tidak harus sama dengan AP gabungan OOF**.

## Hasil asli: delapan kandidat utama

A/C memakai Morgan 2.048 fitur. B/D menambah delapan deskriptor, menjadi 2.056 fitur. A/B memakai baseline; C/D memakai parameter hasil Optuna. Tabel berasal dari [comparison_all.csv](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/04_Hasil/comparison_all.csv>). Angka dibulatkan enam desimal; F1 memakai ambang utama 0,5.

| Kandidat | AP validasi | AP test | AUROC test | Macro F1 test |
|---|---:|---:|---:|---:|
| XGBoost A | 0,249589 | 0,249043 | 0,809232 | 0,274899 |
| XGBoost B | 0,277412 | 0,288125 | 0,822636 | 0,311835 |
| XGBoost C | 0,267049 | 0,273029 | 0,825034 | 0,270369 |
| XGBoost D | 0,296959 | 0,308827 | 0,838546 | 0,305902 |
| LightGBM A | 0,251681 | 0,252837 | 0,809417 | 0,277184 |
| LightGBM B | 0,283211 | 0,296380 | 0,825813 | **0,320721** |
| LightGBM C | 0,269114 | 0,276461 | 0,827912 | 0,252220 |
| LightGBM D | **0,299731** | **0,310531** | **0,842985** | 0,293156 |

**LightGBM D dipilih dari AP validasi tertinggi**, sesuai [experiment.py fungsi select_candidates(), baris 66](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/experiment.py:66>). AP 0,310531 bukan berarti “akurasi aroma 31,05%”. Itu ringkasan kualitas peringkat skor pada label test.

D nggak unggul di semua metrik. LightGBM B memiliki F1 lebih tinggi pada ambang 0,5. Peringkat yang membaik belum tentu memberi keputusan terbaik pada ambang tetap. Dibanding B, D menaikkan recall tetapi menurunkan precision.

## Kenapa ada hasil ambang OOF?

`select_validation_thresholds()` di [validation.py baris 89](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/validation.py:89>) mencoba ambang 0,05 sampai 0,95, langkah 0,05, lalu memilih F1 OOF tertinggi **masing-masing label**. Kalau seri, pilih yang paling dekat 0,5; kalau masih seri, pilih yang lebih besar. Ini evaluasi tambahan; hasil utama tetap ambang 0,5.

Hasil LightGBM D dari [test_summary.json](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/04_Hasil/lgbm_D/test_summary.json>) dan [test_summary_validation_thresholds.json](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/04_Hasil/lgbm_D/test_summary_validation_thresholds.json>):

| Metrik test | Ambang 0,5 | Ambang dari OOF |
|---|---:|---:|
| Macro precision | 0,213862 | 0,303265 |
| Macro recall | 0,515089 | 0,411890 |
| Macro F1 | 0,293156 | 0,330419 |
| Rata-rata label diprediksi/molekul | 9,355589 | 6,372093 |

Model jadi lebih selektif secara keseluruhan: precision naik, tetapi lebih banyak positif terlewat. Target test rata-rata 4,254314 label per molekul. AP dan AUROC tetap karena skor tidak berubah. Nama `true_positive_rate` pada bagian `diagnostics` JSON dihitung sebagai `np.mean(truth)`: itu **proporsi target positif**, bukan recall.

## Batas kesimpulan dan latihan

Menurut [experiment.json baris 171](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/config/experiment.json:171>), test pernah diperiksa pada riwayat pengembangan dan repeated CV ini bukan nested CV. Jadi hasil bersifat **eksploratori**. Simpangan baku AP validasi D sebesar 0,001360 menggambarkan variasi lima seed; itu bukan bukti perbedaan signifikan atau jaminan pada molekul baru.

[Audit 27 Oktober](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/04_Hasil/audit_27_oktober/README.md>) juga membandingkan prediksi semua nol: macro accuracy-nya **0,960970**, melampaui D **0,923660**, tetapi recall dan F1 nol. Ini alasan konkret kenapa accuracy bukan dasar seleksi.

Jawaban sidang singkat: “Saya memilih LightGBM D berdasarkan macro AP validasi. AP test-nya 0,310531. Pada ambang 0,5, F1-nya masih di bawah baseline B. Ambang tambahan dari OOF meningkatkan F1 dengan konsekuensi recall turun. Saya melaporkan kedua kondisi dan batas evaluasinya.”

**Cek paham:** pakai ambang 0,75 pada tabel ilustrasi. Hasilnya TP=1, FP=1, FN=2, TN=2. Kenapa recall berubah tetapi AP tetap? Cari pembanding ambang di [notebook 07](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/07_Evaluasi_dan_Pemilihan.ipynb>).
