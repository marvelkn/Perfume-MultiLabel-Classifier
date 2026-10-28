# 1. Sebenarnya skripsi kita mau menjawab apa?

**Tujuan belajar:** kamu bisa menjelaskan masalah penelitian, bedanya belajar dan memprediksi, serta hubungan penelitianmu dengan penelitian sebelumnya.

## 1.1 Mulai dari satu pertanyaan

Pertanyaan kita kira-kira begini: **“Kalau kita tahu struktur satu molekul, label aroma apa saja yang kemungkinan cocok menurut pola data penelitian?”**

Contohnya **vanillin**. Di dataset gabungan kita, molekul ini punya catatan `caramel`, `chocolate`, `creamy`, `gourmand`, `powdery`, `sweet`, dan `vanilla`. Jadi kita butuh sistem yang bisa memberi **beberapa label sekaligus**. Istilahnya **multi-label**. `Sweet` boleh muncul bersama `vanilla`; sistem tidak dipaksa memilih salah satu.

Yang dipelajari adalah hubungan **struktur → label yang tercatat**. Sistem ini tidak mencium sampel, tidak membaca resep parfum, dan tidak mengukur kadar bahan. Struktur saja juga belum mencakup kondisi penciuman, konsentrasi, atau perbedaan persepsi tiap orang. Batas ini perlu kamu ingat saat menjelaskan manfaatnya.

## 1.2 Input, proses, output

| Bagian | Yang terjadi dalam proyek kita |
|---|---|
| Input pengguna | Teks SMILES yang menyatakan struktur satu molekul |
| Pemeriksaan | RDKit mengecek apakah struktur bisa dibaca dan hanya satu komponen |
| Ubah menjadi angka | Buat Morgan 2.048 bit; pada kondisi B/D tambahkan delapan deskriptor |
| Model | Sekumpulan classifier XGBoost atau LightGBM memberi skor tiap label |
| Output | Sebanyak 109 skor, lalu bisa diurutkan atau dijadikan keputusan dengan ambang |

**Classifier** artinya model untuk membedakan kelas. Untuk label `vanilla`, dua kelasnya adalah target **1** dan **0**.

Jangan tertukar dua kegiatan ini:

- **Training:** model melihat banyak contoh fitur beserta targetnya, lalu membentuk aturan pohon.
- **Prediksi:** model yang sudah tersimpan menerima fitur molekul, lalu menjalankan aturan pohonnya. Model tidak belajar ulang setiap pengguna memasukkan SMILES.

Saat training, nama aroma dipakai untuk menentukan target. Saat prediksi, model menerima **angka fitur struktur**, bukan bocoran label asli atau nama “vanillin”.

## 1.3 Jadi, ada 109 model atau 109 pohon?

Ada **109 classifier biner untuk satu kandidat lengkap**, satu untuk setiap label. Pendekatan ini biasa disebut **Binary Relevance**: masalah besar dipecah menjadi pertanyaan kecil, misalnya “vanilla atau bukan?”, “sweet atau bukan?”, dan seterusnya.

Tetapi **satu classifier berisi banyak pohon**. XGBoost dan LightGBM adalah algoritma boosting: pohon ditambahkan bertahap supaya gabungan prediksinya makin sesuai target training. Jadi gambar yang tepat adalah:

```text
Fitur molekul yang sama
  ├─ classifier vanilla → banyak pohon → satu skor vanilla
  ├─ classifier sweet   → banyak pohon → satu skor sweet
  └─ ... sampai 109 label
```

Cuplikan dipersingkat dari kode aslinya terlihat di [training.py, fungsi fit_final_models, baris 55](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/research/training.py:55>):

```python
for j, name in enumerate(labels):
    # pemeriksaan model yang sudah tersimpan ada di antara baris ini
    model = fit(algorithm, params, x, y[:, j], Limits())
```

`j` adalah nomor label. `x` adalah fitur semua contoh training. `y[:, j]` mengambil **satu kolom target**, misalnya target `vanilla` untuk seluruh molekul. Jadi semua classifier melihat fitur struktur yang sama, tetapi pertanyaan targetnya berbeda.

Kalau membandingkan satu kandidat XGBoost dengan satu kandidat LightGBM, ada **109 × 2 = 218 classifier**. Kalau seluruh delapan kandidat A–D dari dua algoritma dihitung, ada **872 classifier final**. Jumlah tersebut belum menghitung model sementara selama validasi. Saat pengguna memprediksi dengan satu kandidat terpilih, yang dipakai satu set 109 classifier tersebut.

**Batas Binary Relevance:** model `vanilla` tidak menerima skor `sweet` sebagai input. Hubungan antarlabel tidak dipelajari secara eksplisit melalui rantai model.

## 1.4 Skor 80% itu maksudnya apa?

Contoh **ilustrasi**, bukan hasil eksperimen:

| Label | Skor | Keputusan jika ambang 0,5 |
|---|---:|---|
| vanilla | 0,80 | Ya |
| sweet | 0,70 | Ya |
| citrus | 0,10 | Tidak |

Angka `0,80` bisa ditampilkan sebagai `80%`, tetapi sebut **skor prediksi kelas positif**. Jangan menyebutnya “akurasi vanillin 80%”, “kadar vanilla 80%”, atau “pasti 80 dari 100 orang mencium vanilla”. Accuracy menghitung benar-salah pada sekumpulan data yang punya target pembanding. Skor satu molekul berbeda. Skor antarlabel juga **tidak harus berjumlah 100%**.

## 1.5 Penelitian terdahulu dipakai untuk apa?

Penelitian sebelumnya memberi dasar bahwa masalah ini bisa dimodelkan dan membantu menentukan pilihan yang layak diuji. Hasil mereka bukan skor baseline internal kita.

| Studi primer | Hal yang relevan | Batas perbandingan |
|---|---|---|
| [Saini dan Ramanathan, 2022](https://doi.org/10.1038/s41598-022-18086-y) | Menggabungkan 7.374 molekul dengan 109 label; menguji Binary Relevance dan Classifier Chains berbasis Random Forest. Menilai micro-F1. | Jumlah label yang sama tidak menjamin kosakata dan distribusinya sama. Micro-F1 berbeda dari macro AP kita. |
| [Suh dkk., 2025](https://doi.org/10.1038/s42004-025-01651-7) | Menggunakan 8.681 molekul dari 10 sumber. Membandingkan fitur gugus fungsi, delapan deskriptor, dan Morgan secara terpisah pada RF/XGBoost/LightGBM. ST-XGB dilaporkan AUROC 0,828 dan AUPRC 0,237. | Proyek kita memakai lima sumber, 6.686 molekul, dan 109 target. Morgan + delapan deskriptor adalah rancangan eksperimen kita, bukan konfigurasi Suh yang disalin persis. |

**Alasan yang terdokumentasi** di Bab 2: gabungan fitur, tuning, dan algoritma dibandingkan dengan baseline internal pada data penelitian yang sama. **Penjelasan untuk memahaminya:** Morgan merangkum pola lokal struktur, sedangkan delapan angka merangkum sifat molekul; masuk akal untuk menguji apakah keduanya saling membantu. Itu alasan menguji, belum bukti bahwa penambahan fitur pasti meningkatkan hasil.

Untuk dasar algoritmanya, rujukan primer adalah [Chen dan Guestrin, XGBoost, 2016](https://arxiv.org/abs/1603.02754), [Ke dkk., LightGBM, 2017](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), dan [Akiba dkk., Optuna, 2019](https://arxiv.org/abs/1907.10902). Optuna adalah alat pencari pengaturan training; **bukan classifier ketiga** yang memprediksi aroma.

## 1.6 Urutan baca dan latihan

1. Baca [SAFE_EXECUTION_PLAN.md](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/SAFE_EXECUTION_PLAN.md>) untuk keputusan penelitian. Untuk status aplikasi, prioritaskan audit 27 Oktober di HASIL_UNTUK_BELAJAR/audit_27_oktober/README.md karena status ONNX pada plan belum diperbarui.
2. Baca [Bab2.tex](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/REFERENSI_BELAJAR/laporan_snapshot/Bab2.tex:3>), terutama penelitian terdahulu dan multi-label. Ini bab yang dipanggil laporan utama `SKRIPSI.tex`.
3. Lihat [05_Model_dan_Bobot.ipynb](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/05_Model_dan_Bobot.ipynb>) setelah gambaran besar ini terasa masuk akal.

**Latihan:** kalau ada tiga aroma dan dua algoritma, berapa classifier untuk satu kondisi tiap algoritma? Jawaban: **enam**. Kalau tiap classifier punya 100 pohon, jumlah pohonnya 600. Ini hanya contoh.

**Kalau dosen bertanya:**

- **“Apa yang diprediksi?”** “Label aroma yang tercatat pada dataset berdasarkan fitur struktur satu molekul. Beberapa label bisa diprediksi bersamaan.”
- **“Kenapa tidak langsung bilang lebih baik dari Suh?”** “Data, label, pembagian, dan perhitungan metriknya berbeda. Kesimpulan perbandingan saya didasarkan pada kandidat yang diuji dalam protokol penelitian saya.”
- **“109 label berarti satu pohon per aroma?”** “Satu classifier per aroma, dan setiap classifier merupakan gabungan banyak pohon boosting.”

**Cek paham sebelum lanjut:** kamu bisa membedakan training/prediksi, classifier/pohon, skor/accuracy, serta baseline internal/skor artikel lain tanpa membaca teks ini.
