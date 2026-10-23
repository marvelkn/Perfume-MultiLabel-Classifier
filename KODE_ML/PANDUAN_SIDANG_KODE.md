# Panduan sidang: memahami dan mempertanggungjawabkan kode ML

Audit: **23 Oktober 2026**. Fokus: metode penelitian utama, yaitu lima sumber Pyrfume, Morgan 2048 bit, delapan deskriptor RDKit, XGBoost dan LightGBM, serta Optuna. Angka versi pada nama file hanya identitas teknis arsip eksperimen.

**Penilaian:** inti ML masih wajar untuk skripsi S1 Informatika. Kode pengaturan eksperimen dan riwayat modulnya lebih rumit daripada yang diperlukan untuk belajar. Yang paling perlu dibenahi adalah keterbacaan dan cara menelusuri alur, bukan mengganti model atau metodologinya.

Audit ini tidak mengubah kode penelitian, dataset, parameter, atau model. Tidak ada training ulang. Contoh perubahan di bawah hanya diuji secara terpisah di memori.

## 1. Kerumitan dan kewajaran kode

| Bagian | Penilaian | Alasan |
|---|---|---|
| Membaca tabel, join identitas, membersihkan label | Wajar untuk S1 | Lima sumber memang mempunyai bentuk tabel dan identitas berbeda. |
| RDKit, XGBoost, LightGBM, metrik scikit-learn | Wajar | Proyek memakai implementasi pustaka; tidak membuat algoritma boosting atau kimia komputasi dari nol. |
| Satu classifier per label | Wajar, tetapi wajib dipahami | Solusi sederhana untuk multi-label: setiap aroma diperlakukan sebagai masalah biner. |
| Grouped CV, OOF, bobot kelas | Menengah | Perlu mengerti pembagian data, kebocoran, dan apa yang dihitung dari bagian train saja. |
| Union-find pembentukan grup | Menengah | Algoritma struktur data untuk menyatukan hubungan yang saling terhubung. Masih relevan untuk S1. |
| Checkpoint, SQLite, checksum, batas waktu, lock | Lebih rumit secara rekayasa | Berguna untuk eksperimen puluhan jam dan melanjutkan proses yang terputus. Bukan metode ML tambahan. |
| Import antarfile historis, one-liner panjang | Layak dirapikan | Membuat alur sulit dilacak tanpa memberikan manfaat ilmiah. |

Gaya kode tidak dapat membuktikan siapa penulisnya atau apakah seseorang memakai AI. Banyaknya pemeriksaan juga bukan bukti kode “terlalu bagus untuk mahasiswa”. Justru beberapa bagian masih padat dan bercampur riwayat pengembangan. Yang bisa dipertanggungjawabkan adalah alasan setiap keputusan, data yang dipakai, hasil pemeriksaan, dan pemahaman terhadap perilaku program.

Kamu tidak harus hafal setiap baris pengelolaan file. Namun, kamu perlu bisa menunjukkan kode inti dan menjelaskan input, output, serta alasan pemakaiannya tanpa mengandalkan nama file saja.

### Batas folder KODE_ML

Folder ini adalah pintu masuk belajar, **belum merupakan implementasi mandiri seluruh pipeline**:

- Notebook menampilkan artefak hasil preprocessing dan EDA; tidak menjalankan cleaning lengkap.
- `03_optuna.py` membaca konfigurasi dan riwayat trial; objective Optuna ada di `alignment/tuning_v6.py`.
- `04_training_evaluasi.py` membaca hasil dan menghitung ulang metrik; tidak memanggil training.
- Contoh fitur, split, dan prediksi berguna untuk demonstrasi, tetapi pemahaman inti tetap perlu sampai ke fungsi di `alignment/`.

## 2. Peta alur yang benar-benar dijalankan

```text
5 sumber: tabel anotasi + identitas + struktur
  -> cocokkan Stimulus/CID/CAS dengan SMILES
  -> tolak identitas/struktur/label yang tidak memenuhi aturan
  -> samakan istilah label; gabungkan molekul yang identik
  -> bentuk grup struktur/fitur; pisahkan train dan test
  -> pilih label berdasarkan dukungan pada train
  -> siapkan Morgan 2048 + 8 deskriptor dengan urutan baris tetap
  -> buat A/B/C/D untuk XGBoost dan LightGBM
  -> Optuna pada CV train untuk C/D
  -> nilai seluruh kandidat dengan CV berkelompok berulang
  -> pilih kandidat dan ambang tambahan dari validasi; bekukan pilihan
  -> fit final seluruh kandidat pada seluruh train
  -> evaluasi test; simpan model dan prediksi
  -> SMILES baru dapat diprediksi dengan model tersimpan
```

| Tahap | Fungsi yang perlu dibaca | Input → output | Yang perlu dijelaskan |
|---|---|---|---|
| Identitas molekul | `_attach_smiles`, `_attach_sigma`, `_cas_bridge` di [build_dataset.py](../src/build_dataset.py), sekitar baris 30–86 | Tabel anotasi/identitas → anotasi yang memiliki SMILES | Join memakai identitas; bukan posisi baris atau kesamaan nama aroma. Konflik pemetaan tidak dipilih sembarangan. |
| Cleaning dan gabung | `merge_sources` di [perfume_data.py](../alignment/perfume_data.py), baris 146 | Lima sumber → satu baris per SMILES kanonik, label dan jejak sumber | Struktur multikomponen tidak dipotong otomatis. Label dari molekul identik digabung memakai set. |
| Label | `map_labels`, `encode_and_split` di file yang sama, baris 116 dan 216 | Istilah aroma → matriks target 0/1 | Kamus berasal dari IFRA; alias eksplisit. Pemilihan label memakai train, bukan test. |
| Grup/split | [grouped_splits.py](../alignment/grouped_splits.py), baris 10, 45, 52 | Struktur, fingerprint, target → grup dan indeks pembagian | Anggota satu grup tidak menyeberang train/test atau train/validasi. |
| Fitur | `calculate_morgan`, `calculate` di [descriptors_v7.py](../alignment/descriptors_v7.py), baris 28 dan 38 | SMILES → 2048 bit + delapan angka | Fungsi deskriptor ditetapkan sebelumnya; RDKit menghitung nilainya. |
| Estimator dan bobot | `make_model`, `balanced_sample_weights`, `fit` di [experiments.py](../alignment/experiments.py), baris 184, 196, 208 | X, satu kolom y, parameter → satu classifier | Bobot dihitung dari bagian train yang sedang dipakai. |
| Satu trial Optuna | `suggest`, `tune` di [tuning_v6.py](../alignment/tuning_v6.py), baris 24 dan 71 | Ruang parameter + skor trial sebelumnya → resep kandidat dan skor CV | Yang dimaksimalkan adalah macro AP validasi. |
| CV dan OOF | `cross_validate`, `repeated_grouped_cv` di [experiments.py](../alignment/experiments.py), baris 260 dan 371 | X/y train, fold, resep → metrik CV dan prediksi OOF | Model sementara tidak melatih baris yang sedang divalidasi. |
| Seleksi/final fit | `select_candidates`, `run_all` di [experiments_v7.py](../alignment/experiments_v7.py), baris 82 dan 250; `fit_final_models` di [experiments_v5.py](../alignment/experiments_v5.py), baris 289 | Skor CV/resep → pilihan beku dan model final | Pilihan menurut CV; model final mempelajari seluruh 5.353 train. |
| Evaluasi | [metrics.py](../alignment/metrics.py), baris 10 dan 33 | Target test, skor, keputusan → metrik per label dan macro | Accuracy biner rata-rata per label; `auprc` adalah key untuk Average Precision. |
| Inferensi | `predict` di [predict_v7.py](../alignment/predict_v7.py), baris 16 | Satu SMILES → 109 skor dan keputusan label | Urutan fitur/label harus sama; skor belum dikalibrasi dan bukan intensitas. |

### Data aktual: jangan keliru menjelaskan penurunan jumlah baris

| Tahap | Jumlah |
|---|---:|
| Rekaman dari lima sumber | 13.337 |
| Rekaman lolos aturan sebelum gabung identitas | 11.777 |
| Rekaman ditolak | 1.560 |
| Molekul setelah penggabungan identitas | 6.686 |
| Train / test | 5.353 / 1.333 |
| Istilah kamus IFRA / label terpilih | 184 / 109 |

Jadi, selisih dari sekitar 13 ribu menjadi 6 ribu **bukan semuanya data tanpa label yang dibuang**. Banyak rekaman menunjuk molekul yang sama sehingga digabung. Dua molekul berbeda yang sama-sama berlabel `sweet` tetap merupakan dua contoh berbeda.

Target 1 berarti label tercatat setelah pemetaan; target 0 berarti tidak tercatat. Nol **belum tentu negatif sensoris yang benar-benar diuji**. Ada 11 baris train yang menjadi seluruhnya 0 setelah pemilihan label; ini tercatat pada audit, bukan SMILES kosong.

## 3. Konsep inti yang wajib dikuasai

### Satu molekul → fitur → banyak keputusan

- Morgan merupakan fingerprint pola lingkungan atom, radius 2, 2048 posisi, `includeChirality=False`. Bukan teks SMILES yang diubah ke kode biner ASCII.
- Delapan deskriptor: `MolWt`, `MolLogP`, `NumHDonors`, `NumHAcceptors`, `TPSA`, `NumRotatableBonds`, `HeavyAtomCount`, `RingCount`. Urutannya tetap.
- Deskriptor bukan delapan bit tambahan. Nilainya angka, misalnya massa molekul atau jumlah cincin.
- X untuk A/C memiliki 2048 kolom; B/D memiliki 2056 kolom. Target memiliki 109 kolom.
- Satu kandidat terdiri dari **109 classifier biner**. Skor antararoma tidak harus berjumlah 1. Hubungan antarlabel belum dimodelkan secara eksplisit.

Bagian inti fitting final, dari `fit_final_models`:

```python
for j, name in enumerate(labels):
    model = core.fit(algorithm, params, x, y[:, j], core.Limits())
```

Ini cuplikan inti; fungsi asli juga menangani checkpoint dan penyimpanan. `x` berisi fitur semua baris train; `y[:, j]` mengambil satu label, misalnya sweet. `core.fit` membuat estimator, menghitung bobot kelas, lalu memanggil `.fit`. Loop berikutnya melatih label lain. Satu resep parameter dipakai bersama untuk semua label pada kandidat tersebut.

### Grup dan fold: tiga detail yang sering terlewat

1. Grup menyatukan struktur kanonik yang sama setelah stereo dihapus, **atau** fingerprint 1024 asli yang persis identik. Hubungannya transitif melalui union-find: A terhubung B dan B terhubung C berarti ketiganya satu grup. Ini bukan scaffold split atau clustering berdasarkan ambang kemiripan Tanimoto.
2. Model memakai 2048 bit, tetapi grup lama dipertahankan. Kode memeriksa bahwa fingerprint 2048 dapat dilipat kembali sesuai fingerprint 1024 yang mendasari kelompok. Ini menjaga kohor/split penelitian. Tidak ada jaminan semua molekul yang sekadar mirip selalu masuk grup yang sama.
3. Fold dibentuk **per label** dengan `StratifiedGroupKFold`. Untuk label dan seed yang sama, kandidat memakai fold yang sama. Antarlabel, pembagian fold bisa berbeda. `test_size=0.2` dihitung pada grup, sehingga proporsi baris tidak harus tepat 20%.

Kode mengizinkan penurunan jumlah fold jika kelas tidak memungkinkan; pada hasil yang diaudit seluruh label menggunakan lima fold. Informasi stereo yang tidak masuk fitur tetap merupakan keterbatasan representasi; grouping tidak mengembalikan informasi itu.

### Seed, CV, OOF, dan final fit

| Penggunaan | Nilai dan fungsi |
|---|---|
| Split train/test | Seed 42; pembagian beku |
| Split saat tuning | Fold beku pada data train; seed awal 42 |
| Repeated grouped CV | Seed 7, 21, 42, 84, 100; tiap seed membentuk lima fold per label |
| Randomness estimator XGB/LGBM | `random_state=42` tetap |
| Sampler Optuna | TPE seed 42 |

Lima seed CV tidak berarti lima pencarian Optuna terpisah. Urutannya: satu seed → lima fold → seed berikutnya. Untuk satu label dan satu seed, setiap baris train menjadi validasi satu kali. Prediksi dari model yang tidak melatih baris itu disebut OOF. Prediksi OOF lima seed kemudian dirata-ratakan.

Repeated CV saja membutuhkan 8 kandidat × 109 label × 5 seed × 5 fold = **21.800 fitting classifier**. Ini belum memasukkan trial Optuna dan 872 model final. Karena itulah checkpoint dan pengaturan eksekusi punya manfaat nyata.

Setelah resep dan pilihan dibekukan, final fitting memakai seluruh 5.353 train. Model-model fold hanya alat untuk menilai konfigurasi. Model final tidak dipilih dengan mencari satu fold yang mendapat skor paling tinggi.

### Optuna dan parameter

| Skenario | Fitur | Parameter |
|---|---|---|
| A | Morgan 2048 | Baseline, `n_estimators=100` |
| B | Morgan 2048 + 8 deskriptor | Baseline, `n_estimators=100` |
| C | Morgan 2048 | Optuna |
| D | Morgan 2048 + 8 deskriptor | Optuna |

Masing-masing ada pada XGBoost dan LightGBM. Parameter baseline lainnya mengikuti default pustaka dengan versi yang dikunci; default kedua algoritma tidak semuanya sama.

Satu trial: usulkan parameter → latih/validasi semua label dengan fold yang ditetapkan → rata-ratakan AP antar-fold untuk setiap label → rata-ratakan antar-label → kirim skor ke Optuna. TPE memakai riwayat trial untuk menyarankan kombinasi berikutnya. Ada empat studi, masing-masing delapan jam; bukan delapan jam untuk seluruh penelitian. Pruning dan early stopping tidak diaktifkan pada protokol ini.

| Parameter | Makna yang perlu dikuasai | Nilai tersimpan LightGBM D |
|---|---|---:|
| `n_estimators` | Banyak putaran boosting | 1100 |
| `learning_rate` | Besar kontribusi setiap langkah boosting | 0,00759754 |
| `max_depth` | Batas kedalaman pohon | 7 |
| `num_leaves` | Batas jumlah daun pohon | 75 |
| `min_child_samples` | Minimum contoh pada daun | 41 |
| `colsample_bytree` | Proporsi fitur untuk tiap pohon | 0,393844 |
| `subsample`, `subsample_freq` | Proporsi baris dan frekuensi sampling | 0,896796; 1 |
| `reg_alpha`, `reg_lambda` | Penalti regularisasi L1/L2 | 0,00003785; 0,267942 |

Nilai tersebut **terbaik yang ditemukan dalam pencarian ini**, bukan angka terbaik universal. Tidak perlu menghafal semua desimal; pahami fungsi parameter dan tunjukkan `recipe.json` sebagai bukti. Batas 2048 bit, delapan deskriptor, minimal 30 positif train, lima fold, serta budget waktu merupakan keputusan rancangan dengan alasan dan keterbatasan. Jangan mengklaim setiap angkanya sudah terbukti optimal.

### Bobot kelas, metrik dan ambang

Bobot satu kelas dihitung sebagai `jumlah_baris / (2 * jumlah_anggota_kelas)`, pada subset train untuk label tersebut. Kelas jarang mendapat bobot lebih besar. Ini tidak menambah data, tidak memperbaiki anotasi yang salah, dan tidak menjamin probabilitas sudah terkalibrasi.

- AP menilai ranking skor terhadap label positif. Key internal `auprc` di kode dihitung dengan `average_precision_score`; jangan menyebutnya area PR dengan integrasi trapezoid.
- Accuracy adalah akurasi biner per label lalu rata-rata macro, bukan persentase molekul yang semua labelnya benar sekaligus.
- Macro F1 dihitung sebagai rata-rata F1 per label, bukan menghitung F1 dari macro precision dan macro recall.
- Ambang utama 0,5 mengubah skor menjadi keputusan biner. Ambang tambahan dipilih per label untuk memaksimalkan F1 pada OOF, dari grid 0,05–0,95. Test tidak dipakai memilih ambang pada run ini.
- Pemilihan kandidat memakai **rata-rata macro AP validasi**, bukan AP test atau F1 test.

| LightGBM D pada test | Ambang 0,5 | Ambang tambahan OOF |
|---|---:|---:|
| AP | 0,310531 | 0,310531 |
| AUROC | 0,842985 | 0,842985 |
| Macro precision | 0,213862 | 0,303265 |
| Macro recall | 0,515089 | 0,411890 |
| Macro F1 | 0,293156 | 0,330419 |

AP dan AUROC tetap karena skor probabilitasnya sama; yang berubah hanya keputusan ambang. LightGBM B memiliki F1 pada 0,5 sebesar 0,320721, lebih tinggi daripada D. Jadi D dipilih sesuai kriteria AP validasi, bukan karena unggul pada semua metrik.

## 4. Bagian yang memang layak disederhanakan

### Contoh A — pengkodean label, sudah diuji

Asli, `perfume_data.py:217`:

```python
y = np.asarray(
    [[int(label in row) for label in taxonomy] for row in records.labels],
    dtype=np.uint8,
)
```

Lebih eksplisit:

```python
y = np.zeros((len(records), len(taxonomy)), dtype=np.uint8)

for row_index, row_labels in enumerate(records.labels):
    for label_index, label in enumerate(taxonomy):
        if label in row_labels:
            y[row_index, label_index] = 1
```

Versi kedua lebih panjang tetapi mudah dijelaskan: buat matriks nol, lalu tandai satu pada label yang tercatat. Keduanya wajar; comprehension asli bukan kesalahan. Uji pada data aktual menghasilkan **array uint8 (6686,184) identik**, lalu setelah seleksi 109 label dan indeks split hasilnya identik dengan target train/test tersimpan.

### Contoh B — penyusunan fitur, sudah diuji

Asli, `experiments_v7.py:39`, setelah pemeriksaan input:

```python
return np.column_stack([
    morgan, descriptors[:, :descriptor_count(key)]
]).astype(np.float32)
```

Alternatif setelah pemeriksaan input yang sama:

```python
count = descriptor_count(key)
if count == 0:
    result = morgan
else:
    result = np.column_stack([morgan, descriptors[:, :count]])
return result.astype(np.float32)
```

Pemeriksaan bentuk dan angka finite di awal fungsi harus tetap ada. Pada **seluruh 6.686 baris untuk delapan kandidat**, array sebelum/sesudah sama persis beserta `float32` dan dimensinya. Ini hanya penggantian cara menyusun array, bukan perubahan RDKit.

### Contoh C — memilih kandidat, sudah diuji

Asli, `experiments_v7.py:85`:

```python
chosen = {
    a: max(
        (k for k in CANDIDATES if k.startswith(a + "_")),
        key=lambda k: (scores[k], -"ABCD".index(k[-1])),
    )
    for a in ALGORITHMS
}
```

Lebih mudah dibaca:

```python
chosen = {}
for algorithm in ALGORITHMS:
    candidates = [f"{algorithm}_{condition}" for condition in "ABCD"]
    chosen[algorithm] = max(candidates, key=lambda key: scores[key])

overall = max(chosen.values(), key=lambda key: (scores[key], key))
```

`max` mempertahankan kemunculan pertama saat skor sama; urutan A/B/C/D menjaga aturan tie semula. Pemilihan `overall` tetap memakai aturan skor lalu nama kandidat seperti kode asli. Pada **6.561 kombinasi skor sintetik** dari {0; 0,5; 1} untuk delapan kandidat, seluruh keluaran pemilihan identik, termasuk tie. Ini bukti pada skor finite yang diuji, bukan seluruh domain input yang mungkin rusak.

### Perapian lain yang belum diterapkan atau diuji sebagai refactor menyeluruh

- Pecah satu baris berisi beberapa statement, ternary panjang, dan dictionary besar menjadi blok bernama jelas.
- Pisahkan helper identitas data dari runner lama `src/build_dataset.py`; hanya beberapa helper yang dipakai pipeline saat ini.
- Pisahkan core estimator/CV/fitting dari modul runner historis. Saat ini runner memakai fungsi dari beberapa file bernama versi berbeda.
- Perjelas dokumentasi yang usang: docstring `repeated_grouped_cv` menyebut dua finalis, padahal dipakai pada delapan kandidat. Komentar taxonomy Leffingwell pada modul lama juga bukan taxonomy aktif: pipeline sekarang hanya mengambil fungsi `normalize` dari modul tersebut dan memakai IFRA.
- Tambahkan contoh belajar yang menampilkan satu trial dan satu fold sebenarnya. Mengganti nama wrapper menjadi lebih jelas boleh direncanakan, tetapi bukan bukti bahwa kode intinya sudah dipahami.

Jangan menghapus file historis sebelum memeriksa import. Source hash seluruh `alignment/` dan `src/` menjadi bagian identitas run; perubahan komentar pun dapat membuat pemeriksaan resume menolak kode. Arsip hasil tidak boleh “disesuaikan” dengan mengedit checksum supaya lolos.

## 5. Bukti konsistensi dan batas pemeriksaan

| Pemeriksaan yang benar-benar dijalankan | Hasil | Batas kesimpulan |
|---|---|---|
| Hash 11 modul preprocessing/fitur/split/tuning/training/metrik/inferensi terhadap `identity.json` run | Semua cocok | Modul yang dibahas sama dengan arsip run; ini bukan audit seluruh file repositori. |
| Encoding eksplisit vs comprehension | Identik pada 6686×184 target dan train/test 109 label | Tidak menguji perubahan taxonomy/alias. |
| Cabang eksplisit penyusunan fitur vs fungsi asli | Identik seluruh baris, delapan kandidat, dtype float32 | Tidak mengubah perhitungan RDKit atau model. |
| Pemilihan kandidat yang diperluas menjadi loop | Identik pada 6.561 tabel skor termasuk tie | Tidak menjalankan ulang tuning atau fitting. |
| Hitung ulang enam metrik + macro F1 kandidat terpilih, dua mode ambang | Selisih absolut maksimum **0,0** dari ringkasan tersimpan | Menggunakan prediksi yang sudah ada; bukan bukti bahwa training baru akan bit-identik. |

Record SHA256 yang dipakai untuk uji encoding: `e62cbe41c117c8ffb72e88e4b67bd5831ec6d5f47d7ffdb40f40c352836f9671`.

Manifest dataset SHA256: `886e8e49d08bf8c5c73f0c3d13de25bd01b63ada4bdcc7e58f4b49ad7cd60c17`.

Mengganti dtype, urutan fitur, pembagian fold, label, weighting, urutan usulan Optuna, versi library, CPU/GPU, atau strategi paralel bukan perubahan kosmetik. Efeknya dapat kecil atau besar. Bahkan optimasi kecepatan pada pencarian berbatas waktu bisa mengubah jumlah trial yang selesai. **Kesetaraan hasil seluruh pipeline belum diuji karena tidak ada training ulang.**

Audit ini juga tidak mengulang validasi kimia semua anotasi, memeriksa ulang seluruh sitasi jurnal/LaTeX, atau membuktikan signifikansi statistik perbedaan model.

## 6. Risiko ilmiah yang perlu diakui saat sidang

1. **Anotasi tidak lengkap.** Target nol diperlakukan negatif secara operasional, tetapi bukan bukti tidak ada aroma.
2. **Cakupan parfum.** Lima sumber relevan dengan parfum; tidak ada verifikasi bahwa setiap molekul eksklusif parfum.
3. **Kesamaan kimia.** Grouping yang digunakan tidak menjamin seluruh scaffold atau molekul yang mirip terpisah.
4. **Optimisme pemilihan.** Repeated CV ini bukan nested CV; data train/CV digunakan untuk pengembangan dan pemilihan. OOF juga bukan test independen untuk menilai threshold yang dipilih darinya.
5. **Test historis pernah dilihat.** Freeze pada run sekarang tidak menghapus riwayat eksplorasi. Hasil tetap perlu dibahas dengan batas itu.
6. **Budget sama, trial berbeda.** XGB C/D menyelesaikan masing-masing 12 trial; LGBM C/D 104/74. Startup TPE 10; jangan mengaku semua studi sudah konvergen atau optimum.
7. **Bukti efek fitur terbatas.** Perbandingan B−A menguji paket delapan deskriptor pada baseline. Itu belum membuktikan tiap deskriptor secara individual berguna atau 2048 selalu lebih baik daripada 1024.
8. **Keunggulan tidak universal.** Model terpilih menang pada aturan AP validasi; beberapa metrik/skenario lainnya dapat lebih baik. Perbedaan langsung terhadap skor Suh tidak apple-to-apple jika kohor, label, dan split berbeda.

## 7. Latihan pertanyaan sidang

**1. “Mana baris yang benar-benar melatih model?”**

Jawab: “`core.fit` membuat estimator, menghitung bobot kelas, lalu menjalankan `.fit`. `cross_validate` memanggilnya untuk fold; `fit_final_models` untuk seluruh train.” Lanjutan: “Mengapa `y[:, j]`?” Karena satu kolom adalah target satu aroma.

**2. “109 label berarti satu klasifikasi dengan 109 pilihan?”**

Jawab: “Tidak. Ada 109 classifier biner; satu molekul bisa positif pada beberapa label.” Lanjutan: “Skornya harus berjumlah satu?” Tidak, tiap label dinilai terpisah.

**3. “Apa sebenarnya input yang diterima model?”**

Jawab: “Array fitur angka. SMILES dibaca RDKit terlebih dahulu menjadi Morgan dan deskriptor.” Lanjutan: “Apakah bit Morgan menyimpan teks SMILES?” Tidak, ia menyimpan pola substruktur yang dipetakan ke posisi bit.

**4. “Dari mana delapan deskriptor dipilih?”**

Jawab: “Jenisnya sudah ditetapkan pada rancangan dengan mengacu jenis fitur referensi; penggabungannya dengan Morgan diuji lewat A/B/C/D. RDKit menghitung nilai fungsi itu.” Lanjutan: “Mana yang paling berguna?” Eksperimen paket ini belum mengisolasi kontribusi masing-masing deskriptor.

**5. “Mengapa jumlah data turun drastis?”**

Jawab: “Dari 13.337 rekaman, 1.560 ditolak. Ada 11.777 rekaman valid yang kemudian digabung berdasarkan identitas menjadi 6.686 molekul.” Lanjutan: “Label sama berarti duplikat?” Tidak jika molekulnya berbeda.

**6. “Kenapa memakai 109 label?”**

Jawab: “Dari 184 istilah IFRA, 109 memenuhi aturan dukungan pada train. Batas minimal positif yang dikonfigurasi adalah 30.” Lanjutan: “Apakah angka 30 universal?” Tidak, itu batas operasional untuk kecukupan dukungan; tidak diklaim optimal universal.

**7. “Apa arti angka nol di target?”**

Jawab: “Label tidak tercatat setelah pemetaan, lalu diperlakukan sebagai negatif saat training/evaluasi.” Lanjutan: “Apakah pasti tidak berbau begitu?” Tidak; kelengkapan anotasi adalah keterbatasan data.

**8. “Kenapa tidak split acak biasa?”**

Jawab: “Saya menjaga anggota kelompok struktur/fitur ekuivalen tetap bersama supaya tidak menyeberang ke sisi evaluasi.” Lanjutan: “Apakah ini scaffold split?” Bukan; kriterianya struktur tanpa stereo yang identik atau fingerprint identik, dengan penggabungan transitif.

**9. “Apakah seluruh label punya lima fold yang sama?”**

Jawab: “Fold dibuat per label untuk stratifikasi kelas. Untuk label dan seed yang sama, semua kandidat dibandingkan pada fold yang sama.” Lanjutan: “Mengapa antararoma bisa beda?” Proporsi positif tiap aroma berbeda.

**10. “Seed 7 berarti mengulang tujuh kali?”**

Jawab: “Bukan. Itu angka awal pengacakan. Ada lima nilai seed yang masing-masing menghasilkan pembagian lima fold.” Lanjutan: “Apakah seed model juga berubah?” Pada kode ini estimator tetap memakai 42; yang diulang adalah split validasi.

**11. “OOF itu 20% test?”**

Jawab: “Bukan. OOF dibuat dari 5.353 train: setiap baris diprediksi model fold yang tidak melatih baris itu. Test 1.333 baris tetap terpisah.” Lanjutan: “Mengapa fit lagi?” Supaya model final belajar dari seluruh train setelah konfigurasi ditetapkan.

**12. “Apa yang dicari Optuna?”**

Jawab: “Kombinasi parameter yang memberi macro AP CV tertinggi dalam ruang dan budget yang ditetapkan.” Lanjutan: “Apakah pasti terbaik?” Hanya terbaik yang ditemukan; tidak menjamin optimum global.

**13. “Apa fungsi weighting?”**

Jawab: “Membesarkan kontribusi kelas yang lebih jarang pada proses fit. Bobot dihitung dari bagian train saat itu, per label.” Lanjutan: “Apakah memakai label test untuk hitung bobot?” Tidak.

**14. “Kenapa D dipilih padahal F1 B lebih tinggi?”**

Jawab: “Aturan pemilihan yang dikunci memakai AP validasi. Saya tetap melaporkan F1 B yang lebih tinggi pada ambang 0,5 sebagai tradeoff.” Lanjutan: “Boleh pindah metrik setelah melihat hasil?” Itu akan mengubah kriteria pemilihan dan perlu dilaporkan, bukan dilakukan diam-diam.

**15. “Apa arti AP 0,31?”**

Jawab: “Itu ukuran kualitas ranking skor terhadap label positif, bukan 31% molekul diprediksi benar.” Lanjutan: “Apa accuracy yang dilaporkan?” Rata-rata akurasi biner per label, bukan semua label tepat sekaligus.

**16. “Mengapa AP sama setelah ambang berubah?”**

Jawab: “Skor model tidak berubah. Ambang hanya mengubah keputusan positif/negatif; AP dihitung dari ranking skor.” Lanjutan: “Dampak ambang OOF?” Precision/F1 dapat naik dengan recall turun, seperti hasil kandidat terpilih.

**17. “Skor 0,8 artinya peluang 80% atau intensitas 80%?”**

Jawab: “Itu keluaran skor model dari `predict_proba`, belum dikalibrasi. Tidak ada target intensitas, jadi tidak bisa dibaca sebagai kekuatan aroma.” Lanjutan: “Bisa menjamin molekul baru?” Tidak; pergeseran distribusi dan keterbatasan anotasi tetap ada.

**18. “Kenapa banyak file pemeriksaan dan checkpoint?”**

Jawab: “Run berlangsung lama. Checkpoint menyimpan progres, checksum menjaga identitas data/model, dan lock mencegah proses bersamaan menulis run yang sama.” Lanjutan: “Apakah itu metode baru?” Bukan; itu pengelolaan eksperimen agar hasil dapat ditelusuri.

**19. “Apakah penelitian Anda lebih baik dari Suh?”**

Jawab: “Bukti utama saya adalah komparasi internal dengan data dan pembagian yang sama. Angka publikasi Suh tidak langsung apple-to-apple karena data, label dan split berbeda.” Lanjutan: “Bukti generalisasi independen?” Belum tersedia; test historis dan non-nested CV membatasi klaim.

**20. “Apakah kode dibantu AI?”**

Contoh jawaban, gunakan hanya sejauh sesuai pekerjaan yang benar-benar kamu lakukan: “Ya, AI membantu implementasi dan penelusuran masalah. Saya bertanggung jawab memahami rancangan, memeriksa kecocokan implementasi dengan metode, mengevaluasi keluaran, dan menjelaskan keterbatasannya.” Lanjutan: penguji dapat meminta penjelasan satu fungsi. Karena itu latihan menelusuri `fit`, `cross_validate`, dan `map_labels` lebih penting daripada menghafal jawaban ini.

## 8. Rekomendasi akhir dan urutan belajar

| Kategori | Bagian | Tindakan |
|---|---|---|
| **Sudah wajar** | Pustaka ML/RDKit, join identitas, loop per-label, metrik, konfigurasi, pemeriksaan shape dan urutan | Pertahankan. Pelajari peran dan data yang melewatinya. |
| **Perlu dipahami lebih dalam** | Grouping union-find, fold per-label, berbagai seed, class weights, AP/OOF, final fit dan batas anotasi | Prioritaskan latihan menjelaskan satu contoh konkret. |
| **Layak disederhanakan/diperbaiki** | Statement padat, nama lokal samar, docstring usang, campuran helper/runner historis, contoh belajar yang belum memperlihatkan inti fitting | Rapikan keterbacaan secara bertahap setelah audit; jaga arsip dan uji kesetaraan tiap bagian. |

Urutan belajar yang paling efisien:

1. Satu molekul: jelaskan SMILES → fitur → 109 skor → keputusan.
2. Satu rekaman mentah: telusuri identitas, pemetaan label, duplikat, dan alasan penolakan.
3. Satu label dan satu fold: tunjukkan indeks train/validasi, bobot kelas dan `.fit`.
4. Satu trial: telusuri usulan parameter, skor objective, lalu perluas ke A/B/C/D.
5. Satu hasil akhir: jelaskan alasan memilih kandidat, final fit, metrik dan batas klaim.
6. Terakhir, pahami garis besar checkpoint, checksum dan resume.

Target pemahaman sebelum sidang: kamu bisa mengikuti satu baris data dan satu label melewati seluruh alur, membedakan validasi dari test, serta menjelaskan mengapa hasil yang baik di satu metrik tidak menjamin semua metrik naik. Tidak perlu merombak model agar terlihat sederhana. Yang dibutuhkan adalah implementasi yang dapat ditelusuri dan penjelasan yang sesuai fakta.
