# 05. Mana yang dipakai belajar dan mana yang dipakai menguji?

## Tujuan dan file yang dibuka

Buka notebook **04_Grup_Split_dan_Fold.ipynb**, `research/splits.py`, `asal_kohor.py` dan `DATA_UNTUK_BELAJAR/02_SETELAH_PREPROCESSING/pembagian_train_test.csv`. Kamu perlu bisa membedakan train, validation, test, fold dan seed sebelum membaca Optuna.

| Istilah | Arti dalam penelitian ini |
|---|---|
| Train | 5.353 molekul untuk mengembangkan resep dan melatih model |
| Validation | Bagian train yang sementara tidak dipakai fit dalam satu fold |
| Test | 1.333 molekul pada partisi beku untuk evaluasi akhir protokol |
| Fold | Satu giliran pembagian fold-train dan fold-validation |
| Seed | Angka untuk membuat pengacakan bisa diulang |
| Group | Kumpulan molekul yang harus tetap di partisi yang sama |

Jangan menyebut semua 5.353 molekul selalu dipakai fit setiap penilaian. Dalam CV, sebagian menjadi validation. Sesudah resep dan pemilihan kandidat dibekukan, model final kandidat dilatih memakai seluruh train.

## Mengapa pembagiannya memakai grup?

Bayangkan dua baris sebenarnya mewakili struktur tanpa stereo yang sama. Kalau satu masuk train dan satu test, penilaian bisa terlalu mudah. Protokol menyatukan molekul ke grup jika strukturnya sama setelah stereo dihapus **atau** fingerprint Morgan historisnya identik. Hubungan ini diteruskan: kalau A tersambung B dan B tersambung C, ketiganya satu grup.

Batasnya: aturan ini menjaga identitas yang ditentukan protokol, bukan menjamin seluruh molekul mirip atau satu scaffold selalu bersama. Jadi jangan mengatakan 'semua kemiripan kimia sudah bebas kebocoran'.

## Mengapa masih ada angka 1024 di asal grup?

Fitur model sekarang tetap 2048+8. Tetapi **identitas grup dan pembagian molekul sudah dibuat pada sejarah kohor memakai struktur tanpa stereo atau Morgan1024 identik**. Penelitian utama mempertahankan pembagian itu agar hasilnya sesuai artefak beku. `asal_kohor.py` menerima fingerprint historis tersimpan untuk memeriksa asal grup; ia tidak membuat fitur model 1024 untuk training utama.

Ini perlu diketahui agar penjelasanmu sesuai kode. Menghapus jejaknya tidak mengubah sejarah pembagian data. Mengelompokkan ulang dari 2048 berarti mengubah eksperimen. Karena itu materi menempatkan informasi ini di asal split, tidak menyebut 1024+5 sebagai model utama.

## Outer split: train dan test

Kode asli `outer_split()` di `asal_kohor.py`:

```python
cv = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
a, b = next(cv.split(np.zeros(len(groups)), groups=groups))
return np.sort(a), np.sort(b)
```

Input `groups` berisi nomor grup setiap molekul. `test_size=0.2` berlaku pada **jumlah grup**, jadi jumlah molekul test tidak harus tepat 20%. `seed=42` membuat pilihan grup dapat diulang. `a` dan `b` adalah indeks baris, bukan isi fitur. `np.sort` merapikan urutan indeks. Hasil aktualnya 5.353 train dan 1.333 test.

Seleksi 109 label memakai dukungan pada train setelah split ini. Jadi bukan memilih label berdasarkan label mana yang menghasilkan skor test bagus. Di jalur saat ini, `dataset.verify()` memeriksa bahwa indeks, urutan, hash dan label masih sesuai kontrak beku.

## Inner fold: latihan penilaian di dalam train

`inner_folds()` di `research/splits.py` menggunakan `StratifiedGroupKFold`. Anggota satu grup wajib tetap bersama; stratified berusaha menjaga pembagian kelas 0/1 untuk label yang sedang dinilai. Setiap classifier memiliki target biner sendiri, sehingga **fold dibentuk per label dan per seed**. Jangan menganggap satu daftar fold universal dipakai untuk seluruh 109 label.

```python
cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
candidates = list(cv.split(np.zeros(len(train)), truth, train_groups))
```

`truth` adalah target satu label, bukan matriks 109 label. `train_groups` adalah grup pada partisi train. `candidates` berisi pasangan indeks fold-train dan fold-validation. Kode memeriksa kedua kelas tersedia; kalau jumlah fold tidak layak bisa diturunkan karena kelayakan kelas, bukan karena skornya jelek. Pada hasil utama semua pasangan label-seed mendapat **5 fold**.

Contoh ilustrasi lima fold: giliran 1 menyimpan kelompok bagian 1 untuk validation dan belajar dari bagian 2-5. Giliran berikutnya menyimpan bagian 2. Setelah lima giliran, tiap molekul train pernah diprediksi saat tidak dipakai fit untuk prediksi itu. Inilah asal skor OOF, dibahas lebih lanjut di bab 07.

Lima seed evaluasi: **7, 21, 42, 84, 100**. Ini mengubah pembagian fold. Seed estimator model tetap 42 dalam implementasi utama. Jadi 'lima seed' tidak berarti ada lima test set baru atau semua komponen random diganti.

## Pemeriksaan hasil pembagian

`validate_grouped_split()` memeriksa indeks tidak dobel, cakupan lengkap dan tidak ada grup melintasi train-test maupun fold-train/validation. Audit terbaru memeriksa 109 x 5 x 5 = **2.725 fold label-seed**; pembagiannya sama untuk delapan kandidat. Ini membantu perbandingan karena model diuji pada pembagian yang sama.

Tetap ada batasan: test pernah dilihat pada pengembangan sebelumnya. CV yang diulang juga bukan nested CV, karena tidak memakai lingkar validasi terpisah untuk mengukur keseluruhan proses pemilihan hyperparameter. Jadi hasilnya dilaporkan sebagai penelitian eksploratori, bukan perkiraan generalisasi yang sepenuhnya bebas pengaruh pengembangan.

Latihan: pilih satu source_row di pembagian_train_test.csv. Lihat split dan group_id. Cari semua anggota grupnya dan cek apakah split-nya sama. Jangan melakukan pembagian ulang.

Pertanyaan dosen: 'Kenapa seed bukan jumlah pengulangan model?' Jawaban: 'Seed adalah pengendali pengacakan. Protokol memakai lima nilai seed untuk mengulang pembagian CV; setiap pengulangan tetap berisi lima fold per label.'

Pertanyaan: 'Apakah group split menjamin tidak ada kebocoran?' Jawaban: 'Pemeriksaan menunjukkan tidak ada irisan grup menurut aturan struktur/fingerprint yang digunakan. Saya tidak menyatakan semua bentuk kemiripan molekul atau seluruh risiko bias sudah hilang.'

Checklist: [ ] bisa menggambar train-test; [ ] bisa menjelaskan validation di dalam train; [ ] tahu fold per label; [ ] tahu asal grup historis; [ ] bisa menyebut batas penilaian.
