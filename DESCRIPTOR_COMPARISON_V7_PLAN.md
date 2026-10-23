# V7 revisi 2: Morgan 2048 dengan atau tanpa 8 deskriptor

Keputusan: 24 September 2026. Persiapan diselesaikan: 25 September 2026.
Status: paket untuk PC kampus/PC lain; training penelitian belum dijalankan.
Revisi ini menggantikan rancangan v7 awal (1024 bit, komparasi 5 vs 8).
Gunakan **perfume-campus-grouped-v7-m2048-d8.zip**.

## Keputusan dan hipotesis

Pengguna menetapkan Morgan **2048 bit**. Tambahannya dipilih **8 deskriptor**:
mencakup seluruh delapan jenis deskriptor Suh, tanpa menganggap pemilihan lima
dari delapan sebelumnya sudah terbukti paling tepat.
Jumlah ikatan berputar, atom berat, dan cincin dapat melengkapi sifat molekul
yang sudah diwakili lima deskriptor. Ini hipotesis, bukan jaminan performa naik.

2048 menyediakan lebih banyak posisi bit daripada 1024 dan dapat mengurangi
tabrakan fitur akibat pelipatan hash. Namun, 2048 bukan angka optimum yang sudah
dibuktikan. Nilai penelitian terletak pada pengujian terkontrol; penggunaan RDKit
atau perubahan ukuran fitur sendiri bukan bukti kebaruan metode.

## Referensi dan fitur

[Suh et al. (2025), Feature extraction](https://www.nature.com/articles/s42004-025-01651-7)
menggunakan delapan jenis deskriptor berikut pada representasi MD, terpisah dari
representasi Morgan. Suh juga membandingkan RF, XGBoost, dan LightGBM.
Gabungan Morgan + delapan deskriptor adalah eksperimen proyek ini, bukan resep
gabungan identik yang direproduksi dari Suh.

| Urutan setelah Morgan | Fungsi RDKit | Makna |
|---|---|---|
| 1 | MolWt | Berat molekul |
| 2 | MolLogP | Perkiraan lipofilisitas |
| 3 | NumHDonors | Jumlah donor ikatan hidrogen |
| 4 | NumHAcceptors | Jumlah akseptor ikatan hidrogen |
| 5 | TPSA | Luas permukaan polar topologis |
| 6 | NumRotatableBonds | Jumlah ikatan yang dapat berputar |
| 7 | HeavyAtomCount | Jumlah atom selain hidrogen |
| 8 | RingCount | Jumlah cincin |

[Morgan RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints):
radius 2, 2048 bit, tanpa kiralitas; versi paket terkunci 2026.3.4.
Matriks model float32: Morgan tetap bernilai 0/1, deskriptor berupa nilai numerik,
bukan delapan bit biner.

## Data dan pembagian tetap

- GoodScents, IFRA 2019, Leffingwell, Arctander 1960, Sigma-Aldrich 2014.
- Dataset beku: data/builds/perfume-five-grouped-v2.
- **6686 molekul, 109 label, 5353 train / 1333 test, 5363 kelompok**.
- Fitur baru: data/builds/perfume-m2048-d8-v7/features.npz.
  descriptors.csv menampilkan delapan nilai per molekul; manifest.json mencatat skema/checksum.
- Tidak trimming, mengubah label, pseudo-labeling, atau membagi ulang data.
- Verifikasi: identitas baris, nilai finite, lima kolom awal persis sama dengan
  deskriptor lama, serta pelipatan OR dua bagian Morgan 2048 persis sama dengan
  Morgan 1024 beku. Kesalahan menghentikan proses, tidak diisi nol diam-diam.
- Kelompok lama berbasis struktur nonstereokimia identik atau fingerprint 1024
  identik dipertahankan. Fingerprint 2048 identik juga mempunyai lipatan 1024 identik.
  Ini konservatif dan menjaga perbandingan, tetapi bukan scaffold split atau
  jaminan semua molekul yang mirip terpisah.
- Sumber relevan dengan parfum, bukan bukti setiap molekul eksklusif parfum.
  Anotasi yang tidak tercatat bukan bukti sensori negatif.

## Delapan kandidat

Setiap baris dijalankan untuk **XGBoost dan LightGBM**.

| Kondisi | Fitur | Dimensi | Training |
|---|---|---:|---|
| A | Morgan 2048 | 2048 | Baseline 100 pohon |
| B | Morgan 2048 + 8 | 2056 | Baseline sama dengan A |
| C | Morgan 2048 | 2048 | Optuna baru 8 jam |
| D | Morgan 2048 + 8 | 2056 | Optuna baru 8 jam |

Total 8 kandidat, 4 studi baru, **32 jam tuning** ditambah CV dan fitting final.
Tidak mengimpor trial, skor, atau parameter terbaik v6.
Dalam tiap algoritma, C/D memulai dari resep generik yang sama: 300 pohon,
learning rate 0,05, depth 5, subsample/colsample 0,8, alpha 0,01, lambda 1.
Ruang pencarian dan parameter lengkap: alignment/protocol_v7.json.

## Validasi, tuning dan keputusan

1. TPE seed 42, 8 jam waktu aktif/studi termasuk overhead loop. Jumlah trial dapat
   berbeda; waktu sama bukan jaminan kualitas optimum sama. Tanpa early stopping/pruning.
2. Objective: rata-rata Average Precision setiap label/fold pada training.
   Kolom historis auprc memakai sklearn average_precision_score, bukan integrasi PR trapezoidal.
3. Semua kandidat memakai lima pengulangan grouped CV, seed 7, 21, 42, 84, 100.
   Masing-masing label memakai lima fold yang layak. Fold untuk label+seed yang
   sama wajib identik pada semua kandidat. Estimator random_state tetap 42.
4. Bobot kelas dihitung hanya dari training fold. Test tidak memilih parameter,
   ambang atau kandidat. Seed/fold tidak dianggap sampel eksperimen independen.
5. Ambang utama 0,5. Ambang tambahan per label dipilih untuk F1 dari rerata
   prediksi out-of-fold: grid 0,05 sampai 0,95, langkah 0,05.
6. Kandidat otomatis: mean macro AP validasi tertinggi. Seri persis dalam algoritma
   mengikuti A, B, C, D; antaralgoritma mengikuti urutan kunci kandidat seperti v6.
   Kestabilan dilaporkan deskriptif, bukan syarat pemenang atau uji signifikansi.
7. Bekukan parameter, CV, ambang dan pilihan sebelum final fit/evaluasi test.
   Kandidat otomatis belum langsung mengganti rilis v6; hasil perlu direview.

## Kontras dan batas klaim

- **B-A:** pengaruh paket delapan fitur pada Morgan 2048 dengan parameter baseline sama.
- **D-C:** dua representasi setelah tuning masing-masing dengan anggaran sama.
  Parameter terbaik boleh berbeda; bukan efek fitur dengan semua parameter tetap.
- **C-A dan D-B:** perubahan setelah tuning pada representasi yang sama.
- Laporkan mean, SD, delta per seed serta jumlah delta positif/negatif/seri.
  Selisih kecil/tidak konsisten belum menjadi bukti kuat suatu metode lebih baik.
- Enam metrik: accuracy, AUROC, AP, specificity, precision, recall; ditambah
  macro-F1 dan hasil per label. Pisahkan ambang utama dan ambang OOF.
- **V6 vs v7 adalah sejarah eksperimen:** A/C berubah 1024 menjadi 2048;
  B/D berubah 1024+5 menjadi 2048+8. Perubahan B/D tidak memisahkan efek panjang
  Morgan dari jumlah deskriptor. V7 ini tidak menguji 5 vs 8 pada panjang Morgan yang sama.
- Data/label/split berbeda dari Suh, jadi angka publikasi bukan bukti keunggulan langsung.
- Test historis sudah dilihat. Hasil tetap eksploratori; repeated CV bukan nested CV.
  Evaluasi konfirmatori membutuhkan data independen atau rancangan evaluasi tersendiri.

## Pelaksanaan dan resume

Panduan: WINDOWS_GUIDE_V7.md / LINUX_GUIDE_V7.md.
CPU 6 thread, estimator berurutan, tanpa GPU dan tanpa instalasi sensor suhu.
Pemeriksaan RAM/disk/data serta pembatasan host laptop asal tetap aktif.
Ekstrak ke folder baru; jangan resume v7 1024 dengan kode revisi ini.

Runner: alignment/experiments_v7.py.
Run ID: perfume-five-grouped-v7-m2048-d8.
Urutan: verifikasi -> tuning C/D -> repeated CV A-D -> cek fold ->
freeze -> final fit/evaluasi -> tabel -> ZIP otomatis melalui launcher.

Ctrl+C dan tunggu proses berhenti. Resume dengan launcher sama, folder/host dan
dependency sama. Trial terputus menjadi FAIL; trial selesai dan budget tercatat
dipertahankan. Pencatatan tiap 15 detik membatasi kehilangan catatan saat hard kill.
Setelah freeze, tuning/CV tidak diulang. Artefak selesai diverifikasi checksum.

## Hasil dan agenda laporan

Output: runs/perfume-five-grouped-v7-m2048-d8, termasuk:
comparison_all.csv, descriptor_deltas.csv, paired_descriptor_cv.json,
descriptor_deltas_per_label_test.csv, historical_v6_all.csv, comparison_history.csv,
RESULTS_SUMMARY.md, inference_bundle.json, model, OOF, threshold, SQLite, log, checksum.
Bundle memakai 2048 atau 2056 fitur sesuai kandidat pilihan; urutan 109 label dicatat.
Inferensi Python: alignment.predict_v7. ONNX/Android tetap ditunda.

**Setelah hasil tersedia: audit, lalu rombak laporan sesuai metode dan hasil terverifikasi.**
Agenda permanen: AFTER_V7_REPORT_UPDATE.md.
