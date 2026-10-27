# Audit sebelum penguncian proyek — 27 Oktober 2026

**Keputusan:** hasil dan implementasi inti ML layak dipertahankan sebagai penelitian eksploratori dalam cakupan yang diperiksa. Belum tepat menyatakan seluruh proyek/dokumentasi selesai tanpa catatan. Tidak ditemukan kesalahan perhitungan utama atau kebocoran kelompok menurut aturan split yang diterapkan. Tidak ada training ulang yang diperlukan berdasarkan pemeriksaan ini.

Audit hanya membaca source, data, model, notebook, laporan, dan frontend; tidak mengubah dataset, parameter, pipeline, model, notebook, LaTeX, atau frontend. Folder ini berisi catatan dan skrip pemeriksaan, bukan jalur training baru.

## Bukti yang diperiksa ulang

| Pemeriksaan | Hasil |
|---|---|
| Snapshot lima sumber | 15 CSV sesuai checksum |
| Kontrak data dan preflight | Lulus; 6.686 molekul, 5.353 train, 1.333 test, 109 label; fitur 2.048 atau 2.056 |
| Hasil utama | 1.199 file sesuai completion manifest |
| Source aktif | 15 hash source/tes sama dengan verifikasi 26 Oktober |
| Tes backend baca-saja | 7 lulus, 25,19 detik; tidak fitting |
| Perhitungan test independen | Semua delapan kandidat: enam metrik + macro F1 pada 0,5 cocok; galat maksimum 1,11e-16 |
| Pembagian validasi | 2.725 fold label–seed diperiksa; lima fold tiap label/seed; sama pada delapan kandidat dan tanpa irisan kelompok |
| OOF | Target/urutan sesuai; rata-rata skor lima seed dan 109 threshold masing-masing kandidat cocok |
| Pemilihan model | LightGBM D sesuai mean macro AP CV; freeze tercatat sebelum event fitting final/test |
| Notebook | Delapan notebook, 125 sel kode mempunyai output tersimpan tanpa error; source aktif tidak berubah dari verifikasi 26 Oktober |
| ONNX | 109 hash aset dan kontrak fitur/label/threshold cocok dengan bundle model utama |
| Parity baru ONNX/native | 109 model × 32 vektor train = 3.488 skor; galat absolut maksimum 4,001013e-7; semua keputusan 0,5 sama |
| API fitur | Tiga vektor molekul sama persis dengan fitur penelitian; tujuh tes lokal API lulus |

AP diperiksa dengan perhitungan NumPy berdasarkan kelompok skor yang sama, dan AUROC menggunakan perbandingan ranking dengan setengah bobot untuk ties, terpisah dari helper metrik proyek. Ini menguji perhitungan terhadap anotasi tersimpan, bukan membuktikan kebenaran sensoris anotasi.

## Temuan yang perlu ditindaklanjuti

### 1. Status ONNX pada dokumentasi backend sudah usang

Lokasi yang masih menyatakan ditunda/belum selesai:

- `PROJECT_CONTEXT.md:9`
- `PROJECT_OVERVIEW.md:9`
- `IMPLEMENTATION_STATUS.md:9`
- `README.md:21`
- `KODE_ML/README.md:146`
- `KODE_ML/PANDUAN_SIDANG_KODE.md:50`
- bagian status terkini `SAFE_EXECUTION_PLAN.md:24,44` (bagian riwayat lama tidak perlu ditulis ulang).

Kondisi aktual: `../Essenza_Frontend/src/assets/metadata/v7_model_manifest.json` menunjuk LightGBM D yang benar; tersedia 109 ONNX dengan checksum cocok. Parity desktop diverifikasi kembali dalam audit ini. `../Essenza_Frontend/docs/VERIFICATION.md` mencatat demonstrasi Android fisik sebelumnya. LaTeX Bab 3/4/5 dan abstrak sudah menjelaskan API RDKit + inferensi ONNX Android.

**Perbaikan yang disarankan:** ubah status kini menjadi “integrasi prototipe tersedia dan telah diuji terbatas; produksi/hosting dan pengujian luas belum selesai.” Bedakan API lama pada repo ML dari `feature_api` aktif di repo frontend. Jangan mengubah metadata artefak historis untuk mengganti status saat artefak tersebut dibuat. Audit ini tidak menguji ulang HP.

### 2. Ada input identik dengan target berbeda

Dihitung langsung dari seluruh 2.048 bit Morgan dan delapan deskriptor yang digunakan model:

- 638 kelompok fitur identik mencakup 1.520 molekul.
- 616 kelompok tersebut, mencakup **1.474 molekul**, memiliki set target 109 label yang berbeda di antara anggotanya.
- 605 kelompok konflik mempunyai struktur yang sama setelah stereo dihapus; 11 kelompok masih mempunyai struktur nonstereokimia berbeda.
- Tidak ada kelompok fitur identik yang melintasi train–test.
- Pada train terdapat 489 kelompok konflik / 1.180 molekul; pada test 127 kelompok / 294 molekul.

Contoh: `C(=C/C1OCCO1)/c1ccccc1` memiliki target cinnamon/spicy/warm, sedangkan `C(=C\C1OCCO1)/c1ccccc1` memiliki cinnamon/spicy/sweet. Vektor fitur keduanya sama pada representasi sekarang.

**Makna:** model deterministik yang hanya menerima fitur tersebut tidak dapat mengeluarkan dua profil berbeda untuk input yang sama. Ini merupakan batas representasi dan ketidakkonsistenan anotasi terhadap representasi, bukan otomatis bukti anotasi kimia salah. Sebagian perbedaan dapat terkait stereokimia atau perbedaan pencatatan sumber; penyebab sensoris tidak ditentukan oleh audit ini. Tidak semua 1.474 molekul otomatis salah diprediksi.

**Tindakan:** tambahkan besaran ini dalam keterbatasan fitur/data Bab 4, misalnya setelah pembahasan split/fitur dan pada keterbatasan generalisasi. Pertahankan hasil yang sudah dikunci. Menambah chirality, menyatukan target, atau mengubah deduplikasi akan menjadi eksperimen baru dan tidak dilakukan dalam audit ini.

### 3. Pelaporan harmonisasi label belum mencakup audit token terbaru

Audit label 26 Oktober sudah menyimpan hasil yang dapat ditelusuri:

- 45.689 kemunculan istilah mencapai mapper: 37.940 exact, 1.255 alias, 6.494 tidak terpetakan.
- Istilah tak terpetakan tersebut terdiri atas 498 bentuk ternormalisasi berbeda.
- **3.898 baris sumber yang tetap diterima** mempunyai sebagian token yang tidak ikut sebagai target.
- 31 alias manual tidak sama dengan 184 istilah IFRA maupun 109 target terpilih.
- Setelah seleksi 109 target, ada 11 baris train dan 4 baris test yang seluruh targetnya nol.

LaTeX Bab 4 sudah benar menjelaskan 552 baris yang ditolak karena tidak punya label terpetakan. Namun, angka itu **tidak mencakup hilangnya sebagian token pada baris yang masih diterima**. Penjelasan harmonisasi belum tervalidasi pakar sudah ada, tetapi hitungan token terbaru di atas belum dicantumkan.

**Tindakan:** lengkapi rekonsiliasi label pada Bab 3/4 dari `KODE_ML/hasil_audit_label/summary.json`, bukan mengubah kamus diam-diam. Nol tetap berarti anotasi tidak tercatat, bukan bukti aroma tidak ada. Audit tidak mengesahkan akurasi semantik semua alias.

### 4. Batas input aplikasi belum konsisten antarmode

Reproduksi lokal tanpa menjalankan server:

- `resolve(ResolveRequest(query='Air'))` mengembalikan SMILES `O`, `prediction_supported=False`.
- `feature(FeatureRequest(smiles='O', feature_schema_id=...))` mengembalikan `status='ok'` dan 2.056 fitur.
- UI pencarian nama memeriksa flag tersebut (`AnalysisScreen.js:115–118`), tetapi mode SMILES memakai pemeriksaan teks `src/domain/analysis.js:7`; API `/fingerprint` memeriksa struktur dan skema, belum menyamakan aturan domain tersebut (`feature_api/app.py:195,223`).

**Dampak:** penghalang contoh non-aroma di mode nama belum berlaku konsisten pada jalur SMILES langsung. SMILES valid juga tidak membuktikan bahwa molekul berada dalam domain bahan pewangi yang dipelajari. Ini tidak mengubah evaluasi dataset beku, tetapi perlu dicatat sebelum menyatakan aplikasi membatasi seluruh input secara konsisten.

**Tindakan pada pekerjaan aplikasi berikutnya:** sepakati satu aturan kelayakan domain di layanan fitur untuk seluruh mode; bedakan struktur valid dari prediksi yang didukung model. Jangan menganggap daftar hitam air/CO2 sudah merupakan validasi domain penuh. Frontend tetap read-only pada audit ini.

### 5. Detail kecil penjelasan TPE dan kesiapan kernel

- `src/2.Isi/Bab3.tex:183` menyebut trial setelah resep awal ditentukan TPE. Perlu diperjelas bahwa startup masih menggunakan pengambilan acak sampai jumlah trial selesai memenuhi `n_startup_trials=10`. Kode Optuna terpasang mengonfirmasi perilaku ini. XGBoost hanya menyelesaikan 12 trial per studi; pemenangnya trial 2, masih fase awal. Ini sah disebut pencarian dengan Optuna/konfigurasi TPESampler, tetapi bukan bukti resep terbaik XGBoost dihasilkan fase adaptif TPE atau bahwa TPE mengungguli random search.
- Environment ML lokal belum mempunyai IPython/ipykernel/nbclient/nbconvert. Output notebook bisa dibaca; untuk menjalankan sel, ikuti pemasangan `KODE_ML/requirements_belajar.txt` pada README. Tidak ada dependency yang dipasang atau diubah dalam audit ini.

## Batas penelitian yang sudah dijelaskan dengan benar

- Test pernah dilihat pada pengembangan sebelumnya; repeated CV bukan nested CV. Hasil eksploratori.
- Seed/fold saling berkaitan; standar deviasi lima seed bukan interval kepercayaan independen.
- Kelompok mencegah identitas yang ditentukan protokol melintasi partisi; bukan seluruh kemiripan/scaffold.
- Lima sumber tidak membuktikan semua molekul hanya digunakan dalam parfum.
- AP bukan accuracy. Label accuracy LightGBM D 0,923660; prediksi semua nol mencapai 0,960970, dengan recall/F1 nol.
- LightGBM D dipilih berdasarkan AP, bukan pemenang semua metrik: macro F1 pada 0,5 = 0,293156, sedangkan LightGBM B = 0,320721.
- Skor belum terkalibrasi dan bukan intensitas aroma. Tidak ada validasi sensoris atau bukti produksi.
- Tidak ada klaim keunggulan langsung atas Suh karena data, label, split, dan evaluasi berbeda.
- Optuna terbaik dalam anggaran: XGB C/D masing-masing 12 trial, LGBM C 104 dan D 74; LGBM D terbaik pada trial terakhir selesai. Bukan optimum global atau bukti konvergensi.

## Rekomendasi keputusan

1. Pertahankan kode, parameter, data, split, dan model utama; tidak ada temuan audit yang mengharuskan membatalkan hasil atau melakukan tuning ulang.
2. Finalkan dokumentasi dengan status aplikasi yang benar dan temuan data/label di atas. Laporan sudah memuat banyak batasan utama; tidak perlu merombak metodologi agar hasil terlihat lebih tinggi.
3. Tutup ketidakkonsistenan batas input dalam pekerjaan aplikasi terpisah sebelum memperluas klaim demonstrasi.
4. Belajar dari delapan notebook boleh dimulai sekarang. Notebook dan model yang tersimpan bukan berarti seluruh riset sudah tervalidasi secara sensoris.

## Cakupan dan keterbatasan audit

Tidak melakukan fitting, trial baru, pembersihan data, perubahan label, ekspor ulang ONNX, build APK, deployment, atau kompilasi LaTeX. Notebook tidak dieksekusi ulang penuh; bukti fresh-kernel 26 Oktober digunakan kembali setelah source dan output tersimpan diperiksa. Audit backend tidak mencakup seluruh tes legacy yang melakukan fitting. Demonstrasi HP ditinjau dari bukti terdahulu; parity baru berjalan pada desktop CPU. Tidak melakukan validasi pakar terhadap seluruh anotasi atau audit ulang semua isi literatur.

Tes API awal dalam sandbox tertahan pada TestClient, dihentikan, lalu lulus di luar sandbox; tidak ada bukti kegagalan fungsi API dari percobaan tertahan tersebut. Pytest backend lulus dengan warning deprecation dependency dan pesan finalizer konsol Windows saat shutdown; keduanya bukan kegagalan asersi/metrik.

Bukti numerik: `independent_checks.json`, `data_and_mobile_checks.json`, `onnx_parity.json`, `domain_boundary.json`, `method/method_evidence.json`, dan `test_status.json`. Skrip pemeriksa disertakan untuk penelusuran; semuanya hanya membaca proyek dan menulis evidence di folder skripnya sendiri. Implementasi penelitian tetap `KODE_ML/research/`.
