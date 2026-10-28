# Panduan sidang kode — metode utama

Perapian 26 Oktober 2026 menempatkan implementasi aktif di `research/` dalam KODE_ML, bersama delapan notebook belajar. Fungsi inti, fitur, target, grup, split, dan model hasil utama dipertahankan. Kode checkpoint lebih panjang karena eksperimen berlangsung puluhan jam; itu alat menjaga proses, bukan metode ML baru.

Untuk belajar bertahap dari nol, buka [MULAI_BELAJAR.md](MULAI_BELAJAR.md) dan [panduan lengkap PDF](PANDUAN_BELAJAR_DARI_NOL.pdf).

## Mana yang perlu dikuasai?

| Kategori | Bagian | Yang harus bisa dijelaskan |
|---|---|---|
| Sudah wajar | CSV, NumPy/pandas, RDKit, pustaka boosting | Menggunakan library bukan membuat boosting dari nol |
| Perlu dipahami | Join identitas, kamus IFRA, penggabungan label | Mengapa dua record bisa menjadi satu molekul |
| Perlu dipahami | Grup, fold per label, seed, OOF | Bagian mana boleh memengaruhi keputusan model |
| Perlu dipahami | Bobot kelas, AP, F1, threshold | Apa yang diukur dan apa keterbatasannya |
| Pendukung rekayasa | Hash, SQLite, lock, budget | Mencegah input berubah dan menghindari pencampuran run |
| Sudah dirapikan | Import berantai V5/V6, preflight 1024+5 | Sekarang memakai modul fungsi dan fitur utama |

## Satu contoh yang harus bisa kamu ikuti

SMILES vanilin → RDKit membaca atom/ikatan → Morgan 2048 + delapan angka → `features('lgbm_D',...)` menghasilkan `(1,2056)` → model sweet memberi satu skor → model woody memberi skor lain → 109 skor dibandingkan dengan ambang 0,5. Skor bukan intensitas dan tidak harus berjumlah 100%.

Pada training, `y[:, j]` mengambil target satu aroma dari semua molekul. `fit_final_models` mengulang untuk 109 aroma. Setiap classifier berisi banyak pohon; `n_estimators` mengatur putaran boosting. Total delapan kandidat berarti 872 model final, bukan 872 pohon.

## Pertanyaan dan jawaban latihan

| Pertanyaan dosen | Jawaban singkat | Pertanyaan lanjutan yang perlu siap |
|---|---|---|
| Input model apa? | Angka fitur RDKit. SMILES adalah masukan sebelum ekstraksi fitur. | Jelaskan bentuk `(n,2048)` dan `(n,2056)`. |
| Apa deskriptor dipilih otomatis? | Tidak. Delapan jenis ditetapkan; RDKit menghitung nilainya. | Mengapa angka jumlah cincin bukan bit? |
| Kenapa 2048? | Keputusan representasi eksperimen; bukan jaminan paling optimal. | Jelaskan hashing dan kemungkinan collision. |
| Kenapa data 13 ribu menjadi 6 ribu? | Ada penolakan dan penggabungan catatan molekul sama. Bukan semuanya data tanpa label dibuang. | Bedakan 1.560 ditolak dan 11.777 diterima sebelum merge. |
| Berapa data akhir? | 6.686 molekul: 5.353 train, 1.333 test; 109 label. | Mengapa proporsi baris bukan tepat 80:20? Pembagian berbasis grup. |
| Label 0 artinya tidak berbau itu? | Tidak tercatat setelah pemetaan, belum tentu negatif yang diuji sensoris. | Bagaimana missing annotation memengaruhi evaluasi? |
| Kenapa label berbeda bisa memakai fold berbeda? | Stratifikasi dilakukan per target biner; grup tetap dijaga. | Untuk label dan seed sama, semua kandidat memakai fold sama. |
| Grup dibentuk dari 2048? | Tidak. Grup historis dari struktur tanpa stereo atau Morgan1024 identik dipertahankan. | Verifikasi sekarang memakai hash/identitas/split beku, bukan regroup. |
| Seed 7 berarti tujuh kali? | Tidak. Itu nilai awal pengacakan. Ada lima nilai seed, masing-masing sampai lima fold. | Estimator tetap random_state42; seed CV mengubah fold. |
| OOF itu test20%? | Bukan. OOF adalah prediksi pada bagian validasi di dalam train. | Setiap baris diprediksi model yang tidak melatih baris itu. |
| Kenapa final fit lagi? | Model fold untuk penilaian. Model final belajar semua train setelah pilihan dikunci. | Test tidak ikut final fit. |
| Apa Optuna selalu menemukan terbaik? | Terbaik yang ditemukan di ruang/budget ini, bukan optimum global. | TPE memakai riwayat trial; empat studi × delapan jam. |
| Objective Optuna apa? | Rata-rata AP fold per label, lalu macro antar109 label. | Skor itu tidak sama persis dengan AP gabungan semua OOF. |
| Cara mengatasi kelas langka? | Bobot `n/(2*n_kelas)` dari bagian train saat itu. | Tidak memperbaiki anotasi salah atau menjamin probabilitas terkalibrasi. |
| Accuracy tinggi berarti bagus? | Belum tentu; banyak nol. Perlu AP, recall, precision dan macro F1. | Accuracy di sini rata-rata akurasi biner per label. |
| AUPRC di kode artinya apa? | Key `auprc` memakai sklearn Average Precision, bukan integral trapezoid PR. | AP menilai ranking, tidak bergantung ambang0,5. |
| Ambang0,5 dipakai buat apa? | Ubah skor menjadi keputusan positif/negatif. | Ambang tambahan dipilih dari OOF untuk F1 tiap label. |
| Mengapa B−A dan D−C? | B−A menguji tambahan fitur pada baseline sama. D−C membandingkan representasi dengan tuning masing-masing. | D−C tidak mengisolasi fitur dari resep tuning. |
| Apakah lebih baik dari Suh? | Tidak bisa disimpulkan langsung karena kohor/prosedur berbeda. Suh acuan teknis, kontrol utama antar kandidat kita. | Jelaskan repeated CV bukan nested CV dan test pernah dilihat. |
| Apakah refactor menjaga hasil? | Fitur, target, split dan prediksi model tersimpan sudah dibandingkan. Training ulang tidak dijalankan. | Tunjukkan laporan pemeriksaan dan batas klaimnya. |
| Apakah pakai AI? | Ya, untuk membantu implementasi; saya bertanggung jawab memahami serta memeriksa metode, data dan hasil. | Tunjukkan satu fungsi dan jelaskan sendiri alurnya. |

## Batas yang jangan dilewati saat menjawab

Jangan menyebut skor sebagai intensitas aroma; jangan mengklaim memprediksi campuran. Jangan menyebut seluruh sumber hanya berisi parfum. Jangan menyebut lima seed sebagai lima eksperimen independen untuk uji signifikansi. V9 tambahan tidak otomatis mengganti hasil utama. Prototipe API RDKit + 109 ONNX tersedia dan telah diuji terbatas; audit 27 Oktober memverifikasi parity desktop, bukan menguji ulang Android atau kesiapan produksi. Lihat [audit terbaru](HASIL_UNTUK_BELAJAR/audit_27_oktober/README.md).

Peta delapan notebook, kode, dan alamat hasil ada di [README](README.md).


## Tambahan hasil audit 26 Oktober 2026

**Dari mana 109 kelas?** Python membaca 184 istilah unik dari tiga kolom IFRA.
Sesudah outer split, hanya label dengan minimal 30 contoh positif dan 2 negatif training dipilih;
hasilnya 109. Daftar ini disimpan supaya urutan kelas tidak berubah.

**Apakah ALIASES berisi 109 label?** Tidak. Ada 31 pasangan istilah manual; 17 terpakai.
Tujuannya menangani variasi istilah, bukan menetapkan semua kelas model.

**Apakah semua istilah berhasil dipetakan?** Tidak. Ada 498 istilah berbeda/6.494 kemunculan
yang gagal pada baris yang mencapai mapper. Mereka dicatat dalam audit, tidak masuk label hasil.
Sebagian baris tetap diterima karena punya label lain yang cocok.

**Mengapa belum ditambahkan fuzzy?** Kesamaan ejaan belum menjamin kesamaan makna aroma.
Pemetaan baru perlu tinjauan dan eksperimen terpisah, bukan mengubah label dari hasil terkunci.

**Mengapa kode asli ada di dalam KODE_ML tetapi notebook masih import?** Agar notebook dan
runner memakai satu fungsi yang sama. Source fungsi itu ditampilkan utuh di output notebook,
dengan alamat/baris/hash, kemudian dipanggil. Tidak ada implementasi versi belajar yang berbeda.

**Apakah sudah membuktikan training ulang hasilnya sama?** Belum. Audit membuktikan struktur
kode/metode, data, fitur, grup/split/target, metrik dan prediksi yang diperiksa tetap konsisten;
training penuh tidak diulang dalam perapian ini.
