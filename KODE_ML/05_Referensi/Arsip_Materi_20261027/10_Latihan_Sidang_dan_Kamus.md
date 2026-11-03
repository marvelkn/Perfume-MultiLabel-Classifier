# 10 — Latihan jawab dosen pakai pemahaman sendiri

Status aplikasi terbaru ada di 04_Hasil/audit_27_oktober/README.md; catatan penundaan ONNX pada plan lama sudah usang.

**Target belajar:** jelaskan alur, tunjukkan bukti, sebutkan batasnya. Sesuaikan jawaban dengan pekerjaan yang benar-benar kamu lakukan.

Pegangan utama: [SAFE_EXECUTION_PLAN.md](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/SAFE_EXECUTION_PLAN.md>), [config penelitian](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/config/experiment.json>), serta kode aktif di `KODE_ML/03_Kode/`. Nama arsip V7 menunjukkan hasil yang dikunci, bukan nama algoritma baru. V9 adalah eksperimen tambahan.

## Putaran pertama: jelaskan alurnya

**1. Penelitianmu memprediksi apa?**

“Beberapa label aroma untuk satu struktur molekul. Satu molekul bisa punya beberapa label, jadi tugasnya multilabel.” Ini prediksi dari anotasi, bukan pengukuran aroma atau prediksi campuran parfum.

**2. Bentuk input dan outputnya apa?**

“Input berupa SMILES. Struktur diubah menjadi fitur angka, kemudian model menghasilkan 109 skor label.” Untuk LightGBM D, satu input berbentuk **1×2.056 fitur**, keluar **1×109 skor**. Jika ditampilkan persen, 0,7 menjadi 70%; itu tetap skor model yang belum dikalibrasi.

**3. Model membaca huruf SMILES langsung?**

“RDKit membaca struktur SMILES, lalu menghitung Morgan dan deskriptor.” Bukti: [features.py, calculate_morgan(), baris 29](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/features.py:29>) dan `calculate()` baris 19. Struktur tidak valid atau multikomponen ditolak.

**4. Kenapa 2.048+8?**

“Morgan merangkum pola lingkungan atom. Deskriptor menambah sifat molekul, misalnya berat molekul dan logP. Manfaat tambahannya diuji lewat perbandingan skenario.” Jangan bilang 2.048 pasti optimal dibanding semua ukuran.

**5. Apa arti label 0 dan 1?**

“Satu berarti aroma tercatat setelah pemetaan label. Nol berarti tidak tercatat.” Jangan menjawab “nol pasti tidak berbau itu”, karena anotasi sumber bisa belum lengkap dan label aroma bersifat subjektif.

**6. Berarti ada 109 pohon?**

“Ada 109 classifier biner untuk satu kandidat algoritma. Setiap classifier punya banyak pohon boosting.” XGBoost dan LightGBM dibandingkan sebagai dua pilihan algoritma. Satu prediksi dengan kandidat terpilih menggunakan 109 classifier kandidat itu. Lihat [training.py, fit_final_models(), baris 55](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/training.py:55>).

**7. Pohon-pohon itu bekerjanya gimana?**

“Pohon membagi data berdasarkan fitur. Dalam boosting, pohon berikutnya menambah koreksi terhadap prediksi gabungan sebelumnya berdasarkan loss.” Pembelajarannya menggunakan turunan loss, bukan sekadar menghitung tebakan salah.

**8. Kalau labelnya jarang muncul gimana?**

“Kode memberi bobot sampel berdasarkan frekuensi kelas pada bagian training yang sedang dipakai. Kelas langka mendapat bobot lebih besar.” Rumusnya **bobot kelas = N/(2×jumlah kelas itu)**; N adalah jumlah sampel training. Jika N=100 dan positif=10, bobot positif=5, negatif=100/180≈0,556. Bukti: [training.py, balanced_sample_weights(), baris 21](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/training.py:21>). Ini bobot saat menghitung loss, bukan membuat molekul baru.

## Putaran kedua: alasan eksperimen dan hasil

**9. A, B, C, D bedanya apa?**

| Skenario | Fitur | Parameter |
|---|---|---|
| A | Morgan 2.048 | Baseline |
| B | Morgan 2.048 + 8 deskriptor | Baseline |
| C | Morgan 2.048 | Hasil Optuna |
| D | Morgan 2.048 + 8 deskriptor | Hasil Optuna |

“Ada delapan kandidat. A lawan B memeriksa tambahan deskriptor; B lawan D memeriksa tuning dengan fitur sama.” Aturan bobot kelasnya sama.

**10. Optuna memilih berdasarkan apa?**

“Setiap trial mencoba satu kombinasi parameter. Nilainya dihitung dari rata-rata AP fold per label, lalu rata-rata seluruh 109 label pada data training. Nilai lebih tinggi lebih baik.” Empat studi C/D mendapat anggaran sama, delapan jam per studi. Bukan memilih parameter dari AP test. Detailnya di [tuning.py](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/tuning.py>).

**11. Train, validasi, test bedanya apa?**

“Training untuk mempelajari pola. Validasi untuk menilai pilihan saat pengembangan. Test untuk pelaporan setelah pilihan dibekukan.” Data utama berisi 6.686 molekul: 5.353 train dan 1.333 test. Jangan menyebut semua data dipakai melatih model final penelitian.

**12. Kenapa pakai grup? Ada jejak 1.024 bit?**

“Pembagian mempertahankan grup historis supaya anggota grup tidak menyeberang train dan test. Asal grup memakai kesamaan struktur tanpa stereo atau fingerprint Morgan 1.024 yang identik. Fitur model utama sekarang tetap Morgan 2.048.” Jangan bilang seluruh molekul mirip pasti terpisah; aturan ini hanya melindungi grup yang didefinisikan. [splits.py, validate_grouped_split(), baris 21](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/splits.py:21>) memeriksa pemisahannya.

**13. Seed itu apa, dan apakah semua label punya fold sama?**

“Seed mengendalikan pengacakan. Evaluasi memakai lima seed; fold dibuat per label memakai target dan grup, jadi antarlabel bisa berbeda.” Kode mengizinkan pengurangan fold saat kelas tidak mencukupi; audit hasil utama menemukan lima fold setiap label/seed. Lihat [splits.py, inner_folds(), baris 6](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/splits.py:6>).

**14. Kenapa LightGBM D dipilih?**

“Karena mean macro AP validasinya paling tinggi, 0,299731. AP test-nya 0,310531 dan AUROC test-nya 0,842985.” Tambahkan: D tidak tertinggi pada semua metrik; F1 ambang 0,5 LightGBM B lebih tinggi. Jangan mengganti aturan seleksi sesudah membaca test.

**15. Kenapa accuracy tinggi tetapi precision rendah?**

“Sebagian besar pasangan molekul–label bernilai nol. Banyak keputusan negatif yang benar bisa membuat accuracy tinggi. Precision menyorot prediksi positif, jadi mengungkap masalah yang bisa tertutup oleh accuracy.” Untuk D pada 0,5, macro accuracy 0,923660 tetapi precision 0,213862. Karena itu angka accuracy saja tidak cukup.

**16. Ambang OOF bikin semua metrik naik?**

“Tidak. Pada LightGBM D, macro F1 naik dari 0,293156 menjadi 0,330419, tetapi recall turun dari 0,515089 menjadi 0,411890. Ambangnya dipilih dari validasi OOF training, bukan test.” AP dan AUROC tetap karena urutan skor tidak diubah.

## Putaran ketiga: batas dan tanggung jawab

**17. Berarti model terbukti lebih baik dari penelitian sebelumnya?**

“Saya membatasi kesimpulan pada kandidat dan dataset penelitian ini. Perbedaan sumber data, label, pembagian, dan protokol membuat angka lintas penelitian belum tentu sebanding.” Test pernah diperiksa dalam riwayat pengembangan; repeated CV bukan nested CV. Selisih kecil dan simpangan baku lima seed belum membuktikan signifikansi statistik.

**18. Apakah fitur yang sama selalu punya label sama?**

“Tidak. Audit menemukan 616 kelompok fitur identik, meliputi 1.474 molekul, dengan target berbeda. Model deterministik tidak bisa menghasilkan dua profil berbeda dari fitur yang sama.” Sebanyak 605 kelompok konfliknya berstruktur sama setelah stereo dihapus; 11 masih berbeda. Tidak ada kelompok fitur identik melintasi train–test. Ini batas representasi/anotasi, bukan bukti semua molekul tersebut salah diprediksi.

**19. Kalau kode dibantu AI, kontribusimu apa?**

“Saya menggunakan AI untuk membantu penulisan dan penjelasan kode. Saya bertanggung jawab memahami rancangan, memeriksa implementasi, dan menilai hasilnya.” Sebutkan **pekerjaan yang benar-benar sudah dilakukan**, misalnya menelusuri fungsi atau menghitung metrik. Kalau belum dilakukan, jangan diklaim. Hafalan tidak menggantikan pemeriksaan kode.

**20. Aplikasinya sudah terbukti siap produksi?**

“Integrasi prototipe tersedia. Audit desktop mengecek 109 ONNX pada 32 vektor: selisih skor maksimum 4,001013×10⁻⁷ dan keputusan ambang 0,5 sama dengan native.” Bukti Android sebelumnya tersedia, tetapi audit ini tidak menguji ulang HP. Pengujian terbatas belum membuktikan kesiapan produksi atau validitas sensoris. Rujukan temuan dan batas terbaru: [audit 27 Oktober](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/04_Hasil/audit_27_oktober/README.md>).

## Kamus cepat

| Istilah | Arti gampangnya |
|---|---|
| SMILES | Teks untuk menuliskan struktur molekul |
| Fitur | Angka yang dibaca model sebagai masukan |
| Morgan | Representasi bit dari pola lingkungan atom |
| Deskriptor | Angka sifat molekul yang dihitung RDKit |
| Label | Nama aroma yang menjadi target |
| Multilabel | Satu data boleh memiliki beberapa label |
| Classifier biner | Model untuk satu target 0/1 |
| Boosting | Menambahkan pohon koreksi secara bertahap |
| Loss | Ukuran kesalahan yang diminimalkan saat belajar |
| Parameter model | Nilai yang dipelajari, misalnya nilai daun pohon |
| Hyperparameter | Pengaturan proses belajar, misalnya jumlah pohon |
| Trial | Satu percobaan kombinasi hyperparameter |
| Fold | Satu bagian pembagian validasi silang |
| Seed | Angka pengendali pengacakan |
| OOF | Prediksi baris ketika baris itu menjadi validasi |
| Threshold/ambang | Batas skor untuk memutuskan 0/1 |
| Macro | Rata-rata metrik per label dengan bobot sama |
| Kalibrasi | Kesesuaian skor probabilitas dengan frekuensi kejadian nyata |
| Leakage | Informasi evaluasi bocor ke proses belajar/pemilihan |
| Overfitting | Pola training dipelajari terlalu khusus sehingga sulit berlaku pada data baru |

## Checklist: siap karena bisa melakukan

- [ ] Gambar alur SMILES → fitur → 109 classifier → skor → ambang → label.
- [ ] Tunjukkan fungsi asli pembuat fitur, pembobotan, pelatihan, dan metrik.
- [ ] Bedakan 109 classifier dari jumlah pohon masing-masing classifier.
- [ ] Hitung precision, recall, F1, dan AP contoh kecil tanpa bantuan kode.
- [ ] Temukan angka D dan B di CSV asli, lalu jelaskan kelebihan dan kekurangannya.
- [ ] Jelaskan kenapa skor 70% bukan accuracy 70% atau jaminan aroma ada.
- [ ] Sebutkan batas label nol, grup historis, dan test yang pernah dilihat.
- [ ] Akui bagian yang belum dipahami; jangan mengarang langkah yang belum dikerjakan.

Latihan: rekam jawaban nomor 2, 6, 14, dan 17 masing-masing 30–60 detik tanpa membaca. Cek ke kode dan artefak: apakah penjelasanmu punya bukti?