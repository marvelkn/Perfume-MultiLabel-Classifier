# Rencana penyelarasan Suh: fitur, tuning, validasi, dan label

Tanggal: 9 September 2026  
Status: **DRAF UNTUK DIBACA PENGGUNA — belum dijalankan.**

## 1. Keputusan yang disarankan

Tujuan penelitian: menguji apakah penambahan lima deskriptor pada Morgan dan tuning Optuna meningkatkan kemampuan XGBoost/LightGBM memprediksi label aroma, dibandingkan implementasi acuan pada data dan evaluasi yang sama.

Output tetap skor prediksi dan label aroma untuk satu molekul. Yang diperbaiki adalah ketepatan prediksinya. Mengganti arti label atau memilih label yang mudah bukan bukti model menjadi lebih baik.

| Bagian | Rekomendasi | Alasan |
| --- | --- | --- |
| Dataset | Dataset kurasi penulis acuan, bukan gabungan lima sumber lama | Mengurangi penyebab perbedaan selain modifikasi yang diuji |
| Label | Daftar dan pemetaan acuan, setelah masalah jumlah label diselesaikan | Jumlah sama saja tidak cukup; arti dan target tiap molekul juga harus sama |
| Fitur | Morgan acuan dibandingkan Morgan yang sama + lima deskriptor | Menguji kontribusi tambahan informasi |
| Tuning | Optuna untuk kedua algoritma, dengan budget dan aturan yang ditetapkan di awal | Pencarian parameter terarah dan dapat diaudit |
| Validasi | Lima fold sebagai protokol utama | Menjaga keselarasan dengan acuan |
| Evaluasi | Enam metrik acuan; AUPRC sebagai usulan ukuran utama | Tetap sebanding dan memperhatikan label langka |
| Tambahan opsional | Pembobotan kelas positif, diuji terpisah | Ada dasar penelitian untuk ketidakseimbangan kelas |
| Klaim | Laporkan membaik, memburuk, atau belum meyakinkan sesuai hasil | Tidak menjanjikan semua metrik meningkat |

## 2. Sumber data yang dituju

Sepuluh nama sumber yang disebut Suh:

1. Arctander
2. AromaDb
3. FlavorDb (odor)
4. FlavorNet
5. The Good Scents Company Information System (TGSC)
6. IFRA Fragrance Ingredient Glossary
7. Leffingwell
8. Sharma_A
9. Sharma_B
10. Sigma Fragrance & Flavor Catalog

Sumber: [Suh et al. (2025), bagian Dataset](https://www.nature.com/articles/s42004-025-01651-7).

Ini nama dalam artikel, bukan daftar path folder yang sudah diverifikasi. Pemetaan arsip Pyrfume dan revisinya masih menunggu kode/data penulis. Jangan menebak folder Sharma_A dan Sharma_B.

Prioritas adalah mengambil **hasil kurasi penulis dari OSF**, bukan sekadar mengunduh sepuluh sumber terbaru. Versi data, penggabungan identitas molekul, dan pemetaan istilah dapat menghasilkan target yang berbeda meskipun nama sumber sama.

Jika data kurasi atau indeks pembagian asli tidak tersedia, lakukan adaptasi terdokumentasi dengan satu dataset dan satu pembagian yang dipakai semua eksperimen. Saat itu, pembandingan utama adalah baseline yang kita jalankan ulang; angka artikel hanya konteks, bukan bukti reproduksi persis.

Mengganti lima sumber lama dengan data acuan berarti klaim kontribusi “menggabungkan lima dataset” perlu dilepas dari eksperimen utama. Lima sumber lama masih dapat didokumentasikan sebagai pekerjaan sebelumnya. Tidak perlu menambah eksperimen dataset lain untuk tujuan utama saat ini.

## 3. Modifikasi yang memiliki dasar dan layak diuji

| Modifikasi | Dasar yang ditemukan | Batas bukti dan keputusan |
| --- | --- | --- |
| Morgan + lima deskriptor | Menggabungkan pola struktur lokal dengan sifat molekul; ini hipotesis representasi fitur proyek | Belum ada bukti bahwa kombinasi lima ini pasti meningkatkan hasil. Masuk eksperimen utama |
| Tuning Optuna | Akiba et al. menjelaskan framework pencarian hyperparameter; Anghel et al. menguji optimasi Bayesian pada XGBoost dan LightGBM di beberapa dataset | Mendukung penggunaan tuning, bukan jaminan Optuna selalu menang. Masuk eksperimen utama |
| Pembobotan kelas positif | Wang et al. menguji weighted loss dan focal loss untuk XGBoost pada data biner tidak seimbang | Hasilnya bergantung dataset/metrik. Weighted loss layak menjadi eksperimen tambahan |
| Pembobotan pada LightGBM | Parameter scale_pos_weight tersedia dalam dokumentasi resmi | Dukungan implementasi bukan bukti empiris dari studi Wang untuk LightGBM |
| Focal loss | Termasuk modifikasi dalam penelitian Wang et al. | Membutuhkan loss khusus dan pemeriksaan lebih banyak; ditunda |
| Threshold per label | Bisa mengubah kompromi precision–recall | Tidak memperbaiki AUROC/AUPRC jika skor tetap; bukan eksperimen utama |
| Ensemble atau classifier chains | Menambah perubahan model dan kebutuhan validasi | Ditunda agar kontribusi fitur dan tuning bisa dijelaskan dengan jelas |

[Anghel et al., Benchmarking and Optimization of Gradient Boosting Decision Tree Algorithms](https://arxiv.org/abs/1809.04559) adalah paper workshop/preprint, bukan bukti kemenangan universal atau perbandingan langsung Optuna melawan default. Studi ini menunjukkan bahwa hasil serta biaya pencarian XGBoost/LightGBM berbeda antardataset.

[Akiba et al. (2019), Optuna](https://arxiv.org/abs/1907.10902) menjadi rujukan alat tuning dan strategi pencarian.

[Wang, Deng, dan Wang (2020), Imbalance-XGBoost](https://doi.org/10.1016/j.patrec.2020.05.035), Pattern Recognition Letters; [naskah penulis](https://arxiv.org/abs/1908.01672). Penelitian ini mendukung weighted/focal loss untuk XGBoost biner tidak seimbang. Tidak semua metrik naik bersama; hasil di data tersebut belum membuktikan hasil di data aroma.

[Dokumentasi LightGBM 4.6.0](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#scale_pos_weight) juga mengingatkan bahwa pembobotan dapat memengaruhi kualitas estimasi probabilitas.

### Lima deskriptor yang dipertahankan

Daftar kode saat ini, src/featurize.py:

| Deskriptor | Makna sederhana |
| --- | --- |
| MolWt | Berat molekul |
| MolLogP | Kecenderungan larut dalam minyak dibanding air |
| NumHDonors | Jumlah donor ikatan hidrogen |
| NumHAcceptors | Jumlah penerima ikatan hidrogen |
| TPSA | Luas permukaan polar |

RDKit menghitung Morgan maupun deskriptor ini. Karena itu, sebut modifikasi sebagai **penggabungan Morgan dan lima deskriptor**, bukan sekadar “menggunakan RDKit”.

Lima fitur dipertahankan sebagai rancangan ringkas yang telah dipilih sebelum melihat hasil baru. Angka lima bukan angka optimum yang sudah dibuktikan. Jangan memilih ulang fitur memakai skor test.

Spesifikasi Morgan, penanganan molekul, versi pustaka, dan urutan fitur harus dibekukan. Radius, jumlah bit, serta chirality acuan belum boleh diasumsikan sama dengan konfigurasi lama. A dan B harus berbeda hanya pada lima kolom tambahan.

## 4. Eksperimen utama: empat kondisi untuk setiap algoritma

| ID | Fitur | Parameter |
| --- | --- | --- |
| A | Morgan acuan | Pengaturan acuan yang terverifikasi |
| B | Morgan yang sama + lima deskriptor | Sama dengan A |
| C | Morgan yang sama | Optuna |
| D | Morgan yang sama + lima deskriptor | Optuna |

Terapkan A–D pada XGBoost dan LightGBM: total delapan kondisi utama.

- A vs B: pengaruh tambahan fitur tanpa tuning.
- A vs C: pengaruh tuning pada Morgan.
- B vs D: pengaruh tuning pada fitur gabungan.
- C vs D: perbandingan fitur setelah masing-masing mendapat tuning dengan aturan sama.
- A vs D: pengaruh paket modifikasi lengkap.

C vs D bukan efek fitur pada parameter tetap, karena parameter hasil tuning bisa berbeda.

Strategi penanganan ketidakseimbangan, threshold, data, dan evaluasi harus sama pada A–D. Jangan sekaligus memasukkan pilihan pembobotan atau resampling ke pencarian utama: itu menambah penyebab perubahan hasil.

**Opsi E**, jika nanti dipilih sebelum evaluasi test: D ditambah pembobotan positif, dengan parameter D lainnya tetap. Hitung bobot hanya dari bagian training fold. Bandingkan D vs E. Jika bobot dituning, laporkan sebagai eksperimen tambahan beserta budget terpisah. Jangan menjadikan E otomatis wajib.

## 5. Rancangan Optuna yang dapat dipertanggungjawabkan

Usulan: sampler TPE dengan seed tetap, satu studi per algoritma per representasi fitur, dan tujuan rerata AUPRC antarfold pada label yang memenuhi aturan evaluasi. Detail perhitungan AUPRC harus diperiksa dari kode acuan sebelum dikunci.

Parameter calon pencarian:

| XGBoost | LightGBM | Tujuan |
| --- | --- | --- |
| max_depth, min_child_weight | num_leaves, max_depth, min_child_samples | Mengatur kerumitan pohon |
| learning_rate, jumlah boosting | learning_rate, jumlah boosting | Mengatur proses belajar |
| subsample, colsample_bytree | bagging_fraction, feature_fraction | Mengatur pemakaian sampel/fitur |
| reg_alpha, reg_lambda | lambda_l1, lambda_l2 | Membatasi model agar tidak terlalu mengikuti data latihan |

Rentang numerik belum dikunci. Tentukan setelah konfigurasi acuan dan ukuran data tersedia; untuk LightGBM, selaraskan batas num_leaves/max_depth dan aktifkan bagging_freq bila memakai bagging_fraction.

Aturan wajib sebelum menjalankan:

1. Fold, objective, ruang pencarian, seed, timeout, serta kebijakan pruning dicatat dahulu.
2. Kandidat dibandingkan menggunakan fold dan kumpulan label yang sama.
3. Jika memakai early stopping, sediakan bagian validasi internal dari training fold; bagian penilaian fold tidak dipakai menentukan kapan berhenti.
4. Test tidak dipakai memilih parameter, fitur, bobot, atau threshold.
5. Simpan semua trial: skor, waktu, kegagalan, alasan pruning, dan parameter terbaik.
6. Kurva best-so-far membantu melihat perkembangan pencarian, tetapi plateau bukan bukti optimum global.

**Budget:** keputusan lama adalah empat jam komputasi aktif per algoritma. Untuk rancangan C dan D, usulan awal pembagiannya **dua jam C + dua jam D per algoritma**, bukan otomatis empat jam untuk setiap studi. Waktu baseline, final fit, dan analisis tambahan dicatat terpisah. Pembagian ini masih usulan untuk ditinjau, bukan konfigurasi yang sudah diterapkan.

Jika ukuran data/label baru membuat budget terlalu kecil untuk percobaan yang bermakna, laporkan keterbatasannya dan revisi anggaran sebelum studi utama. Jangan mengurangi label agar terlihat berhasil. Jumlah trial selesai tidak harus sama antaralgoritma pada budget waktu yang sama; catat keduanya.

## 6. Pilihan fold: lima untuk eksperimen utama

**Rekomendasi: 5-fold sekali sebagai protokol utama**, mengikuti acuan. Setiap bagian bergiliran menjadi validasi; rata-rata dan sebaran skor dilaporkan. Test terpisah tetap disimpan sampai seluruh pilihan model selesai.

Alasan memilih lima:
- Menjaga pembandingan dengan metode acuan.
- Menggunakan lima pelatihan per kandidat per label, dibanding sepuluh pada 10-fold.
- Tidak mengklaim lima selalu paling akurat atau paling stabil.

[Kohavi (1995)](https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf) mendukung stratified 10-fold pada konteks eksperimen C4.5/Naive Bayes mereka. Itu alasan 10-fold layak dipertimbangkan, **bukan bukti 5-fold salah atau 10-fold pasti terbaik untuk proyek ini**.

Prioritaskan indeks dan stratifikasi penulis asli. Bila indeks asli tidak diperoleh, usulkan iterative stratification untuk menjaga distribusi beberapa label pada pembagian bersama. Dasarnya [Sechidis, Tsoumakas, dan Vlahavas (2011)](https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10). Perubahan ini harus disebut adaptasi, bukan reproduksi identik.

Label sangat langka tetap bisa tidak memiliki dua kelas dalam fold. Catat dukungan positif/negatif dan aturan label yang bisa dihitung; stratifikasi tidak menyelesaikan semua kelangkaan.

**Apakah perlu trial-and-error jumlah fold?** Tidak untuk memilih angka dengan skor tertinggi. Itu mencampur pemilihan cara menilai dengan keinginan memperoleh skor bagus.

Jika perlu pemeriksaan tambahan, pilih sebelum melihat test:
- Ulangi 5-fold dengan tiga pembagian/seed yang telah ditetapkan dan laporkan semuanya. Angka tiga adalah kompromi biaya proyek, bukan angka wajib dari jurnal.
- Atau bandingkan 3/5/10-fold pada data pengembangan dan konfigurasi yang sama, lalu laporkan skor, variasi, waktu, serta cakupan label; jangan hanya mengambil skor tertinggi.

Tambahan tersebut opsional. Mengulang CV pada konfigurasi yang sudah dipilih hanya memeriksa sensitivitas terhadap pembagian; bukan nested CV dan bukan estimasi bebas bias dari seluruh proses tuning. Skor fold juga tidak independen, sehingga jangan langsung memakai uji t biasa seolah-olah lima eksperimen terpisah.

[Cawley dan Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html) menjelaskan risiko optimisme dari pemilihan model menggunakan estimasi validasi. Karena itu, kesimpulan utama harus menggunakan test yang tidak ikut pemilihan.

## 7. Metrik yang disamakan

Enam metrik acuan: **Accuracy, AUROC, AUPRC, Specificity, Precision, Recall**. Laporkan CV dan test secara terpisah.

Kesamaan nama belum cukup:

| Detail | Rencana |
| --- | --- |
| Accuracy | Hitung biner per label; subset accuracy seluruh label bukan pengganti |
| AUPRC | Verifikasi Average Precision atau integral kurva PR dari kode penulis |
| Rata-rata | Verifikasi cara agregasi antarfold dan antarlabel |
| Threshold | Ikuti aturan acuan; jangan menganggap 0,5 tanpa bukti |
| Label tidak dapat dinilai | Tampilkan NA dan alasan; samakan aturan serta cakupan label antar kondisi |
| Precision/Recall/Specificity | Catat penanganan pembagi nol |
| F1, Hamming loss, metrik tambahan | Boleh sebagai pelengkap, bukan pengganti enam metrik |

Usulan ukuran utama adalah **AUPRC**, karena sasaran proyek mencakup label yang jarang muncul. Ukuran lain tetap dilaporkan. Jika AUPRC naik tetapi accuracy atau precision turun, tulis komprominya; jangan menyebut “semua performa lebih baik”.

Definisi objective dan cara agregasi akan ditetapkan sebelum tuning, setelah kode acuan diperiksa. Bila kode tidak tersedia, dokumentasikan konvensi sendiri dan batasi klaim kesamaan dengan angka artikel.

## 8. Label: mengikuti acuan, bukan mempertahankan 25

Berdasarkan audit lampiran resmi yang sudah disimpan:
- Ada 200 nama termasuk OTHERS pada setiap kombinasi model/fitur.
- Tidak semua memiliki nilai lengkap: contohnya 188 accuracy_cv dan 166 auroc_test per kombinasi.
- Teks artikel menyebut taksonomi 201. Perbedaannya masih perlu dijelaskan lewat data/kode.

Bukti lokal: supplement_audit.json, label_names_from_supplement.json, dan Supplementary_Data_1.xlsx dalam folder ini.

Rekomendasi:
1. Gunakan nama, arti, dan pemetaan target acuan; daftar 200 hasil adalah petunjuk awal, bukan keputusan final matriks training.
2. Jangan menafsirkan 188 sebagai batas jumlah label training tanpa memeriksa penyebab nilai kosong.
3. Pertahankan daftar penuh pada laporan, tandai label yang tidak dapat dilatih/dievaluasi.
4. Tentukan aturan kelayakan dari kode acuan dan ketersediaan kelas, bukan skor yang bagus.
5. Untuk membandingkan rerata dua model, gunakan cakupan label yang sama per metrik dan cantumkan jumlahnya.
6. Jangan mengubah nama/merge label atau memperlakukan label tanpa anotasi sebagai negatif terverifikasi tanpa mencatat asumsi itu.

Memakai 25 label akan mengubah tugas dan tingkat kesulitannya. Itu boleh untuk eksperimen terbatas, tetapi tidak mendukung klaim bahwa versi kita mengungguli acuan secara keseluruhan.

## 9. Apa yang cukup untuk mengatakan modifikasi membantu?

Rencana bukti:
- Baseline A dijalankan ulang, bukan hanya mengambil angka artikel.
- Semua kondisi memakai sampel, target, dan pembagian yang sama.
- Deduplikasi identitas molekul dilakukan sebelum pembagian untuk mencegah molekul yang sama masuk training dan test.
- Seluruh konfigurasi dibekukan sebelum evaluasi test. Tidak ada tuning ulang setelah melihat hasil test.
- Laporkan skor A–D, selisihnya, hasil per label, cakupan label, variasi CV, dan waktu komputasi.
- Nyatakan pembandingan utama A vs D terlebih dahulu; perbandingan lainnya menjelaskan sumber perubahan.
- Kenaikan kecil pada satu pembagian belum cukup untuk menyebut peningkatan signifikan. Pemeriksaan ketidakpastian tambahan harus direncanakan sebelum melihat hasil, dengan metode yang sesuai untuk prediksi berpasangan pada molekul yang sama.
- Jika hasil turun, kesimpulan “modifikasi ini tidak memperbaiki hasil pada protokol yang diuji” tetap sah.

Contoh rumusan skripsi:

> Penelitian ini mengadaptasi pendekatan prediksi aroma berbasis struktur molekul dengan XGBoost dan LightGBM. Pengujian dilakukan untuk menilai pengaruh penggabungan Morgan fingerprint dengan lima deskriptor molekuler serta optimasi hyperparameter menggunakan Optuna pada data, label, dan prosedur evaluasi yang diselaraskan dengan penelitian acuan.

## 10. Urutan setelah rencana disetujui — belum dilaksanakan

1. Peroleh data kurasi dan kode penulis; periksa izin penggunaan serta versi sumber.
2. Selesaikan daftar label, parameter Morgan/model, pembagian data, dan definisi metrik.
3. Finalkan protokol, rentang Optuna, budget, serta pilihan eksperimen tambahan.
4. Sesuaikan preprocessing dan notebook audit dengan data acuan.
5. Siapkan baseline A dan kondisi B–D, lalu verifikasi kesamaan data/evaluasi.
6. Jalankan eksperimen dan tuning sesuai protokol yang dibekukan.
7. Evaluasi test, analisis selisih serta keterbatasan, dan simpan model yang dipilih dari data pengembangan.
8. Perbarui laporan dan paket inference berdasarkan hasil aktual.

**Status akses:** data kurasi dan kode OSF belum diperoleh; pemeriksaan sebelumnya mengalami timeout saat daftar berkas. Ini belum terselesaikan.

**Status proyek:** dokumen ini saja yang diperbarui pada langkah perencanaan ini. Dataset lama 25 label, konfigurasi training, notebook, model, laporan LaTeX, frontend, dan paket kampus belum dimigrasikan ke rancangan ini. Panduan/paket kampus lama belum mewakili rencana baru.
