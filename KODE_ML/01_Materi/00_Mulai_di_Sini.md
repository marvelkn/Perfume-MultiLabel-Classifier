# 00. Mulai belajar dari sini

Kamu nggak perlu langsung paham semua kode. Target awalnya cukup: kamu bisa cerita apa yang masuk, apa yang dikerjakan, dan apa yang keluar. Setelah alurnya nyambung, baru kita bongkar satu bagian setiap sesi.

Isi penelitian diperiksa pada **27 Oktober 2026**; nama folder dan tautannya dirapikan pada **28 Oktober 2026**. Hasil utama yang dibahas adalah arsip eksperimen utama lima sumber, Morgan 2048 dan delapan deskriptor. Nama arsip V7 bukan nama metode. Eksperimen tambahan V8/V9 tidak dicampur ke tabel hasil utama.

## Satu cerita yang perlu kamu pegang

Kamu memasukkan SMILES satu molekul. RDKit membaca strukturnya lalu mengubahnya menjadi angka. Untuk model terpilih, angkanya berupa **2048 bit Morgan + 8 deskriptor = 2056 fitur**. Angka ini masuk ke 109 classifier LightGBM, masing-masing untuk satu label aroma. Tiap classifier berisi banyak pohon. Keluar 109 skor; aturan ambang mengubah skor menjadi daftar label.

Model belajar dari pasangan fitur dan anotasi yang sudah tersedia. Jadi, ini prediksi anotasi aroma berdasarkan struktur, bukan alat yang mencium molekul secara langsung. Angka 90% pada tampilan adalah skor 0,90 yang dikali 100, bukan akurasi 90% dan belum terbukti sebagai peluang sensoris yang terkalibrasi.

Contoh nyata yang akan kita ikuti adalah vanillin: `COc1cc(C=O)ccc1O`. Vanillin sudah ada dalam dataset. Contoh ini membantu menelusuri alur dan tidak dipakai sebagai bukti kemampuan pada molekul baru.

## Urutan belajar yang masuk akal

Satu sesi bisa 30-45 menit. Kalau masih bingung, ulangi satu sesi; nggak perlu mengejar semua bab dalam sehari.

| Sesi | Buka materi | Setelah itu kamu bisa menjelaskan |
|---|---|---|
| 1 | 01 Dasar dan penelitian terdahulu | Masalah penelitian dan hubungannya dengan penelitian acuan |
| 2 | 02 Molekul, fitur, dan target | Perbedaan SMILES, X, y, skor dan label |
| 3 | 03 Dataset dan sumber | Bentuk data sebelum dibersihkan dan cara menyambung identitas |
| 4 | 04 Preprocessing | Baris ditolak, label dirapikan, molekul digabung |
| 5 | 05 Grup, split dan fold | Kenapa data dibagi dan bagaimana mencegah irisan grup |
| 6 | 06 Model dan proses belajar | 109 classifier, boosting, loss dan bobot |
| 7 | 07 Optuna dan validasi | Satu trial, CV, OOF dan training final |
| 8 | 08 Evaluasi dan hasil | Arti metrik dan alasan pemilihan LightGBM D |
| 9 | 09 Menjalankan dan prediksi | Membuka notebook serta memakai model tersimpan |
| 10 | 10 Latihan sidang dan kamus | Menjelaskan alur tanpa membaca catatan |

Kenapa teori dulu? Kalau langsung lihat `fit()` sebelum tahu apa itu X dan y, kode terasa seperti kumpulan perintah acak. Setelah tahu tujuan tiap langkah, kamu bisa menghubungkan baris kode dengan pekerjaan yang nyata.

## Cara menggunakan folder ini

Folder `01_Materi/` berisi versi per bab dari panduan ini. Baca materi konsep dulu, lalu buka notebook yang disebutkan. Source asli ada di **KODE_ML/03_Kode/**. Nama import Python tetap **research** lewat penghubung di root proyek. Restart kernel notebook setelah perubahan lokasi folder. Notebook memanggil dan menampilkan fungsi dari source itu; kamu tidak sedang belajar algoritma pengganti.

| Yang kamu cari | Buka dari KODE_ML |
|---|---|
| Urutan belajar dan progres | MULAI_BELAJAR.md dan CHECKLIST_BELAJAR.md |
| Panduan lengkap untuk dibaca | PANDUAN_BELAJAR_DARI_NOL.pdf |
| Penjelasan tiap bab | 01_Materi/00 sampai 10 |
| Praktik tahap demi tahap | Delapan notebook bernomor 01 sampai 08 |
| Implementasi aktif | `03_Kode/*.py` dan `03_Kode/config/*.json` |
| Data mentah, bersih dan siap training | 02_Data/README.md |
| Angka hasil dan parameter tersimpan | 04_Hasil/README.md |
| Paper acuan dan daftar bacaan | 05_Referensi/README.md |
| Pemeriksaan istilah dan label | 06_Audit_Label/README.md |
| Cuplikan source dengan baris dan hash | 01_Materi/11_Peta_Source_dan_Cuplikan.md |

Nama notebook panjang `BELAJAR_KODE_ML_LENGKAP.ipynb` sudah menjadi arsip. Yang aktif sekarang delapan notebook. Urutannya untuk belajar, bukan kewajiban menjalankan notebook 01 dulu agar notebook 08 bisa berjalan.

## Membaca kode tanpa tenggelam

Untuk satu fungsi, jawab empat hal: **inputnya apa, proses utamanya apa, outputnya apa, dan dipakai siapa setelahnya**. Baru baca baris per baris. Sel `tampilkan_sumber(...)` memberi lokasi file, baris dan kode aktual. Catat variabel yang belum kamu pahami; jangan menebak dari namanya saja.

File `belajar.py` cuma membantu menampilkan source. `audit_labels.py` memeriksa pemetaan label. `asal_kohor.py` menelusuri asal pembagian data historis. Ketiganya tidak berarti training sedang berjalan.

## Patokan progres

Kamu boleh lanjut kalau bisa menjelaskan dengan kata sendiri, memberi satu contoh, dan menunjuk bukti di file. Kalau cuma bisa mengulang istilah seperti 'boosting' tanpa tahu yang dijumlahkan apa, kembali ke contoh kecilnya.

Latihan awal: ceritakan alur vanillin dalam lima kalimat. Jangan pakai kata 'akurasi' untuk satu skor aroma. Contoh jawaban: 'SMILES dibaca menjadi struktur. Struktur diubah menjadi fitur. Setiap classifier memberi skor untuk satu label. Skor dibandingkan dengan ambang. Beberapa label bisa lolos bersamaan.'

Pertanyaan dosen: 'Apa sebenarnya yang dipelajari model?' Jawaban singkat: 'Hubungan antara fitur struktur molekul dan anotasi aroma pada dataset. Prediksinya bergantung pada kualitas anotasi dan representasi fitur yang digunakan.'

Checklist: [ ] bisa cerita input-proses-output; [ ] tahu folder source; [ ] tahu hasil utama yang dipakai; [ ] bisa membedakan contoh dan hasil nyata.
