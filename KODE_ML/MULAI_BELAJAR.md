# Mulai dari sini, pelan-pelan

**Folder sudah dirapikan 28 Oktober 2026:** 01_Materi, 02_Data, 03_Kode, 04_Hasil, 05_Referensi, 06_Audit_Label. Baca penjelasan di 01_Materi; data di 02_Data; implementasi asli di 03_Kode.

Setelah pembaruan lokasi, restart kernel notebook. Nama import Python tetap research; folder fisiknya 03_Kode.

Buka **[panduan lengkap PDF](PANDUAN_BELAJAR_DARI_NOL.pdf)**. Untuk hari pertama, cukup pengantar dan bab 01. Kamu belum perlu menjalankan Python.

1. [00 - Cara belajar dan peta file](01_Materi/00_Mulai_di_Sini.md)
2. [01 - Masalah dan penelitian terdahulu](01_Materi/01_Dasar_dan_Penelitian_Terdahulu.md)
3. [02 - Molekul, fitur dan target](01_Materi/02_Molekul_Fitur_dan_Target.md)
4. [03 - Bentuk data awal](01_Materi/03_Dataset_dan_Sumber.md)
5. [04 - Preprocessing dan label](01_Materi/04_Preprocessing_dan_Audit_Label.md)
6. [05 - Grup, split dan fold](01_Materi/05_Grup_Split_dan_Fold.md)
7. [06 - Model dan hitungan sederhana](01_Materi/06_Model_dan_Proses_Belajar.md)
8. [07 - Optuna, CV, OOF dan training final](01_Materi/07_Optuna_Validasi_dan_Training_Final.md)
9. [08 - Evaluasi dan hasil asli](01_Materi/08_Evaluasi_dan_Hasil_Penelitian.md)
10. [09 - Membuka notebook dan prediksi](01_Materi/09_Menjalankan_Project_dan_Prediksi.md)
11. [10 - Latihan sidang dan kamus](01_Materi/10_Latihan_Sidang_dan_Kamus.md)
12. [11 - Peta source dan cuplikan aktual](01_Materi/11_Peta_Source_dan_Cuplikan.md)
13. [12 - Library dan membaca baris kode](01_Materi/12_Library_dan_Cara_Membaca_Kode.md)

Pakai [checklist progres](CHECKLIST_BELAJAR.md). Setelah konsep nyambung, ikuti notebook 01-08. Notebook boleh dibuka mandiri dengan kernel baru. Output yang sudah disimpan bisa dibaca tanpa memasang library.

**Yang kamu perlukan sudah dipetakan di folder ini:**

- [Data mentah, bersih dan siap training](02_Data/README.md).
- [Salinan angka hasil dan parameter](04_Hasil/README.md).
- [Paper dan bab laporan acuan](05_Referensi/README.md).
- Source aktif **03_Kode/**. Notebook memakai fungsi yang sama, bukan versi algoritma berbeda.

Materi diperiksa 27 Oktober 2026. Prototipe Android sudah tersedia dan diuji terbatas; status lama yang menyebut ONNX ditunda kalah baru dari [audit 27 Oktober](04_Hasil/audit_27_oktober/README.md). Training/model/data tidak diubah saat menyusun bahan ini. Model berukuran besar tetap dimuat notebook dari lokasi aslinya melalui 03_Kode/results.py, jadi kamu tidak perlu mencarinya manual.
