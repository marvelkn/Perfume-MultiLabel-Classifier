# Mulai dari PDF ini

**Buka [PANDUAN_BELAJAR_DARI_NOL.pdf](PANDUAN_BELAJAR_DARI_NOL.pdf), lalu ikuti jalur inti.**
Panduan dan delapan notebook tetap menjadi satu jalur belajar aktif. Revisi 6 November 2026 membedakan kedalaman bacaan; tidak perlu membaca seluruh fungsi secara berurutan.

| Penanda | Target belajar |
|---|---|
| **Inti** | Jelaskan keputusan, alasan, output, dan batasnya; kerjakan satu contoh kecil bila relevan. |
| **Rujukan fungsi** | Cukup tahu tujuan dan lokasi bukti. Versi, helper, penyimpanan, log, checksum, dan pengaman tetap tersedia. |
| **Pendalaman** | Buka untuk penelusuran rinci, bukan syarat menyelesaikan alur inti. |

| Tahap | Halaman PDF | Notebook yang dibuka |
|---|---|---|
| Dasar ilmiah, gap, dan kontribusi penelitian | 2-3 | Belum perlu notebook |
| Masalah dan penelitian terdahulu | 4 | Belum perlu notebook |
| Pemilihan XGBoost-LightGBM dan batas tanpa RF | 5-6 | Belum perlu notebook |
| Dataset | 7 | 01_Sumber_Data.ipynb |
| Preprocessing dan label | 8 | 02_Preprocessing_dan_Audit_Label.ipynb |
| Morgan dan delapan deskriptor | 9 | 03_Dataset_dan_Fitur.ipynb |
| Grup, train/test, fold | 10 | 04_Grup_Split_dan_Fold.ipynb |
| Model dan bobot | 11-12 | 05_Model_dan_Bobot.ipynb |
| Tuning, CV, OOF | 13-15 | 06_Optuna_CV_dan_OOF.ipynb |
| Evaluasi dan hasil | 16-20 | 07_Evaluasi_dan_Pemilihan.ipynb; bagian threshold di notebook 06 |
| Prediksi dan persiapan sidang | 21-22 | 08_Prediksi_Molekul.ipynb |
| Peta 33 referensi skripsi: fungsi, halaman tercetak/PDF, dan batas klaim | 23-34 | Tidak perlu notebook |

**Tambahan 9 November 2026:** PDF sekarang berjumlah **34 halaman**. Materi inti tetap halaman 1-22; peta referensi berada pada halaman 23-34. Semua 33 referensi aktif disitasi. Nomor halaman skripsi mengacu pada PDF 94 halaman yang diperiksa dan dapat berubah setelah tata letak direvisi. Versi teks: [PETA_REFERENSI_SKRIPSI.md](05_Referensi/PETA_REFERENSI_SKRIPSI.md).

**Mau langsung ke bagian kode tertentu?** Lihat [Peta_Notebook.md](01_Materi/Peta_Notebook.md).
Untuk belajar dari output tersimpan, ikuti **Prioritas membaca** di awal setiap notebook dan bagian bertanda **Inti**. Jika akan menjalankan sel, baca **Persiapan** lebih dulu. Bagian campuran menjelaskan temuan yang tetap inti serta rincian yang cukup dirujuk.

**Latihan inti:** bobot kelas, keputusan threshold, TP/FP/FN/TN, precision/recall/F1, AP, macro AP, dan mean fold/seed. **Pendalaman hitung:** akar SD serta enumerasi banyak pasangan AUROC. Makna SD/AUROC tetap perlu dipahami; lima seed bukan lima dataset independen dan SD bukan interval kepercayaan.

## Isi folder

| Folder | Kapan diperlukan |
|---|---|
| [01_Materi](01_Materi/README.md) | Peta bagian notebook yang dirujuk PDF |
| [02_Data](02_Data/README.md) | Melihat data mentah, bersih, dan siap training |
| [03_Kode](03_Kode/README.md) | Implementasi aktif yang ditampilkan notebook |
| [04_Hasil](04_Hasil/README.md) | Memeriksa angka asli, parameter, dan hasil audit |
| [05_Referensi](05_Referensi/README.md) | Paper, snapshot laporan, dan arsip dokumentasi lama |
| [06_Audit_Label](06_Audit_Label/README.md) | Menelusuri pemetaan istilah aroma |

`belajar.py`, `audit_labels.py`, dan `asal_kohor.py` adalah helper yang dipakai notebook; jangan dihapus. Folder fisik source adalah **03_Kode**, sedangkan nama import tetap **research** melalui penghubung di root backend.

## Saat perlu menjalankan notebook

Gunakan kernel `.venv` penelitian proyek. Versi ML mengikuti `requirements_research.txt` di root backend. `requirements_belajar.txt` hanya berisi alat notebook. Bila alat tersebut belum terpasang, ikuti file requirements itu pada pekerjaan setup tersendiri.

Notebook boleh dibuka sendiri dengan kernel baru. **Run All bukan training penelitian secara default**: ada pembacaan hasil, audit, perhitungan fitur/metrik, serta inferensi model tersimpan. Pada notebook 07, biarkan **JALANKAN_EKSPERIMEN_BARU = False**. Tidak perlu training ulang untuk belajar.

## Pegangan hasil dan sumber

Hasil utama dikunci: lima sumber, 6.686 molekul, 109 label, Morgan 2.048 bit + delapan deskriptor untuk B/D. **LightGBM D dipilih dari mean macro AP validasi 0,299731; AP test 0,310531.** Angka AP bukan accuracy. Target nol berarti anotasi tidak tercatat; hasil eksploratori karena test pernah diakses dan CV bukan nested CV.

Laporan utama berada di `C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael`; `SKRIPSI.tex` menyertakan Bab1-Bab5. **05_Referensi/Laporan adalah snapshot lama**, bukan naskah aktif. Rujukan implementasi terbaru mengikuti laporan utama dan dokumentasi aplikasi lokal; snapshot tersebut tidak menjadi acuan arsitektur aktif.

**Arsitektur aplikasi per 7 November 2026:** Android mengirim SMILES ke komputer melalui USB; RDKit menghitung 2.056 fitur dan **109 model LightGBM D asli (joblib)** menjalankan inferensi pada komputer yang sama. Android menerima skor, menampilkan label, dan menyimpan Riwayat/Koleksi. Prediksi baru tetap memerlukan komputer, tetapi tidak membutuhkan hosting/internet setelah setup lokal selesai. Notebook 08 memperlihatkan inferensi Python; pengujian Android harus dibaca dari bukti aplikasi. Gunakan [panduan aplikasi lokal](../../Essenza_Frontend/docs/LOCAL_DEVELOPMENT.md) untuk implementasi terbaru. Dokumen audit lama adalah bukti bertanggal, bukan spesifikasi runtime saat ini. Pengujian teknis belum membuktikan validitas sensori, manfaat pengguna, atau kesiapan produksi.

Panduan masuk lama yang tumpang tindih sudah digabung ke PDF ini. Bukti audit dan materi lama yang masih berguna disimpan di **05_Referensi**, terpisah dari jalur belajar aktif. [Catatan penataan](05_Referensi/Arsip_Penataan_20261103/README.md) mencatat apa yang dipindah/dihapus. Penataan 3 November tersebut tidak mengubah kode, notebook, data, hasil eksperimen, atau model. Revisi 6 November hanya menambahkan prioritas baca pada narasi notebook; sel kode dan output tersimpannya tetap sama.

Pembaruan arsitektur 7 November ini hanya mengoreksi narasi implementasi. Sel kode dan output notebook, data, model, angka eksperimen, serta protokol tetap dipertahankan.
