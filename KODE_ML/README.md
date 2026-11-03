# Mulai dari PDF ini

**Buka [PANDUAN_BELAJAR_DARI_NOL.pdf](PANDUAN_BELAJAR_DARI_NOL.pdf), lalu ikuti halaman 1-18.**
Panduan mengarahkanmu ke nama notebook dan judul bagian yang harus dibaca. Tidak perlu membaca seluruh folder sekaligus.

| Tahap | Halaman PDF | Notebook yang dibuka |
|---|---|---|
| Masalah dan penelitian terdahulu | 2 | Belum perlu notebook |
| Dataset | 3 | 01_Sumber_Data.ipynb |
| Preprocessing dan label | 4 | 02_Preprocessing_dan_Audit_Label.ipynb |
| Morgan dan delapan deskriptor | 5 | 03_Dataset_dan_Fitur.ipynb |
| Grup, train/test, fold | 6 | 04_Grup_Split_dan_Fold.ipynb |
| Model dan bobot | 7-8 | 05_Model_dan_Bobot.ipynb |
| Tuning, CV, OOF | 9-11 | 06_Optuna_CV_dan_OOF.ipynb |
| Evaluasi dan hasil | 12-16 | 07_Evaluasi_dan_Pemilihan.ipynb; bagian threshold di notebook 06 |
| Prediksi dan persiapan sidang | 17-18 | 08_Prediksi_Molekul.ipynb |

**Mau langsung ke bagian kode tertentu?** Lihat [Peta_Notebook.md](01_Materi/Peta_Notebook.md).
Setiap kali membuka notebook mandiri, baca **Persiapan**, lalu bagian yang ditunjuk PDF. Output tersimpan dapat dibaca tanpa menjalankan sel.

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

Laporan utama berada di `C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael`; `SKRIPSI.tex` menyertakan Bab1-Bab5. **05_Referensi/Laporan adalah snapshot lama**, bukan naskah aktif. PDF baru diperiksa terhadap laporan utama dan kode aktual, bukan snapshot tersebut.

Prototipe Android memakai layanan fitur RDKit lokal dan 109 model ONNX pada perangkat. Integrasi tersedia dan diuji terbatas; dokumentasi historis yang menyebut ONNX ditunda tidak mewakili status kini.

Panduan masuk lama yang tumpang tindih sudah digabung ke PDF ini. Bukti audit dan materi lama yang masih berguna disimpan di **05_Referensi**, terpisah dari jalur belajar aktif. [Catatan penataan](05_Referensi/Arsip_Penataan_20261103/README.md) mencatat apa yang dipindah/dihapus. Kode, notebook, data, hasil eksperimen, dan model tidak diubah.
