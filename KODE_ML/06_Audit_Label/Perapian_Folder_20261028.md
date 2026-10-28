# Perapian nama folder - 28 Oktober 2026

Folder utama sekarang: **01_Materi, 02_Data, 03_Kode, 04_Hasil, 05_Referensi, 06_Audit_Label**.

Di dalam data: **01_Data_Mentah, 02_Data_Bersih, 03_Siap_Training**. Salinan bab laporan ada di **05_Referensi/Laporan**. Nama sumber dan kandidat eksperimen tetap sebagai identitas arsip.

PDF, materi Markdown, tautan README, dan delapan notebook sudah disesuaikan. Import Python tetap `research`; penghubung di root mengarah ke 03_Kode. Restart kernel notebook setelah perubahan ini.

Pemeriksaan sebelum pemasangan:

- 103 tautan materi tidak putus.
- 125 sel kode pada delapan notebook dapat diparse.
- 72 tampilan source tersimpan cocok dengan kode dan hash file saat ini.
- 147 file CSV/JSON/NPZ/TeX/Bib tetap identik byte, termasuk konfigurasi dan bukti penelitian.
- Seluruh 37 halaman PDF diperiksa dari hasil render.
- Source ML tetap sama kecuali penyesuaian pemeriksaan lokasi source dan tes lokasi; rumus, parameter, fitur, target, grup dan split tidak berubah.
- Tujuh tes kontrak baca-saja lulus setelah pemasangan (15,11 detik), termasuk kesamaan prediksi terhadap arsip. Bootstrap import notebook dari KODE_ML juga berhasil.
- Tidak menjalankan training atau tuning; pemeriksaan inferensi memakai model tersimpan tanpa fit.

Manifest dan audit bertanggal lama sengaja menyimpan path pada tanggal asalnya. **../PETA_FOLDER.json** memetakan alamat tersebut ke lokasi sekarang beserta hash sebelum/sesudah. Backup sebelum perapian disimpan di workspace tugas.
