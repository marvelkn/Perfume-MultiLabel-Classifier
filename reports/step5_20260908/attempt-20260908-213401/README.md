# Step 5 — Persiapan snapshot dengan pemantauan suhu

**Persiapan snapshot selesai; Step 5 keseluruhan belum GO.** Pengguna meminta melanjutkan Step 5 sambil memantau suhu dan menghentikan pekerjaan jika kondisi pengaman terpicu. Tidak ada fitting model pada tahap ini.

Hasil yang dibuat adalah paket **calon run nonoperasional** pada [candidate/](candidate/):

- 47 berkas snapshot source, konfigurasi, protokol, metadata dataset, dan partisi; seluruh checksum cocok sumber.
- [candidate_run.json](candidate/candidate_run.json) menyimpan identitas dataset/schema, partisi frozen, environment, seed, serta profil final. Status PREPARED_NOT_EXECUTABLE dan training_allowed=false.
- [budget_pending.json](candidate/budget_pending.json) menyimpan cap 14.400 detik/model dan rumus jumlah trial. Runtime kedua learner dan jumlah trial masih null; ini bukan budget freeze atau ledger operasional.
- [snapshot_hashes.json](candidate/snapshot_hashes.json) mendeteksi perubahan snapshot. Tidak ada run.json atau run baru dalam runs/ yang dapat dipakai CLI training.

| Pemeriksaan | Hasil |
| --- | --- |
| Pembuatan snapshot | 21:37:19–21:37:21 WIB, 1.802 detik aktif |
| Suhu selama persiapan, dari sampel guard | 75.375°C awal/maksimum/akhir |
| Peak RSS proses persiapan | 28.02 MiB |
| RAM sistem bebas minimum selama persiapan | 9.54 GiB |
| Seluruh rekaman sensor | 180 detik, 21:34:21–21:37:21 WIB |
| Suhu seluruh rekaman | 74,625–83,125°C; akhir 75,375°C |
| Sampel sumber unik / observasi collector | 90 / 179 |
| Peak RSS collector | 20.89 MiB |
| RAM bebas minimum seluruh rekaman | 8.32 GiB |
| Kondisi stop selama persiapan | Tidak terpicu |

Guard persiapan memakai batas suhu safe-training yang sudah ada (mulai <80°C, stop 85°C/10 detik, hard limit 90°C), dengan durasi lokal dipersempit menjadi satu menit dan satu thread. Snapshot profil final tetap safe-training asli dua thread/20 menit. Tidak ada perubahan YAML atau pelonggaran batas. Pembacaan 82,625°C terjadi sebelum pekerjaan ini; saat guard memulai, suhu sudah turun ke 75,375°C.

Operasi berlangsung kurang dari dua detik; banyak pemeriksaan memakai sampel sumber CPU yang sama karena CSV diperbarui sekitar dua detik. Angka tersebut bukan bukti suhu konstan secara kontinu atau jaminan keamanan fitting panjang. Rekaman 180 detik ini juga bukan baseline baru 10 menit dan tidak mengubah status STOP Step 4.

Validasi lulus: 85 berkas praregistrasi, 47 snapshot, environment Python/dependency/source, partisi tersalin identik, checksum arsip CSV, kecocokan sampel guard dengan sumber, serta penolakan telemetry STOP setelah collector selesai. Kode model utama, frontend, dan isi LaTeX tidak diubah. Tidak ada matriks development/test yang dimuat; artefak dataset hanya dibaca sebagai byte untuk checksum. Partisi belum diregenerasi oleh initializer final. Tidak ada commit/push.

Collector selesai exit 0. Monitor milik sesi ini (PID 30952, mulai 21:34:01 WIB) ditutup setelah identitasnya diverifikasi. Tidak ada pemantauan latar yang terus berjalan setelah penutupan.

Untuk menyelesaikan Step 5: selesaikan smoke real-data serta cooldown, validasi runtime kedua learner, hitung jumlah trial sesuai rumus, regenerasi/verifikasi partisi, lalu buat run final beserta budget freeze dan ledger. Persiapan ini tidak menggantikan bukti tersebut. Permintaan menjalankan algoritme secara terpisah dengan cooldown perlu dicatat dalam amandemen runner sebelum probe berikutnya; belum diimplementasikan oleh pekerjaan metadata ini.

Bukti: [preparation-result.json](preparation-result.json), [thermal-validation.json](thermal-validation.json), [validation.json](validation.json), [preparation-resources.jsonl](preparation-resources.jsonl), dan [completion.json](completion.json).
