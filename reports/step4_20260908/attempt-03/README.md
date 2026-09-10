# Step 4 attempt 03 — STOP sebelum smoke

Pengguna mengizinkan **Step 4 sampai Step 5** melalui “oke lanjut terus sampai step 5 juga”. Baseline baru selesai, tetapi smoke real-data belum boleh dimulai: suhu akhir **84.875°C**, sedangkan syarat mulai **<75°C**. Step 5 sudah diizinkan dan persiapannya tercatat, tetapi belum selesai. Step 6 dan seterusnya belum diizinkan.

| Pengukuran | Hasil |
| --- | --- |
| Rekaman | 2026-09-08T21:08:53.410799+07:00 hingga 2026-09-08T21:19:03.407614+07:00 |
| Durasi collector | 610 detik; tanpa fitting |
| Suhu awal / minimum / maksimum / akhir | 73.375 / 71.750 / 85.625 / 84.875°C |
| Rata-rata suhu sampel sumber unik | 78.940°C |
| Suhu 5 menit terakhir | 79,625–85,625°C |
| Sampel sumber unik / observasi collector | 304 / 608 |
| Sampel sumber di bawah 75°C | 115; rentang berturut-turut terpanjang hanya 204 detik |
| Umur sumber maksimum / gap sumber maksimum | 2.674 / 3 detik |
| CPU sistem rata-rata / 60 observasi terakhir | 20,93% / 38,24% |
| Peak RSS collector | 21.09 MiB; bukan peak seluruh aplikasi |
| RAM sistem bebas minimum | 12.52 GiB |
| Fitting nyata / budget probe terpakai | 0 label-fit / 0 dari 300 detik |

Pada 21:17:52 WIB, ResourceGuard menolak mulai pada **85,125°C** sebelum dataset dimuat. Penurunan suhu sementara ke bawah 75°C tidak bertahan selama 5 menit. Seluruh observasi cocok dengan arsip CSV; sensor berubah, timestamp berurutan, dan tidak stale. Ini memvalidasi pencatatan dan penolakan preflight, belum memvalidasi callback saat fitting nyata.

Power plan Acer bertipe Balanced dan charger terhubung. Sampel beban latar selama 10 detik mencatat rsEngineSvc.exe sekitar 9,04% kapasitas CPU total. Pengukuran itu tidak menetapkan penyebab panas. Chat, monitor dan layanan latar aktif; suhu ruangan dan kecepatan kipas tidak diukur. Tidak ada layanan keamanan, BIOS, daya, atau batas temperatur yang diubah.

Collector selesai exit 0 dan telemetry menjadi STOP, yang ditolak ResourceGuard. Monitor PID 29908 milik attempt ini telah ditutup dengan pemeriksaan nama serta waktu mulai. CSV diarsipkan setelah penutupan dan cocok checksum sumber.

Validasi ulang lulus: **85 berkas protokol**, **3 berkas runner**, Python dan **14 dependency**, **29 bukti kedua attempt sebelumnya**, serta **53 berkas laporan**. Frontend bersih dan read-only. Tidak ada run operasional, pemuatan matriks development/test, training, atau perubahan isi LaTeX. Tes tidak diulang karena kode tidak berubah pada attempt ini; bukti 8 tes sintetis runner tetap berlaku, dengan batasan belum ada fitting nyata. Tidak ada commit/push oleh sesi ini; HEAD tercatat pada preflight dan validation.

Bukti utama: [thermal-validation.json](thermal-validation.json), [guard-preflight.json](guard-preflight.json), [validation.json](validation.json), [completion.json](completion.json), rekaman [baseline.jsonl](baseline.jsonl) dan [summary](baseline.summary.json), [sensor-source.csv](sensor-source.csv).

Lanjutkan dari pemeriksaan termal baru setelah kondisi berubah dan suhu memiliki margin di bawah 75°C. Jika kondisi perangkat tetap tidak memenuhi batas, gunakan resource lain setelah environment dan pengaman diselaraskan. Setelah smoke lengkap serta cooldown GO, teruskan [Step 5](../../step5_20260908/README.md) sesuai izin yang sudah diberikan. Jangan menggunakan suhu baseline untuk mengarang runtime trial atau menaikkan batas temperatur.
