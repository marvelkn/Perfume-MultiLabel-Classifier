# Step 4 — Pemeriksaan termal sebelum smoke

**Status terbaru: [attempt 03](attempt-03/README.md)** tetap STOP sebelum fitting; baseline 610 detik, suhu akhir 84,875°C. Step 4–5 sudah diizinkan. [Step 5](../step5_20260908/README.md) tertahan sampai smoke GO. Bagian berikut mempertahankan hasil dan cakupan izin attempt pertama pada waktu itu.

**STOP sebelum fitting.** Rekaman CPU baru selama **610 detik** tidak memenuhi syarat mulai safe-smoke **<75°C**. Tidak ada label-fit, run smoke/final, tuning, pemuatan matriks dataset, atau evaluasi test yang dijalankan. Budget aktif probe terpakai **0 dari 300 detik**.

## Hasil pengukuran

| Pengukuran | Hasil |
| --- | --- |
| Rekaman | 2026-09-08T11:34:44.416810+07:00 hingga 2026-09-08T11:44:54.412695+07:00 |
| Suhu awal / minimum / maksimum / akhir | 85.125 / 76.625 / 85.125 / 77.125°C |
| Rata-rata suhu sampel sumber unik | 81.931°C |
| Sampel sumber unik / observasi collector | 306 / 608 |
| Sampel di bawah 75°C | **0** |
| Gap sumber maksimum / umur maksimum | 3.000 / 2.655 detik |
| CPU sistem rata-rata / rata-rata 60 observasi akhir | 20.80% / 14.55% |
| Peak RSS collector | 21.07 MiB |
| RAM sistem bebas minimum | 10.74 GiB |
| RAM monitor pada satu pengamatan | 57.09 MiB; bukan peak |

Durasi collector memakai waktu monotonik 610 detik. Rentang timestamp sumber adalah 611 detik karena waktu CSV dibulatkan per detik dan sampel pertama dapat berasal dari sebelum collector mulai; ini bukan tambahan durasi fitting. Semua observasi cocok dengan arsip CSV sumber. Sensor memakai Core (Tctl/Tdie), /amdcpu/0/temperature/2, pada Ryzen 5 5500U dan cocok dengan identitas yang diverifikasi Step 1. Tidak ada screenshot UI baru.

## Guard dan beban latar

Pada 11:36:39 WIB, ResourceGuard menolak preflight pada **84°C** dengan alasan `CPU too hot to start; wait for cooling below the start threshold`. Pemeriksaan ini tidak memuat dataset atau melatih model. Ini membuktikan penolakan mulai dalam kondisi panas; perilaku callback ketika fitting nyata berlangsung masih belum diuji di Step 4.

Dua sampel terpisah selama sekitar 11 dan 15 detik mencatat `rsEngineSvc.exe` sebesar **16,14%** dan **13,34%** dari kapasitas CPU total; CPU sistem saat itu 25,2% dan 22,9%. Windows mengaitkannya dengan **Reason Security Engine Service**. Path/version executable tidak tersedia lewat pembacaan proses. Ini adalah bukti beban latar, bukan diagnosis tunggal penyebab panas atau bukti layanan bermasalah.

Charger terhubung pada pengamatan; skema daya bernama Acer. Kesetaraannya dengan profil Balanced tidak diverifikasi. Tidak ada layanan dihentikan, antivirus dinonaktifkan, atau pengaturan daya/BIOS diubah. Chat, monitor, collector, serta layanan latar tetap aktif; pengukuran ini bukan idle laboratorium mendekati 0% CPU. Suhu ruangan, kecepatan kipas, dan kondisi ventilasi tidak diukur ulang secara independen.

## Validasi dan preservasi

- Skrip [analyze_thermal_preflight.py](analyze_thermal_preflight.py) telah dijalankan pada rekaman lengkap; Ruff lulus tanpa cache setelah perbaikan dua penggunaan pairwise. Percobaan lint pertama tidak dapat menulis cache pada sandbox.
- Seluruh **85 checksum protokol** tetap cocok; source produksi, tes, dan YAML tidak berubah.
- **53 berkas laporan lama** cocok dengan backup; frontend tetap bersih dan read-only. Tidak ada run operasional baru.
- Collector selesai dengan exit 0 dan menginvalidasi telemetry menjadi STOP; ResourceGuard menolaknya.
- Monitor yang dimulai sesi ini sudah dihentikan. Penutupan biasa tidak berhasil karena proses berjalan sebagai administrator; helper dengan hak yang sama menghentikan hanya PID dan waktu mulai yang telah diverifikasi. Arsip CSV dibuat setelah proses berhenti dan cocok hash sumber.
- Tidak ada commit/push atau perubahan isi LaTeX. Tes ML tidak diulang karena source/tes tidak berubah.

## Artefak

- [thermal-validation.json](thermal-validation.json): statistik dan validasi sumber.
- [pre-smoke-idle-01.jsonl](pre-smoke-idle-01.jsonl) dan [summary](pre-smoke-idle-01.summary.json): seluruh observasi collector.
- [sensor-source-20260908.csv](sensor-source-20260908.csv): arsip CSV aktual; mencakup waktu sebelum/sesudah baseline, sementara statistik di atas hanya memakai interval collector.
- [guard-preflight-01.json](guard-preflight-01.json) dan [log](guard-preflight-01.jsonl): penolakan preflight nyata.
- [background-load-01.json](background-load-01.json), [background-load-02.json](background-load-02.json), [top-process-identity.json](top-process-identity.json): sampel beban latar dan identitas layanan.
- [host-observation.json](host-observation.json), [monitor-shutdown.json](monitor-shutdown.json), [validation.json](validation.json), [completion.json](completion.json): metadata, cleanup, preservasi, dan hasil akhir.

## Langkah berikutnya

Step 4 belum GO. Ulangi pemeriksaan termal setelah kondisi CPU memiliki margin di bawah 75°C dan sensor valid; batas suhu tetap. Probe terdaftar pada [PREREGISTRATION.md](../../PREREGISTRATION.md) baru boleh dijalankan jika syarat mulai terpenuhi. Jika kondisi lokal tetap tidak memenuhi batas, pilih resource lain dengan amandemen environment sebelum menjalankan eksperimen. Jangan memakai baseline ini untuk menebak waktu trial. **Step 5, jumlah trial final, dan training utama belum dijalankan atau diizinkan.**

## Pembaruan setelah attempt awal

8 September 2026: [attempt 02](attempt-02/README.md) tetap STOP termal (76,375–80,500°C selama 120 detik). [Runner smoke](SMOKE_RUNNER.md) sudah disiapkan; delapan tes sintetis dan validate-only lulus. Statistik di atas tetap merupakan attempt awal, bukan digabung dengan rekaman berikutnya.
