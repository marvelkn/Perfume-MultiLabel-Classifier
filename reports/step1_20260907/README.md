# Step 1 — GO untuk sensor dan baseline; STOP untuk kesiapan training

PawnIO telah dipasang setelah persetujuan eksplisit pengguna. Sensor CPU nyata, pencocokan dengan aplikasi monitor, dan pembacaan oleh ResourceGuard sudah terverifikasi. Baseline direkam setelah pengguna menyatakan menghentikan aktivitas berat, menggunakan meja keras, dan membuka ventilasi.

**GO ini menutup Step 1 saja.** Seluruh sampel baseline berada di atas batas stop safe-smoke 80°C. Workload training belum boleh dimulai dalam kondisi termal yang terukur ini; Step 2 merupakan pekerjaan kode/profil dan tes ringan yang masih menunggu perintah tersendiri.

## Hasil baseline

| Ukuran | Hasil |
| --- | ---: |
| Waktu pengukuran | 7 September 2026, 14:12:04–14:22:14 WIB |
| Durasi collector | 610 detik |
| Rentang timestamp sumber | 609 detik |
| Sampel sumber unik / observasi collector | 305 / 608 |
| Suhu awal | 83,875°C |
| Suhu minimum / maksimum | 80,125 / 85,875°C |
| Suhu rata-rata / median | 83,654 / 83,875°C |
| Suhu akhir | 81,000°C |
| Gap pembaruan sumber maksimum | 3 detik |
| Umur telemetry maksimum saat dibaca | 2,994 detik |
| Beban CPU sistem rata-rata | 17,23% |
| Beban CPU rata-rata 60 observasi terakhir | 12,21% |
| Peak RSS collector | 21,07 MiB |
| RAM sistem bebas minimum | 9,87 GiB |

Suhu diringkas dari sampel sumber unik agar polling ulang sampel yang sama tidak menggandakan bobot. CPU persen diringkas dari observasi collector, dengan observasi pertama dikeluarkan karena interval belum penuh.

Ini adalah baseline **tanpa aktivitas berat pengguna**, dengan layanan latar dan aplikasi chat/monitor tetap aktif. Bukan kondisi idle laboratorium mendekati 0% CPU. Suhu ruangan dan kecepatan kipas tidak diukur. Peak RAM monitor selama baseline tidak direkam; nilai sebelum baseline tersedia sebagai pengamatan terpisah dan tidak dianggap peak seluruh sesi.

## Verifikasi

- Sensor: `Core (Tctl/Tdie)`, `/amdcpu/0/temperature/2`, pada perangkat AMD Ryzen 5 5500U.
- Nilai aplikasi cocok dengan pembulatan log CSV; bukti screenshot dan JSON tersedia.
- Sampel berubah, bernilai valid, timestamp berurutan, gap sumber maksimal 3 detik, tanpa telemetry stale selama pencatatan.
- `ResourceGuard.temperature()` menerima telemetry saat live.
- Collector selesai otomatis; kedua file live kemudian berstatus STOP dan ditolak ResourceGuard. Nilai 0 pada file STOP bukan hasil pengukuran.
- 17 tes adapter/guard serta Ruff lulus pada persiapan. Source adapter tidak berubah setelah tes akhir.
- Pemasangan driver exit 0 (6,884 detik). Penolakan persetujuan awal telah diselesaikan dengan izin spesifik pengguna; bukan hambatan yang tersisa.
- Pengamatan awal `observation-attempt-01.jsonl` disimpan terpisah dan tidak dicampur dengan baseline setelah konfirmasi.
- Penutupan administratif dilakukan sekitar 19:30 WIB setelah jeda sesi. Jeda ini bukan durasi baseline/training. Monitor sudah tidak berjalan saat verifikasi akhir; waktu penutupannya tidak diamati.

## Artefak

- `baseline-validation.json`: statistik, kriteria verifikasi, batasan, dan hash bukti.
- `idle-attempt-01.jsonl` serta `idle-attempt-01.summary.json`: baseline setelah konfirmasi.
- `idle-readiness.json`: kondisi yang dikonfirmasi pengguna.
- `source-monitor-log.csv` dan `source-monitor-log.sha256.json`: arsip log aplikasi; dapat mencakup waktu di luar baseline. Gunakan rentang baseline yang ditetapkan untuk analisisnya.
- `live-sensor-verification.json` dan `sensor-monitor-observation.png`: pencocokan sensor.
- `driver-install-authorization.json` serta `driver-install-result.json`: izin dan hasil instalasi.
- `status-before-driver-approval.json`: status historis sebelum izin; `status.json` adalah status terkini.
- `session-closed.json`: verifikasi proses dan akhir collector.

Petunjuk menjalankan ulang sensor terdapat di `SENSOR_SETUP.md` pada root ML. Semua perubahan dilakukan di repository ML asli; frontend dan isi laporan tidak diubah pada Step 1. Tidak ada training, fitting, maupun analisis matriks final test yang dijalankan.
