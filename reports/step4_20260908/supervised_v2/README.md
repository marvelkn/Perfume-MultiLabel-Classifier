# Supervised smoke v2 — resource amendment

Pengguna meminta bypass syarat mulai dan pemantauan yang menghentikan proses ketika kondisi pengaman terjadi. Scope: smoke resource XGBoost dahulu, LightGBM setelah hasil lengkap dan cooldown. Ini bukan izin melewati bukti runtime Step 5, memilih jumlah trial tanpa data, atau menjalankan final tuning.

Perubahan operasional dari v1:

- Syarat awal <75°C dan baseline baru sebelum smoke dilewati; syarat mulai baru tetap **<80°C**. Jika pembacaan pertama >=80°C, worker tidak dibuat.
- Supervisor terpisah memeriksa suhu, sensor, RAM dan waktu setiap sekitar **0,25 detik**. Stop pada pembacaan pertama >=80°C, tanpa menunggu toleransi panas lima detik; hard limit lama 90°C tetap sebagai lapisan tambahan.
- Sensor dibaca langsung dari CSV CPU die LibreHardwareMonitor dengan timestamp akuisisi. Umur maksimum lima detik, tidak boleh mundur, dan nilai tidak boleh tidak berubah selama 60 detik. CSV biasanya diperbarui dua detik: pemantauan tidak sama dengan pembacaan suhu tanpa latensi.
- Worker ditempatkan dalam Windows Job Object dengan KILL_ON_JOB_CLOSE sebelum mendapat sinyal untuk melanjutkan. Error assignment membatalkan worker; penutupan job menghentikan worker dan turunan di job. Callback kooperatif lama tetap dipakai sebagai lapisan tambahan. [Dokumentasi Microsoft](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).
- Profil khusus bernama supervised-smoke-v2; YAML safe-smoke dan validator produksi tidak diubah. Batas satu thread, 3 GiB RSS parent+worker, minimum RAM sistem bebas 4 GiB, dan total aktif **300 detik gabungan** tetap. Estimasi tahap kedua hanya mendapat waktu yang tersisa.
- XGBoost dan LightGBM dijalankan dalam subprocess terpisah. Setelah setiap tahap, cooldown 610 detik dengan lima menit terakhir <75°C sebelum melanjutkan. Setelah worker dihentikan, pengamatan cooldown dicoba tanpa fitting; kegagalan sensor/cooldown dicatat dan tidak memulai tahap berikutnya.
- Worker yang terputus tidak dilanjutkan otomatis. Unit yang sudah selesai tetap tercatat. Estimasi gabungan hanya diberikan setelah kedua learner lengkap dan cooldown lulus. Status sukses masih memerlukan review Step 4 sebelum freeze Step 5.

Dataset, pembagian data, urutan label, sembilan fit per learner, strategi imbalance, parameter probe, seed, early stopping, rumus estimasi dan cap tuning 4 jam/model tidak berubah. Fungsi measure dan fit_binary v1 dipakai kembali. Tidak ada AP/F1 untuk memilih model dan tidak ada pembukaan matriks final test.

Amandemen ini dibuat sebelum fitting protokol baru. Freeze induk 85 berkas dan tiga berkas runner v1 tetap utuh. Hash source baru terdapat pada amendment.json. Jangan menggunakan status v2 sebagai kelulusan batas mulai v1.

Validasi: 10 tes lulus (1,311 detik), Ruff lulus. Tes menggunakan sensor sintetis bertanda jelas serta subprocess tidur/print; tidak fitting model. Cakupan: suhu awal panas, sensor stale/hilang saat berjalan, penghentian worker nyata saat suhu sintetis naik atau timeout, kegagalan Job Object, child selesai normal, validasi budget worker, total budget lintas cooldown, dan sukses setelah kedua stage lengkap. Perbandingan floating-point pada satu tes budget sempat gagal dan diperbaiki dengan toleransi; log awal dipertahankan. Ini belum membuktikan batas temperatur pada fitting nyata atau menjamin semua gangguan perangkat terdeteksi.

Jalankan dari repository asli:

```powershell
.\.venv\Scripts\python.exe reports\step4_20260908\supervised_v2\run_supervised_smoke.py validate
.\.venv\Scripts\python.exe reports\step4_20260908\supervised_v2\run_supervised_smoke.py run --run runs/smoke-supervised-NAMA-BARU --csv .local-tools/LibreHardwareMonitor-0.9.6/LibreHardwareMonitorLog-2026-09-08.csv
```

Monitor asli harus aktif dan diverifikasi terlebih dahulu. Perintah worker merupakan antarmuka internal supervisor. Artefak sensor-history, supervisor/worker resources, unit timings, cooldown, dan hasil gabungan disimpan pada run smoke yang baru; tanpa run.json final atau model produksi.
