# Sensor CPU Windows — Step 1

Status 7 September 2026: **GO Step 1 untuk sensor dan baseline; STOP kesiapan training**. PawnIO berhasil dipasang setelah izin spesifik pengguna. Baseline 610 detik setelah penghentian aktivitas berat pengguna tervalidasi: 80,125–85,875°C, akhir 81°C, umur telemetry maksimum 2,994 detik. Beban latar rata-rata 17,23% masih aktif, sehingga ini bukan idle laboratorium mendekati 0% CPU. Seluruh sampel masih di atas batas safe-smoke 80°C. Hasil lengkap pada `reports/step1_20260907/README.md`.

## Sumber dan konfigurasi

- Monitor: [LibreHardwareMonitor v0.9.6, rilis resmi](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/releases/tag/v0.9.6).
- Paket lokal: `.local-tools/LibreHardwareMonitor-0.9.6/LibreHardwareMonitor.zip`.
- SHA-256 paket: `086d9f1b5a99e643edc2cfaaac16051685b551e4c5ac0b32a57c58c0e529c001`, cocok dengan digest GitHub Releases.
- Driver yang disertakan paket: `PawnIO_setup.exe` versi berkas **2.1.0.0**, SHA-256 `a3a46226c5e2824f4cdd42be0eecbabfc672c86f7889710f5ab1e6ad385b47a0`. Authenticode valid, signer `namazso.eu / namazso`.
- Executable monitor tidak memiliki tanda tangan Authenticode; integritas arsipnya diperiksa terhadap digest rilis resmi. Status driver dan monitor dibedakan.
- Pengaturan aplikasi berada pada `LibreHardwareMonitor.config`, berbeda dari konfigurasi runtime `.exe.config`. Isinya: CPU saja, pembaruan 1 detik, log CSV harian setiap 2 detik. Web server dan auto-start tidak diaktifkan. Lokasi konfigurasi dikoreksi sebelum pengukuran dimulai.
- [Dokumentasi proyek](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor#administrator-rights) menyatakan sebagian sensor membutuhkan administrator.
- Format CSV diperiksa dari [Logger.cs v0.9.6](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/blob/v0.9.6/LibreHardwareMonitor/Utilities/Logger.cs). Sensor pada perangkat ini sudah diperiksa; identitas tetap perlu dicocokkan jika konfigurasi atau hardware berubah.

## Persetujuan dan hasil instalasi driver

Pada percobaan awal, review otomatis menolak instalasi karena akses kernel persisten belum diizinkan secara spesifik. Pengguna kemudian menjawab **boleh** atas pertanyaan eksplisit instalasi PawnIO. Percobaan setelah izin disetujui dan berhasil pada 13:59:10 WIB: exit code 0, driver PawnIO Running dengan StartMode Manual. Penolakan awal sudah terselesaikan; bukan hambatan aktif.

Installer yang telah dijalankan:

```text
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\.local-tools\LibreHardwareMonitor-0.9.6\PawnIO_setup.exe
```

Persetujuan spesifik pengguna tersimpan pada `reports/step1_20260907/driver-install-authorization.json` dan hasil pada `driver-install-result.json`. Driver tidak perlu dipasang ulang untuk setiap pengukuran. Tidak ada perubahan Defender, Secure Boot, memory integrity, fan control, power limit, atau BIOS yang diperlukan/disetujui oleh langkah ini.

## Cara menjalankan dan memverifikasi sensor

1. Jalankan monitor dengan hak yang diperlukan. Pastikan perangkat yang ditampilkan adalah AMD Ryzen 5 5500U. Amati sensor CPU package/die, misalnya `Core (Tctl/Tdie)`, beserta satuan Celsius.
2. Periksa file log aktual pada folder monitor. Jangan membuat log buatan untuk melewati validasi.
3. Dari root ML asli, tampilkan identifier sensor log:

```powershell
.\.venv\Scripts\python.exe -m src.cpu_telemetry --log "<path log CSV aktual>" --list-sensors
```

4. Cocokkan identifier, nama, dan nilai CSV dengan sensor CPU pada aplikasi. Simpan waktu pengamatan dan bukti perbandingan. Jangan memilih sensor GPU/disk atau mengasumsikan indeks sensor sebelum dilihat.
5. Siapkan direktori history yang baru, lalu rekam minimal 610 detik:

```powershell
.\.venv\Scripts\python.exe -m src.cpu_telemetry --log "<path log CSV aktual>" --sensor-id "<identifier CPU terverifikasi>" --output telemetry\cpu.json --history reports\step1_20260907\idle-attempt-01.jsonl --duration-seconds 610
```

6. Pastikan rentang timestamp sampel sumber mencakup minimal 600 detik; periksa gap pembaruan, umur telemetry, suhu awal/min/maks/akhir, RAM, dan beban CPU. Catat aktivitas latar yang membuat kondisi tidak benar-benar idle.
7. Status `RECORDED` hanya berarti collector menyelesaikan durasi. **GO Step 1 tetap membutuhkan verifikasi sensor nyata dan penilaian baseline**. Jika sensor salah, beku, stale, atau pengamatan kurang 10 menit, ulangi setelah penyebabnya diselesaikan.
8. Collector berhenti setelah durasi yang diminta atau Ctrl+C. File live kemudian diinvalidasi; history tetap tersedia. Jalankan collector baru bersama sensor aktif pada sesi berikutnya yang diizinkan.

## Perilaku adapter dan batasan

`src/cpu_telemetry.py` mempertahankan timestamp akuisisi CSV dan menulis JSON secara atomik. Hanya identifier CPU dan nama package/die yang dikenali diterima. Sampel lebih tua dari 5 detik, lebih dari 2 detik di masa depan, non-finite, kosong, atau di luar rentang valid ditolak. Nilai identik selama 60 detik memicu STOP konservatif untuk pemeriksaan manual. Ini dapat menolak sensor sehat yang sangat stabil.

Log menggunakan waktu lokal tanpa offset; collector harus berjalan di Windows yang sama, dengan timezone/jam yang stabil. Collector menolak perubahan header dan timestamp mundur. Timestamp CSV bukan bukti independen bahwa hardware berhasil diperbarui; pengamatan aplikasi dan variasi sensor nyata tetap diperlukan. Pembacaan yang memenuhi rentang numerik saja tidak membuktikan sensor benar.

Saat berhenti, JSON berstatus STOP memiliki timestamp 0 dan nilai 0 yang sengaja tidak valid untuk guard. Nilai tersebut adalah penanda penghentian, bukan pembacaan hardware. Adapter tidak mengubah `src/runtime.py`, profil aman, mekanisme resume, atau model. Satu collector saja boleh menulis ke satu path live.

History memuat usia sampel, RAM collector, RAM sistem bebas, dan persentase CPU sistem. RAM tersebut tidak mencakup seluruh aplikasi monitor; catat RAM monitor terpisah pada pengukuran nyata. Sampel CPU persen pertama belum memiliki interval penuh dan jangan digunakan untuk menyimpulkan idle.

## Pemeriksaan yang sudah dijalankan

- 13 tes adapter + 4 tes guard yang sudah ada: **17 lulus**.
- Ruff pada dua file baru: lulus.
- Fixture sintetis hanya berada di direktori tes sementara. Tidak dipakai sebagai bukti sensor hardware, input training, atau baseline idle.
- Percobaan tes pertama gagal pada akses direktori temp pytest lama. Pengulangan dengan direktori sementara unik dalam `.local-tools` berhasil, tanpa mengubah permission direktori lama.
- Bukti: `reports/step1_20260907/adapter-validation.json` dan `adapter-tests.xml`.

Seluruh path relatif di dokumen ini mengacu pada root ML asli:

```text
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier
```

## Sensor yang ditemukan pada perangkat ini

- CPU Windows: AMD Ryzen 5 5500U with Radeon Graphics, 12 logical processor.
- Nama CSV: `Core (Tctl/Tdie)`; identifier `/amdcpu/0/temperature/2`.
- Log 7 September: `.local-tools/LibreHardwareMonitor-0.9.6/LibreHardwareMonitorLog-2026-09-07.csv`.
- Verifikasi sekitar 14:04 WIB: aplikasi menampilkan 85,8°C, sesuai pembulatan log 85,75°C. Nilai dapat maju di antara dua pengamatan.
- `ResourceGuard.temperature()` berhasil membaca file live tanpa training. Bukti pada `live-sensor-verification.json` dan `sensor-monitor-observation.png`.
- Pengamatan awal menunjukkan sekitar 84–86°C dan beban latar sekitar 20%. Perekaman tanpa training tidak otomatis membuktikan kondisi idle yang layak.
## Hasil akhir dan sesi berikutnya

Pencatatan baseline berjalan 14:12:04–14:22:14 WIB; collector berhenti otomatis setelah 610 detik. Verifikasi akhir setelah jeda sesi memastikan file live berstatus STOP dan ditolak guard. Monitor juga sudah tidak berjalan saat verifikasi akhir; waktu penutupannya tidak diamati. Driver tetap terpasang dan tidak perlu diinstal ulang per sesi.

Untuk pemakaian berikutnya, buka executable monitor dari folder lokalnya dengan hak yang diperlukan, gunakan log tanggal sesi aktual, lalu jalankan collector dengan history baru. Jalankan satu collector per path live. Tidak ada training yang otomatis dijalankan oleh monitor atau adapter ini.

Step 2 hanya menyiapkan profil aman dan tes stop/resume. Sebelum smoke/training nanti, ukur ulang kondisi termal sampai tersedia margin yang memadai di bawah batas profil; jangan menaikkan batas untuk menutupi baseline panas. Hasil Step 1 tidak membuktikan penyebab panas dan tidak menyatakan hardware siap dibebani.