# Runtime safety dan resume

Status 7 September 2026: **Step 2 GO untuk implementasi dan tes sintetis**. Training data nyata belum dijalankan. Baseline Step 1 berada pada 80,125–85,875°C; kesiapan termal untuk smoke/training masih **STOP**. Urutan dan persetujuan mengikuti [SAFE_EXECUTION_PLAN.md](SAFE_EXECUTION_PLAN.md).

## Profil operasional

| Batas | safe-smoke | safe-training |
| --- | ---: | ---: |
| Thread learner | 1 | 2 |
| RSS proses dan anak maksimum | 3 GiB | 4 GiB |
| RAM sistem bebas minimum | 4 GiB | 4 GiB |
| Durasi sesi | 5 menit | 20 menit |
| Suhu saat mulai | <75°C | <80°C |
| Suhu stop berkelanjutan | ≥80°C selama 5 detik | ≥85°C selama 10 detik |
| Batas tanpa toleransi durasi | ≥90°C pada pemeriksaan berikutnya | ≥90°C pada pemeriksaan berikutnya |
| Umur telemetry maksimum | 10 detik | 10 detik |
| Interval sampling guard | 1 detik | 1 detik |

Batas durasi/suhu/RAM/thread berasal dari plan. Tambahan jarak mulai 5°C dan penolakan pembacaan ≥90°C merupakan keputusan rekayasa konservatif agar sesi tidak dimulai dekat batas stop. Angka ini bukan temuan eksperimen, batas optimal model, atau jaminan keselamatan perangkat.

`config.yaml` dan run referensi lama tidak diubah. Pilih `config.safe-smoke.yaml` atau `config.safe-training.yaml` lewat `ESSENZA_CONFIG` **sebelum proses Python berjalan**. Run menyimpan profil di `run.json`; perubahan environment tidak mengubah profil run yang telah dibuat. Perintah operasional menolak profil yang hilang, sensor dinonaktifkan, atau batas yang dilonggarkan melewati profil.

Untuk proses smoke yang sudah diizinkan:

```powershell
$env:ESSENZA_CONFIG = (Resolve-Path .\config.safe-smoke.yaml).Path
$env:ESSENZA_THREADS = '1'
```

Untuk run final setelah Step 5 diizinkan, pilih `config.safe-training.yaml` dan `ESSENZA_THREADS='2'`. Kedua assignment ini hanya memilih konfigurasi, bukan memulai eksperimen. `threadpoolctl` membatasi pool native yang telah termuat selama sesi; parameter learner mengikuti thread pada run. Ini bukan batas jumlah seluruh thread OS.

## Sensor dan penghentian

Ikuti [SENSOR_SETUP.md](SENSOR_SETUP.md) untuk mengalirkan data CPU nyata ke file JSON. Sensor yang tidak tersedia, STOP, stale, terlalu jauh di masa depan (>2 detik), nonnumerik, atau nonfinite ditolak. Adapter Step 1 mempertahankan timestamp sumber, membatasi umur sumber 5 detik, dan menolak sensor yang tidak berubah selama 60 detik.

`baseline`, `tune`, `fit`, dan `explain` memeriksa resource sebelum memuat dataset. Pemeriksaan berikutnya terjadi setelah load, di sekitar unit pekerjaan, pada callback boosting kedua learner, dan setelah blok terakhir sebelum sesi dianggap selesai. Satu guard yang pernah STOP tidak dapat dipakai kembali; pengulangan perintah membuat sesi baru.

Guard bersifat **kooperatif**. Penghitungan fitur/resampling, pemuatan data, prediksi/SHAP, I/O, atau satu iterasi native yang sedang berlangsung tidak dapat diputus di tengah. Waktu/RAM/suhu dapat melewati batas sebelum Python mendapat kontrol kembali. Ini bukan watchdog terpisah atau hard limit dari OS. Step 4 masih wajib memvalidasi perilaku pada workload nyata; lolos unit test tidak membuktikan performa termal laptop.

Setelah STOP, baca alasan, hentikan beban, dan lakukan cooldown sesuai plan: minimal 10 menit serta suhu stabil 5 menit, sekaligus memenuhi batas mulai profil. Pengawasan manusia dan larangan menjalankan workload berat bersamaan tetap berlaku. Durasi cooldown/stabilitas belum diotomatisasi oleh guard.

## Log sesi

Setiap sesi memakai file baru `runs/<run>/sessions/<operation>-<algorithm>-<uuid>.jsonl`.

| Event/field | Arti |
| --- | --- |
| `start` | Profil, operasi, learner, dan path telemetry |
| `sample` | Suhu CPU, umur sensor, RSS proses dan anak, RAM sistem bebas |
| `stop` | Alasan penghentian dan durasi hingga deteksi |
| `end` | completed, stopped, interrupted, atau failed |
| `timestamp` | Waktu penulisan event |
| `measured_timestamp` | Waktu sampling resource terakhir |
| `elapsed_seconds` | Durasi monotonik sejak guard dibuat |

Event stop/end dapat membawa sampel terakhir, bukan pengukuran baru tepat saat event. Jika preflight gagal sebelum context masuk, log berakhir pada stop tanpa end. Jika proses dimatikan paksa, end tidak dijamin tersedia. Gagal membuka/menulis log menghentikan workload; ketika disk gagal, bukti akhir mungkin tidak dapat ditulis.

Lock `.resource-session.lock` menyimpan PID dan waktu mulai serta menolak penulis kedua pada **run yang sama**, termasuk learner berbeda. Lock dilepas saat selesai, error Python, atau Ctrl+C. Kill paksa dapat meninggalkan lock: periksa PID, command line, dan waktu proses serta pastikan tidak ada sesi aktif sebelum pemulihan manual. Jangan menghapus lock otomatis atau menjalankan dua run berat bersamaan; lock ini tidak mengunci seluruh komputer.

## Resume baseline dan fitting

Baseline memakai:

- `<algorithm>_baseline_progress.json`: versi, signature protokol, daftar checkpoint, rounds, dan ringkasan skor.
- `<algorithm>_baseline_checkpoints/<strategy>/<fold>/<label>.npz`: prediksi validasi satu label yang sudah selesai.
- `<algorithm>_baselines.json`: hasil lengkap setelah seluruh strategi/fold selesai.

Prediksi ditulis melalui temporary file lalu replace; progress dicatat setelah prediksi tersimpan. Resume memverifikasi checksum, ukuran, rentang skor, rounds, serta signature dataset/split/seed/fitur/label/profil/source/dependency. Skor AP dihitung ulang dari prediksi terverifikasi, bukan dipercaya langsung dari ringkasan progress.

Label yang selesai tidak dilatih ulang. Label yang terputus sebelum checkpoint dikerjakan ulang dari awal; tidak ada resume di tengah boosting satu label. File prediksi yang belum tercatat dalam progress setelah crash boleh ditulis ulang ketika unit tersebut diulang. Progress lama tanpa versi/signature, protokol berubah, atau file rusak ditolak agar hasil tidak tercampur. Run yang sudah memiliki hasil baseline final ditolak jika dijalankan lagi.

Tuning mempertahankan trial Optuna dalam SQLite dan sampler checkpoint. Trial yang terputus tidak dilanjutkan dari tengah fold/label; pemanggilan berikutnya menambah trial sesuai budget yang diizinkan. Final fitting mempertahankan model per label dengan checksum dan recipe yang cocok.

Resume menggunakan perintah, run, source, dependency, dan profil yang sama setelah kondisi aman diperiksa ulang. Jangan mengedit artefak lama agar lolos validasi. `runs/_reference/` tetap arsip baca-saja, bukan target eksekusi.

## Bukti dan batas kesimpulan

[Step 2](reports/step2_20260907/README.md) mencatat 90 tes lulus: payload sensor invalid/stale, preflight panas, RAM proses anak, timeout, stop suhu/cool reset, callback XGBoost/LightGBM, log gagal, lock sesi, serta resume baseline/final fitting. Fixture kecil dipakai untuk memeriksa fungsi kode. Tidak ada skor penelitian baru, tuning data nyata, final test, atau perubahan frontend/laporan LaTeX.
