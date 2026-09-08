# Step 2 — Profil aman, log, dan stop/resume

**GO untuk implementasi dan tes sintetis.** Pekerjaan pada checkout ML asli dilakukan 7 September 2026; penutupan administratif dan verifikasi ulang pada 8 September 2026. Training data nyata belum dimulai. Kesiapan termal tetap **STOP** berdasarkan baseline Step 1 yang panas, sampai pengukuran ulang memenuhi syarat.

## Hasil perubahan

| Profil | Thread | RAM proses dan anak | Durasi sesi | Suhu untuk mulai | Stop berkelanjutan |
| --- | ---: | ---: | ---: | ---: | --- |
| safe-smoke | 1 | 3 GiB | 5 menit | <75°C | ≥80°C selama 5 detik |
| safe-training | 2 | 4 GiB | 20 menit | <80°C | ≥85°C selama 10 detik |

Keduanya memerlukan minimal 4 GiB RAM sistem bebas, telemetry valid maksimal 10 detik, serta stop pada pembacaan ≥90°C tanpa toleransi durasi. Guard menolak memulai baseline/tune/fit/explain sebelum data dimuat jika kondisi tidak terpenuhi.

Sesi mencatat waktu, suhu, RAM, durasi, serta alasan selesai/stop. Lock per run menolak penulis kedua. Baseline menyimpan prediksi validasi per strategi/fold/label dengan checksum dan signature. Resume memakai kembali hasil yang selesai dan menghitung ulang skor; label yang terputus dikerjakan ulang dari awal. Tuning Optuna dan resume final fitting tetap tersedia.

Panduan: [RUNTIME_SAFETY.md](../../RUNTIME_SAFETY.md), [sensor](../../SENSOR_SETUP.md), dan [plan](../../SAFE_EXECUTION_PLAN.md).

## Validasi

| Pemeriksaan | Hasil |
| --- | --- |
| Seluruh tes ML | **90 lulus**, 0 gagal, 0 dilewati |
| Durasi pytest / runner | **7,607 / 9,390 detik** |
| Peak RSS launcher pytest dan proses anak | **308,87 MiB** tersampling setiap 50 ms |
| RAM sistem bebas minimum selama tes | **12,14 GiB** |
| Suhu CPU aktual Step 2 | **Tidak tersedia** |
| Ruff runtime/checkpoint dan tes baru | Lulus |
| Ruff E9/F63/F7/F82 pada file integrasi | Lulus |
| git diff --check | Lulus |

Cakupan mencakup sensor stale/invalid/STOP, preflight panas sebelum dataset dimuat, suhu berkelanjutan dan reset saat dingin, batas 90°C tanpa toleransi, RAM proses anak, timeout, callback STOP pada kedua learner, kegagalan log, lock sesi, serta pemeriksaan setelah blok terakhir.

Tes resume baseline kedua learner memutus pekerjaan setelah sebagian label selesai, lalu membandingkan hasil resume dengan eksekusi tanpa interupsi. Checkpoint yang selesai tetap sama hash dan waktu modifikasinya. Progress tanpa versi, perubahan seed/source, file rusak, dan ringkasan skor yang diubah turut diperiksa. Tes fitting/ekspor memakai fixture sintetis kecil, bukan dataset penelitian.

Terdapat 18 warning dependency/feature names; tidak ada kegagalan tes akhir. Pemeriksaan awal memiliki satu kegagalan akibat susunan context manager pada tes lock, yang sudah diperbaiki. Bukti awal disimpan terpisah. RAM merupakan penjumlahan RSS proses tes yang tersampling, dapat menghitung shared pages lebih dari sekali, dan bukan peak seluruh Step 2.

## Integritas dan batas kesimpulan

Verifikasi ulang 8 September mencocokkan hash **29 artefak dataset/snapshot/run referensi**, **53 berkas laporan lama**, serta konfigurasi historis `config.yaml`. Frontend Git bersih. Tidak ada run operasional baru, training penelitian, pembacaan matriks final test untuk analisis, perubahan isi LaTeX, commit, atau push.

Guard bersifat kooperatif: satu operasi native/I/O dapat melampaui batas sebelum Python mendapat kontrol kembali. Lolos unit test belum membuktikan latensi penghentian atau kondisi termal pada workload nyata. Cooldown dan pengawasan tetap mengikuti plan. Suhu baseline Step 1 **80,125–85,875°C** adalah catatan historis, bukan pengukuran Step 2 atau suhu saat penutupan.

Pekerjaan utama sekitar 19:39–20:10 WIB pada 7 September. Jeda hingga penutupan 8 September tidak dihitung sebagai compute. File kode/config tidak berubah setelah tes terakhir; penutupan hanya melengkapi dokumentasi dan pemeriksaan integritas.

## Artefak bukti

- [completion.json](completion.json): hasil akhir, pemeriksaan, hash kode/config, dan environment.
- [tests.xml](tests.xml) dan [test-validation.json](test-validation.json): seluruh tes dan pengukuran resource.
- [preservation-validation.json](preservation-validation.json): daftar hash artefak; verifikasi ulang dicatat di completion.
- [preflight.json](preflight.json): lokasi asli, persetujuan, hash sebelum edit, dan lokasi backup.
- [initial-tests.xml](initial-tests.xml) dan [initial-test-validation.json](initial-test-validation.json): pemeriksaan awal sebelum koreksi tes lock.

**Berikutnya: Step 3 — audit final dataset dan praregistrasi protokol**, setelah instruksi tersendiri. Step ini belum dijalankan.
