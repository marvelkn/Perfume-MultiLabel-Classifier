# Step 4 — Attempt 02 dan persiapan runner

**STOP sebelum smoke:** suhu seluruh rekaman ulang masih >=75°C. **Runner selesai disiapkan dan delapan tes sintetis lulus**, tetapi belum ada fitting data nyata. Step 5 tetap belum diizinkan.

## Pengamatan baru

| Pengukuran | Nilai |
| --- | --- |
| Waktu | 2026-09-08T12:36:08.215532+07:00 sampai 2026-09-08T12:38:08.223535+07:00 |
| Durasi | 120 detik; pemeriksaan singkat, bukan baseline baru 10 menit |
| Suhu awal / min / max / akhir | 76.375 / 76.375 / 80.500 / 77.625°C |
| Rata-rata suhu sumber unik | 78.073°C |
| Sampel unik / observasi | 60 / 120 |
| Sampel <75°C | **0** |
| Umur telemetry maksimum | 2.490 detik |
| CPU sistem rata-rata selama rekaman | 15.11% |
| Peak RSS collector / RAM sistem bebas minimum | 20.94 MiB / 10.20 GiB |

Sensor Core (Tctl/Tdie), /amdcpu/0/temperature/2, cocok dengan identitas CPU die Step 1 dan arsip CSV aktual. Semua nilai history cocok sumber. Pengukuran ruangan/kipas dan screenshot UI baru tidak dilakukan. Baseline 610 detik sebelumnya tetap utuh di folder induk.

Guard kembali menolak mulai pada **77,125°C** sebelum backend dimuat. Pada sampel CPU 15 detik, rsEngineSvc.exe memakai sekitar **8,89%** kapasitas CPU total dan sistem **14,1%**. Ini merupakan pengamatan beban latar, bukan diagnosis tunggal penyebab panas.

Bukti powercfg kini memastikan profil Acer mempunyai **Power plan type Balanced**, indeks AC/DC 2. Ini melengkapi ketidakpastian pada pengamatan sebelumnya. Tidak ada perubahan profil daya, layanan keamanan, BIOS, atau pengaturan sistem oleh agent.

Collector selesai exit 0 dan telemetry diinvalidasi. Monitor PID 24212 yang dibuat attempt ini telah dihentikan dengan pemeriksaan nama dan waktu mulai, lalu CSV diarsipkan. Tidak ada model/probe yang dibiarkan berjalan.

## Runner yang disiapkan

[SMOKE_RUNNER.md](../SMOKE_RUNNER.md) menjelaskan penggunaan, batas, dan [checksum implementasi](../smoke_runner_freeze.json). [run_smoke_probe.py](../run_smoke_probe.py) menguji 18 unit sesuai protokol dengan satu deadline 300 detik lintas learner. Prediksi hanya diperiksa validitasnya dan diukur waktunya; tidak ada AP/F1 atau seleksi model.

[Delapan tes](../test_smoke_probe.py) memakai estimator palsu, X sintetis 8 x 2 dan Y sintetis 8 x 25. Percobaan awal menemukan timer monotonik Windows beresolusi 15,625 ms dapat memberikan durasi nol untuk operasi singkat. Pengukuran durasi diperbaiki ke perf_counter beresolusi 100 ns; timer guard sesi tetap.

**Hasil akhir: delapan tes lulus dalam 0,456 detik**, Ruff lulus, dan perintah runner validate berhasil. Peak RAM/suhu tes yang berhasil tidak disampling; jangan memakai peak percobaan gagal sebagai peak hasil akhir. Log percobaan gagal tetap pada smoke-runner-tests.log/json; hasil akhir terpisah pada smoke-runner-tests-final.json.

Catatan implementasi memuat definisi konservatif load: waktu import backend bersama dibebankan pada load estimate setiap learner dan disimpan terpisah. Rumus t_hat, parameter probe, budget, seed, metrik, label, dan partisi praregistrasi tetap. Tidak ada waktu trial atau jumlah trial empiris yang dihasilkan dari tes sintetis.

## Validasi dan bukti

- [thermal-validation.json](thermal-validation.json), [history](preflight-observation.jsonl), [summary](preflight-observation.summary.json), dan [CSV aktual](sensor-source.csv).
- [guard-preflight.json](guard-preflight.json), [log guard](guard-preflight.jsonl), [background-load.json](background-load.json), [power-personality.txt](power-personality.txt), dan [monitor-shutdown.json](monitor-shutdown.json).
- [smoke-runner-tests-final.json](smoke-runner-tests-final.json), [validation.json](validation.json), dan [completion.json](completion.json).

Seluruh 85 berkas praregistrasi, 14 bukti attempt awal, source pipeline/tes/YAML lama, 53 berkas laporan asli, dan frontend tetap cocok. Tambahan kode terbatas pada runner/tes di reports/step4_20260908. Tidak ada run operasional, final-test loading, edit frontend/LaTeX, commit, atau push.

**Berikutnya:** perbaiki kondisi termal atau pilih resource lain, kemudian ulangi Step 4 sesuai batas. Runner masih memerlukan validasi fitting nyata dan cooldown sebelum Step 4 dapat GO. Jangan menaikkan batas suhu atau menggunakan angka sintetis untuk menetapkan trial Step 5.
