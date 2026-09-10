# Runner untuk probe resource Step 4

**Siap secara kode; belum dijalankan pada data nyata.** Delapan tes sintetis lulus. Step 4 tetap STOP termal sesuai bukti attempt-02. Persiapan ini tidak menetapkan jumlah trial atau mengizinkan Step 5.

Runner menggunakan dataset, primary fold 0, tiga label musty/citrus/fruity, tiga strategi imbalance, parameter probe, seed 42, patience 50, dan batas rounds dari praregistrasi. Tidak ada AP/F1, pemilihan model, perubahan threshold, atau deserialisasi matriks test.

## Pengaman dan pencatatan

- Memverifikasi 85 berkas praregistrasi beserta checksum runner/tes/dokumen ini sebelum eksekusi.
- Membatasi folder output baru pada direct child `runs/smoke-*` di repository asli. Folder yang sudah ada ditolak.
- Memakai satu ResourceGuard safe-smoke selama maksimal 300 detik untuk kedua learner berurutan, termasuk import backend, load, resampling, fit, dan predict. Pengaman kooperatif dapat mendeteksi pelampauan setelah blok native selesai.
- Menguji preflight sensor sebelum mengimpor backend dan memuat matriks development. Hash byte artefak protokol tetap diperiksa lebih awal.
- Menjalankan satu thread learner/native pool. Proses lain pada komputer tetap perlu diawasi; lock hanya melindungi folder run yang sama.
- Merekam resources.jsonl, units.jsonl, dan smoke_result.json. Unit yang terputus tetap terlihat; estimasi hanya valid jika sembilan fit per learner selesai dan guard penutup lulus.
- Status selesai runner adalah `PROBE_COMPLETED_PENDING_COOLDOWN_REVIEW`. Ini belum menyatakan Step 4 GO; sensor, temperatur, RAM, perilaku stop, dan cooldown tetap harus ditinjau.
- CLI tidak membuat run final dan tidak menjalankan Optuna. Folder smoke memakai smoke_result.json dan bukan run.json eksperimen utama.

## Definisi waktu

Waktu unit memakai `time.perf_counter` agar unit singkat tetap mempunyai resolusi memadai pada Windows. Pada interpreter ini, timer tersebut melaporkan resolusi 0,0000001 detik, sedangkan `time.monotonic` memakai GetTickCount64 dengan resolusi 0,015625 detik. Durasi dan batas sesi guard yang sudah ada tetap memakai waktu monotonik.

Untuk perencanaan konservatif, `load_seconds` setiap learner memuat durasi pemuatan development ditambah waktu import backend bersama. Import backend hanya dilakukan sekali pada eksekusi nyata, tetapi dibebankan penuh pada estimasi load masing-masing learner karena sesi tuning nantinya dijalankan terpisah. `dataset_load_seconds` dan `backend_import_seconds` disimpan agar asumsi ini dapat diaudit. Rumus praregistrasi tetap:

`t_hat = 1,5 × (load_seconds + 3 × (max_resample_seconds + 25 × max_fit_predict_seconds))`

Faktor 1,5, cadangan 20%, cap 4 jam, dan syarat trial Step 5 tetap. Ini estimasi perencanaan, bukan batas atas biaya semua trial atau bukti optimalitas.

## Pemeriksaan tanpa fitting

Dari root ML asli:

```powershell
.\.venv\Scripts\python.exe reports\step4_20260908\run_smoke_probe.py validate
```

Perintah ini hanya memeriksa artefak dan konfigurasi; tidak memuat matriks dataset, menjalankan sensor, atau membuat folder run.

Setelah Step 4 diizinkan dan pemeriksaan termal/sensor baru memenuhi syarat, bentuk perintah probe adalah:

```powershell
.\.venv\Scripts\python.exe reports\step4_20260908\run_smoke_probe.py run --run runs/smoke-step4-next --temperature-file telemetry/step4-live.json
```

Path telemetry di atas merupakan contoh untuk collector nyata yang sedang aktif; file attempt lama sudah berstatus STOP. Syarat mulai tetap <75°C, stop berkelanjutan 80°C/5 detik, serta hard stop 90°C. Jangan membuat atau mengganti pembacaan sensor untuk melewati syarat tersebut. Cooldown minimal 10 menit dan kestabilan suhu selama 5 menit tetap mengikuti SAFE_EXECUTION_PLAN.

## Validasi dan rekaman versi

[smoke_runner_freeze.json](smoke_runner_freeze.json) adalah rekaman tambahan implementasi v1, dengan hash parent praregistrasi dan tiga file runner/tes/dokumentasi. Ini menambah tooling Step 4; tidak mengubah source learner, dataset, partisi, metrik, parameter probe, atau batas resource yang telah dibekukan.

Tes [test_smoke_probe.py](test_smoke_probe.py) memakai estimator palsu dan matriks sintetis 8 × 2, dengan target sintetis 8 × 25. Tidak ada fitting XGBoost/LightGBM atau pemuatan dataset penelitian. Cakupan: semua 18 unit dan parameter/seed/weight, satu deadline lintas learner, prediksi invalid, identitas dataset salah, preflight panas sebelum backend, kegagalan learner dengan bukti parsial, penolakan overwrite, dan penolakan folder non-smoke.

Percobaan awal: tujuh tes lulus dan satu gagal karena timer beresolusi rendah menghasilkan nol detik pada unit sintetis. Setelah perbaikan timer, **delapan tes lulus dalam 0,456 detik** dan Ruff lulus. Log awal dipertahankan di attempt-02; [hasil akhir](attempt-02/smoke-runner-tests-final.json) membedakan keduanya.
