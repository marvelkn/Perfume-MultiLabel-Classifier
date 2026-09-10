# Panduan historis — digantikan revisi lima sumber

Gunakan [WINDOWS_GUIDE aktif](../../WINDOWS_GUIDE.md) dan RUN_PERFUME_FIVE.cmd di root repository.
Paket ZIP 9 September tidak otomatis berisi revisi 10 September.

# Windows Guide — satu kali menjalankan seluruh ML di PC kampus

**Versi terbaru: full pipeline, 9 September 2026.** Panduan ini menggantikan panduan smoke/sensor sebelumnya. Tidak ada smoke test, instalasi sensor, pengukuran idle, cooldown wajib, atau menunggu konfirmasi antar tahap.

## 1. Bawa paket yang benar

Dari laptop, salin seluruh folder berikut ke flashdisk/penyimpanan pribadi:

```text
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\.local-tools\campus-transfer-full-20260909
```

Isinya ZIP overlay, SHA256, dan panduan. ZIP berisi data audit yang tidak ada dari clone GitHub dan runner terbaru. Paket sebelumnya hanya menjalankan smoke; jangan gunakan paket itu untuk alur ini.

## 2. Instal alat di Windows kampus — satu kali

Buka PowerShell. Jika Git dan uv sudah ada, lewati dua command instalasi.

```powershell
winget install --id Git.Git -e --source winget
winget install --id astral-sh.uv -e --source winget
```

Tutup dan buka kembali PowerShell setelah instalasi. Jika instalasi dibatasi, gunakan alat yang disediakan pengelola lab. Internet diperlukan. Tidak perlu GPU.

## 3. Clone dan ekstrak paket

Gunakan folder baru, lokal, dan tidak otomatis dihapus saat logout. Contoh:

```powershell
$ErrorActionPreference = 'Stop'
$CampusParent = Join-Path $env:USERPROFILE 'EssenzaWork'
New-Item -ItemType Directory -Force -Path $CampusParent | Out-Null
Set-Location $CampusParent
if (Test-Path 'Perfume-MultiLabel-Classifier') { throw 'Folder sudah ada. Gunakan clone baru atau lanjutkan setup clone yang sudah disiapkan.' }
git -c core.autocrlf=false clone https://github.com/marvelkn/Perfume-MultiLabel-Classifier.git
if ($LASTEXITCODE -ne 0) { throw 'Clone gagal' }
Set-Location Perfume-MultiLabel-Classifier
git switch --detach cf0e9b3f23a89a182b74c07272975268f117188c
if ($LASTEXITCODE -ne 0) { throw 'Commit tidak tersedia' }
$TransferFolder = Read-Host 'Path folder campus-transfer-full-20260909 di flashdisk, tanpa tanda kutip'
$Zip = Join-Path $TransferFolder 'essenza-campus-overlay.zip'
$Expected = ((Get-Content -LiteralPath "$Zip.sha256" -Raw).Trim() -split '\s+')[0]
if ((Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash -ne $Expected) { throw 'Checksum ZIP berbeda; salin ulang paket' }
Expand-Archive -LiteralPath $Zip -DestinationPath (Get-Location).Path -Force
```

Jika baru menyiapkan clone dengan paket lama dan belum menjalankan full pipeline: dari root clone, jalankan kembali mulai `$TransferFolder` dengan paket **full**. Jangan overlay source baru pada run full yang sudah berjalan. Perubahan file Git setelah ekstraksi memang diharapkan karena source lokal belum dipush.

## 4. Instal environment — satu kali

```powershell
uv python install 3.12.14
if ($LASTEXITCODE -ne 0) { throw 'Python exact belum tersedia; jangan ganti versi diam-diam' }
uv venv --python 3.12.14 .venv
if ($LASTEXITCODE -ne 0) { throw 'Pembuatan environment gagal' }
uv pip install --python .venv\Scripts\python.exe -r requirements_lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Instalasi dependensi gagal' }
uv pip check --python .venv\Scripts\python.exe
if ($LASTEXITCODE -ne 0) { throw 'Dependensi tidak konsisten' }
.\.venv\Scripts\python.exe reports\campus_20260909\verify_transfer.py
if ($LASTEXITCODE -ne 0) { throw 'Paket atau versi tidak sesuai' }
```

Jika `.venv` sudah dibuat untuk clone ini, lewati `uv venv` dan cukup cek/install dependensinya. Environment laptop tidak disalin. Metode instalasi uv dan Python merujuk [dokumentasi instalasi uv](https://docs.astral.sh/uv/getting-started/installation/) dan [Python versi tertentu](https://docs.astral.sh/uv/guides/install-python/).

## 5. Jalankan sekali, biarkan sampai selesai

Dari File Explorer di root repository, klik dua kali **`RUN_CAMPUS_ALL.cmd`**. Atau dari PowerShell root repository:

```powershell
.\RUN_CAMPUS_ALL.cmd
```

Selesai. Tidak ada permintaan konfirmasi antar tahap. Script memakai run tetap `runs/campus-full-main`, jadi menjalankan command yang sama setelah interupsi akan melanjutkan checkpoint dan budget sebelumnya, bukan mengulang dari nol. Jangan menjalankan dua salinan bersamaan atau mengubah kode/config di tengah eksperimen.

Script mencoba mencegah sleep otomatis selama eksekusi. Jangan shutdown/logout atau menutup terminal. Kebijakan restart otomatis dari pengelola lab tetap dapat menghentikan aplikasi.

**Urutan otomatis:**

1. Verifikasi source, versi dependensi, identitas dataset, dan partisi secara otomatis.
2. Buat snapshot serta run baru dengan amendment kampus.
3. Baseline XGBoost dan LightGBM, masing-masing none/class_weight/random_oversample.
4. Tuning XGBoost, lalu LightGBM: target **30 attempted trials/model**, maksimal **4 jam aktif/model**, mana yang tercapai lebih dulu.
5. Final fit masing-masing kandidat terbaik berdasarkan CV, 25 label; threshold dari holdout development yang tetap.
6. Sensitivitas random seed 42/123/2026 dan scaffold 42 untuk kedua model, tanpa retuning.
7. Bekukan pilihan berdasarkan CV, kemudian evaluasi test kedua model. Hasil test tidak dipakai untuk memilih ulang atau tuning.
8. Ekspor ONNX, cek parity 25 label pada 128 baris development, dan benchmark CPU desktop.
9. Buat ringkasan dan ZIP hasil secara otomatis.

**Waktu:** 4 jam/model hanya cap tuning. Baseline, final fit, sensitivitas, ekspor, serta packaging dihitung terpisah. Total bisa melebihi 8 jam, bergantung PC dan parameter; tidak ada jaminan selesai dalam satu sesi lab pendek. Dua learner berjalan bergantian dengan 2 thread, bukan bersamaan.

## 6. Ambil hasil sebelum meninggalkan PC

Pada akhir proses, terminal mencetak `RESULT ZIP:`. Folder keluaran:

```text
<root repository>\campus-results\
```

Bawa **ZIP terbaru beserta `.sha256`** dari folder tersebut. Kirim seluruh ZIP itu kembali untuk analisis dan penulisan laporan. Jangan hanya mengirim dua file model.

Isi utama ZIP:

| File/folder | Isi |
| --- | --- |
| `RESULTS_SUMMARY.json` | Ringkasan hasil, jumlah trial, waktu aktif, status setiap tahap, metrik test, sensitivitas |
| `pipeline.json` | Ledger budget, checkpoint tahap, alasan berhenti, identitas host |
| `experiment/xgb/` dan `experiment/lgbm/` | 25 model native per learner, manifest, threshold, prediksi/metrics test, lokasi bundle ONNX, benchmark |
| `experiment/*_baselines.json` | Hasil baseline |
| `experiment/*_trials.csv` | Riwayat Optuna termasuk parameter, state, CV, dan fold scores |
| `experiment/studies.sqlite3`, `*_sampler.pkl` | Study dan state sampler untuk audit/resume |
| `experiment/sensitivity/` | Metrik, dukungan label, OOF, dan checkpoint seluruh varian |
| `sessions/` | Log console, RAM, durasi, serta hasil setiap worker |
| `snapshot/`, `amendment.json` | Source/config/protokol yang digunakan |
| `result_checksums.json` | Hash file dalam paket hasil |

`COMPLETE` berarti seluruh tahap berhasil dan target trial tercapai. `FINISHED_WITH_LIMITATIONS` berarti proses selesai tetapi ada budget terpotong atau tahap gagal; model yang berhasil tetap disimpan. `FAILED`/`INTERRUPTED` menyertakan alasan dan bukti yang sudah tersedia. Ekspor ONNX gagal tidak menghapus model native.

Jika tidak ada satu pun trial COMPLETE untuk sebuah learner, model final learner itu tidak dapat dibuat. Script tidak mengganti hasil dengan model historis atau mengarang hasil. Tidak diperlukan persetujuan tambahan: kegagalan dilaporkan di ZIP, sementara tahap independen yang masih bisa dikerjakan tetap dicoba.

## Aturan otomatis dan perubahan metode

- Tidak ada monitoring temperatur atau auto-stop berdasarkan suhu; jangan menyebut suhu perangkat telah terbukti aman.
- RAM tetap diawasi: minimum 4 GiB bebas dan maksimum 4 GiB untuk process tree. Watchdog parent Windows memeriksa sekitar 0,25 detik dan menutup worker jika limit tercapai; pemeriksaan bukan jaminan batas tepat tanpa overshoot.
- Satu worker/session maksimal 20 menit termasuk startup. Baseline/final fit/sensitivitas dapat lanjut otomatis dari checkpoint jika timeout masih menghasilkan kemajuan; maksimum 6 sesi per tahap/varian. Tidak ada pengulangan tanpa kemajuan.
- Satu trial Optuna per worker. Gagal/pruned/interrupted ikut jumlah attempted trials dan waktu aktif. Dua resource stop berturut-turut mengakhiri tuning learner tersebut; kandidat COMPLETE yang tersedia tetap dapat difinalisasi.
- Ledger persisten menghitung startup, worker, checkpoint, dan overhead controller. Setelah controller mati mendadak, reservasi sesi yang belum diselesaikan dibebankan penuh secara konservatif, bukan dihapus. Sedikit overshoot saat polling/penutupan dicatat sebagai waktu aktual.
- Target 30 trial adalah keputusan engineering sebelum melihat skor: 10 startup TPE + ruang hingga 20 trial berikutnya, **bukan hasil smoke atau bukti optimum**. Ini mengganti rumus N berbasis smoke sebelumnya. Jika jumlah aktual berbeda karena cap waktu, hasil dilaporkan sebagai perbandingan terpotong resource, bukan equal-attempt experiment yang lengkap.
- Dataset, partisi, 25 label, ruang parameter, metric macro AP, aturan seleksi, threshold holdout, dan sensitivitas tetap mengikuti protokol asli. Amendment ditulis terpisah agar perubahan bisa dijelaskan pada laporan.
- Resume tidak menambah cap atau mengganti trial gagal. Tahap yang sudah dinyatakan gagal tidak diputar terus tanpa batas. Pemulihan di tengah penulisan proposal sampler dapat memiliki keterbatasan reproducibility; state dan FAIL tetap disimpan untuk audit.

## Status verifikasi paket

Tooling diuji dengan 19 tes sintetis, termasuk ledger, trial FAIL/COMPLETE, state sampler sebelum fit, checkpoint sensitivitas, urutan test setelah kedua final fit, serta penghentian subprocess tidur menggunakan Windows Job Object. Fitting dataset penelitian dan instalasi pada PC kampus belum dijalankan dari laptop. Hasil eksperimen baru diketahui setelah command di atas dijalankan di kampus.
