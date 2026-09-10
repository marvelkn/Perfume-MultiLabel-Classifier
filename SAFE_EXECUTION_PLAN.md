# Rencana Eksekusi Aman Proyek Skripsi Essenza

## Revisi aktif — 10 September 2026

Eksperimen aktif hanya memakai **GoodScents, IFRA 2019, Leffingwell, Arctander 1960, Sigma-Aldrich 2014**.
Gunakan [WINDOWS_GUIDE.md](WINDOWS_GUIDE.md), RUN_PERFUME_FIVE.cmd, serta
[notebook preprocessing](notebooks/02_perfume_five_preprocessing.ipynb).
Hasil: 6.686 molekul, 109 label, fitur 1.024 atau 1.029; metadata dan split tersimpan pada perfume-five-v1.
Preprocessing/notebook dan 21 tes komponen lulus. Training penelitian terbaru belum dijalankan.
Revisi ini menggantikan eksperimen 25 label dan rencana penggunaan seluruh dataset Suh.
Model/API historis dan arsip tetap disimpan; bagian prosedur lama di bawah bukan panduan run terbaru.


## Amendment PC kampus otomatis — 9 September 2026

Instruksi terbaru pengguna mengizinkan satu pipeline otomatis sampai model final dan hasil, tanpa smoke, sensor suhu, cooldown wajib, atau konfirmasi per tahap pada PC kampus. Ikuti [Windows Guide terbaru](reports/campus_20260909/WINDOWS_GUIDE.md) dan [amendment penuh](reports/campus_20260909/full/amendment.json). Target 30 attempted trials/model atau 4 jam tuning aktif/model; trial gagal/pruned ikut budget. Tahap independen tetap dikerjakan dan hasil yang gagal/terpotong dilaporkan. Metode awal dan buktinya dipertahankan untuk audit. Ini kesiapan tooling kampus, bukan klaim eksperimen telah selesai. Laptop asal tetap dikecualikan dari runner baru.

Dokumen ini adalah panduan lintas sesi untuk menyelesaikan eksperimen ML, integrasi Essenza, pengujian Android, dan laporan skripsi secara bertahap. Setiap step hanya boleh dijalankan setelah pemilik proyek memberikan persetujuan eksplisit, misalnya: **`Jalankan Step 0 saja`**.

Jangan melanjutkan otomatis ke step berikutnya walaupun step sebelumnya berhasil. Setelah setiap step, laporkan hasil, temperatur, durasi, penggunaan RAM, artefak yang dibuat, dan status **GO/STOP**.

## Metadata

| Atribut | Nilai |
| --- | --- |
| Pemilik proyek | Marvel Kevin Nathanael |
| Fokus penelitian | Perbandingan XGBoost dan LightGBM untuk prediksi aroma multi-label |
| Perangkat training | Acer Aspire A515-45, AMD Ryzen 5 5500U |
| Sistem operasi | Windows |
| Tanggal penyusunan plan | 6 September 2026 |
| Status eksekusi | Step 0–3 GO; Step 4 STOP termal; Step 5 snapshot calon run siap, budget/run final belum dikunci |
| Prinsip utama | Satu step per persetujuan, fail-closed, dan tidak menjalankan workload berat secara bersamaan |

## Batas Sesi Backend dan Laporan — 7 September 2026

Instruksi pemilik proyek pada sesi ini menetapkan:

- Edit backend hanya di `C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier`.
- Edit laporan hanya di `C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael`.
- Frontend asli dan worktree frontend **read-only**. Jangan mengubah branch, source, aset, atau `PRODUCT_ROADMAP.md` frontend pada sesi ini.
- Persyaratan frontend pada plan lintas proyek di bawah menjadi dependensi eksternal. Status GO Step 0 sesi ini dinilai untuk backend dan laporan; ini tidak menyatakan integrasi/frontend/Android selesai.
- Changelog laporan disimpan pada `CHANGELOG.md` di laporan asli. Instruksi lama untuk memperbarui roadmap frontend tidak diterapkan karena batas read-only dari pengguna.
- Step 0–3 telah selesai sesuai scope. Step 2 menyediakan pengaman dan tes sintetis; Step 3 mengunci audit dan protokol sebelum training penelitian.
- Pengguna mengizinkan **Step 3** pada 8 September 2026 dan memilih budget **4 jam komputasi aktif per model** untuk tuning; cooldown, baseline, final fitting, dan sensitivitas terpisah. Izin ini hanya berlaku untuk Step 3 pada saat diberikan.
- [PREREGISTRATION.md](PREREGISTRATION.md) kini mengunci aturan tuning; jumlah trial final dihitung pada Step 5 dari pengukuran Step 4. Angka 30 trial bukan target otomatis.
- Pengguna mengizinkan penutupan Step 3 dan pelaksanaan Step 4 melalui **oke lanjut** pada 8 September 2026 setelah usulan kedua pekerjaan tersebut. Step 5 dan tahap sesudahnya belum diizinkan.

Persetujuan terbaru pada 8 September 2026: pengguna meminta **oke lanjut terus sampai step 5 juga**. Step 4 dan Step 5 kini diizinkan dalam rangkaian ini; Step 5 hanya dilaksanakan setelah gate Step 4 GO. Step 6 dan seterusnya belum diizinkan. Bukti: reports/step4_20260908/attempt-03/preflight.json dan reports/step5_20260908/preflight.json.

## Kebijakan Lokasi Edit Asli — Wajib

Semua perubahan source berikutnya harus dilakukan langsung pada checkout/folder asli di bawah `C:\Users\ACER\Documents\Marvel\Skripsi`. Folder di bawah `C:\Users\ACER\Documents\ChatGPT\Project Skripsi` adalah worktree/paket audit dan **bukan target edit operasional berikutnya**.

Target edit yang diizinkan:

```text
ML:
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier

Laporan:
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael
```

Path berikut hanya boleh dipakai sebagai referensi read-only atau sumber rekonsiliasi pada Step 0:

```text
C:\Users\ACER\Documents\ChatGPT\Project Skripsi\Perfume-MultiLabel-Classifier
C:\Users\ACER\Documents\ChatGPT\Project Skripsi\Essenza_Frontend
C:\Users\ACER\Documents\ChatGPT\Project Skripsi\Laporan-Update-20260904
```

Sebelum mengubah source, agent wajib menjalankan `git rev-parse --show-toplevel` dan memastikan hasilnya sama dengan target asli di atas. Jika masih menunjuk ke folder `ChatGPT`, status otomatis **STOP**. Jangan menyalin hasil edit kembali secara manual setelah pekerjaan selesai; edit, test, commit, dan push harus berangkat dari checkout asli.

## Lokasi Lengkap Project

### Repository ML

Root:

```text
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier
```

Path penting:

```text
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\SAFE_EXECUTION_PLAN.md
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\README.md
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\IMPLEMENTATION_STATUS.md
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\PROJECT_CONTEXT.md
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\config.yaml
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\feature_spec.json
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\src
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\tests
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\reports
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\runs
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\data\builds\audited-20260904-v2
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\runs\_reference\audited-20260904-v2\run.json
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\reports\audit_20260904\training_preflight.json
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\reports\audit_20260904\dataset_audit.json
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\.venv\Scripts\python.exe
```

`reports/results_xgb.json`, `reports/results_lgbm.json`, `models/`, dan `mobile_assets/` berisi hasil atau model historis. Artefak tersebut tidak boleh dipakai sebagai hasil final protokol terbaru.

### Repository Essenza Frontend

Root:

```text
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend
```

Path penting:

```text
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\AGENTS.md
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\README.md
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\PRODUCT_ROADMAP.md
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\IMPLEMENTATION_COMPLETE.md
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\App.js
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\package.json
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\validation-results.json
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\src
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\src\services\InferenceService.ts
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\src\hooks\useMolecularAnalysis.ts
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\src\assets\metadata\model_manifest.json
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\src\assets\metadata\xgb_meta.json
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\android
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\android\app\build.gradle
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\android\app\src\main\assets\models
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend\__tests__
```

Status penting checkout asli saat plan dibuat:

- checkout asli frontend berada pada branch `main` dan sinkron dengan `origin/main`;
- pembaruan penelitian terbaru berada pada jalur `origin/marvel` dan worktree `codex/project-reliability`;
- `PRODUCT_ROADMAP.md`, `AGENTS.md`, dan beberapa perubahan reliability belum ada pada checkout asli `main` sampai branch target diselaraskan pada Step 0;
- `PRODUCT_ROADMAP.md` di worktree audit memiliki perubahan lokal yang belum di-commit dan harus direkonsiliasi tanpa menjadikan worktree tersebut target edit lanjutan;
- 13 test frontend, TypeScript, dan pemeriksaan 25 aset ONNX lulus;
- lint masih gagal karena 16 error environment Jest dan mempunyai 24 warning inline style;
- NDK Android belum tersedia;
- aplikasi masih memakai 25 model XGBoost historis;
- `InferenceService.ts` masih memakai endpoint Gradio dan hanya mengembalikan skor yang melewati threshold.

### Laporan Skripsi Asli

Root source LaTeX utama:

```text
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael
```

Path penting:

```text
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\SKRIPSI.tex
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\settings.tex
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\pustaka.bib
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\src\1.Awal\07a-Abstrak.tex
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\src\1.Awal\07b-Abstract.tex
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\src\2.Isi\Bab1.tex
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\src\2.Isi\Bab2.tex
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\src\2.Isi\Bab3.tex
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\src\2.Isi\Bab4.tex
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\src\2.Isi\Bab5.tex
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\src\3.Akhir\lampiran.tex
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\output\pdf\SKRIPSI.pdf
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\qa\static-checks.json
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\qa\compile-console.txt
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\qa\updated-build\SKRIPSI.pdf
```

Folder laporan asli belum menjadi repository Git ketika plan ini dibuat. Perubahan tetap dilakukan langsung di folder ini. Pada Step 0, tentukan apakah laporan akan dimasukkan ke repository tersendiri atau dicadangkan dengan mekanisme lain sebelum perubahan besar.

## Status Awal yang Sudah Diverifikasi

| Komponen | Status |
| --- | --- |
| Dataset audit terbaru | 6.703 molekul; 5.383 development; 1.320 test; 25 label; 2.053 fitur |
| Audit identitas | Tidak ada SMILES identik antara development dan test |
| Risiko scaffold | 1.231 dari 1.320 molekul test berbagi scaffold dengan development |
| Test ML | 33 lulus; rerun 6 September 2026 |
| Test frontend | 13 lulus |
| TypeScript dan asset verifier | Lulus; 25 model terdeteksi |
| Training protokol terbaru | Belum dimulai |
| Bundle aplikasi | XGBoost historis |
| Android | SDK ditemukan; NDK belum tersedia; APK/device test belum dilakukan |
| Naskah | Berhasil dikompilasi tanpa error fatal; Bab IV–V masih sementara |
| Administrasi skripsi | Masih terdapat placeholder dosen, NIDN, tanggal, kata pengantar, Turnitin, dan lampiran |

## Kebijakan Keselamatan Hardware

Tidak ada prosedur komputasi yang dapat menjamin risiko hardware nol. Plan ini memakai batas konservatif, pemantauan aktif, pembatasan durasi, dan mekanisme penghentian otomatis untuk menurunkan risiko panas berkepanjangan.

AMD mencantumkan Tjmax Ryzen 5 5500U sebesar 105°C. Nilai tersebut bukan target training. Plan ini memakai batas jauh di bawahnya:

| Profil | Thread | RAM proses maksimum | Durasi satu sesi | Stop temperatur | Durasi panas |
| --- | ---: | ---: | ---: | ---: | ---: |
| `safe-smoke` | 1 | 3–4 GiB | 5 menit | 80°C | 5 detik |
| `safe-training` | 2 | 4 GiB | 20 menit | 85°C | 10 detik |

Referensi resmi:

- AMD Ryzen 5 5500U, Tjmax 105°C: https://www.amd.com/en/support/downloads/drivers.html/processors/ryzen/ryzen-5000-series/amd-ryzen-5-5500u.html
- Acer Safety Guide, jangan menutup ventilasi atau meletakkan komputer langsung di pangkuan: https://global-download.acer.com/supportfiles/files/support/safetyguide/SafetyGuide_Acer_1.0_EN.pdf
- Microsoft Windows Power Mode: https://support.microsoft.com/en-au/windows/change-the-power-mode-for-your-windows-pc-c2aff038-22c9-f46d-5ca0-78696fdf2de8

Aturan wajib sebelum workload berat:

1. Letakkan laptop di meja keras dan datar dengan seluruh ventilasi terbuka.
2. Gunakan charger asli dan pastikan kabel, baterai, serta adaptor tidak menunjukkan kondisi abnormal.
3. Gunakan Windows **Balanced** atau **Best power efficiency**, bukan **Best performance**.
4. Tutup game, emulator, Android build, browser berat, dan workload lain.
5. Jangan menjalankan training dan Gradle/Android build secara bersamaan.
6. Jangan mematikan thermal throttling, mengubah BIOS, melakukan overclock, atau menaikkan power limit.
7. Training harus selalu diawasi dan tidak boleh ditinggal tidur atau ditinggal keluar.
8. Tekan `Ctrl+C` dan hentikan sesi jika ada bau tidak normal, kipas bermasalah, shutdown, baterai/charger sangat panas, atau temperatur mendekati 90°C.
9. Setelah sesi, tunggu setidaknya 10 menit dan pastikan suhu berada maksimal sekitar 10°C di atas baseline idle serta stabil selama 5 menit.
10. Jika sensor hilang, stale, invalid, atau tidak dapat diverifikasi, status otomatis **STOP**.

## Mekanisme Pengaman yang Sudah Ada

Step 2 menambahkan profil operasional terpisah pada `config.safe-smoke.yaml` dan `config.safe-training.yaml`. Profil smoke memakai 3 GiB; training 4 GiB. Keduanya mensyaratkan 4 GiB RAM sistem bebas. `config.yaml` dan artefak run historis tetap utuh.

`src/runtime.py` dan session wrapper kini:

- memvalidasi profil serta sensor sebelum memuat data untuk baseline/tune/fit/explain;
- menolak telemetry hilang, STOP, stale (>10 detik), future (>2 detik), invalid, atau nonfinite;
- menolak mulai jika suhu ≥75°C untuk smoke atau ≥80°C untuk training;
- menerapkan stop suhu profil, serta stop ≥90°C pada pemeriksaan berikutnya tanpa toleransi durasi;
- mencatat timestamp, suhu/umur sensor, RSS proses dan anak, RAM bebas, durasi, serta alasan stop di JSONL per sesi;
- memeriksa resource pada callback boosting kedua learner dan sebelum sesi selesai;
- mencegah dua sesi menulis run yang sama serta menolak pemakaian ulang guard yang sudah STOP.

`src/baseline_checkpoint.py` dan `src/experiments.py` kini:

- menyimpan prediksi baseline per strategi/fold/label secara atomik dengan checksum dan signature;
- melanjutkan unit yang belum selesai, memverifikasi unit tersimpan, serta menghitung ulang AP dari prediksi;
- menolak checkpoint rusak, progress lama tanpa versi, atau perubahan dataset/protokol/source/dependency;
- mempertahankan SQLite/sampler/CSV Optuna serta resume final fitting per label;
- mempertahankan pembekuan seleksi setelah test evaluation dimulai.

Jarak mulai 5°C dan batas 90°C adalah tambahan rekayasa konservatif, bukan hasil penelitian. Pengaman bersifat **kooperatif**, bukan watchdog OS: satu blok native/I/O yang lama dapat melampaui batas sebelum callback berikutnya. Lock hanya per run, dan cooldown serta larangan workload berat bersamaan tetap harus diawasi. Detail: [RUNTIME_SAFETY.md](RUNTIME_SAFETY.md).

**Sebelum training:** kondisi termal harus diperiksa ulang; baseline Step 1 masih panas. Kelulusan 90 tes sintetis pada Step 2 tidak menggantikan validasi workload nyata di Step 4.

## Aturan Eksekusi Lintas Chat

Pada chat berikutnya:

1. Baca seluruh `SAFE_EXECUTION_PLAN.md`, `README.md`, dan `IMPLEMENTATION_STATUS.md`.
2. Baca `Essenza_Frontend/AGENTS.md` dan `Essenza_Frontend/PRODUCT_ROADMAP.md` sebelum mengubah frontend.
3. Periksa status Git seluruh repository sebelum mengubah file.
4. Jalankan hanya step yang disebut eksplisit oleh pengguna.
5. Jangan meneruskan step berikutnya secara otomatis.
6. Jangan membuka final test sebelum kedua model selesai, keputusan seleksi dikunci, dan eksperimen sensitivitas sudah ditetapkan.
7. Setelah satu step, perbarui **Log Eksekusi** di bagian bawah dokumen ini.
8. Jika kondisi STOP ditemukan, hentikan pekerjaan berat dan laporkan penyebabnya. Jangan menaikkan batas temperatur untuk memaksa proses selesai.

## Plan Eksekusi Per Step

### Step 0 — Stabilkan Repository, Artefak, dan Lokasi Laporan

**Beban:** ringan, tanpa training. **Scope sesi:** backend dan laporan; frontend read-only.

Pekerjaan:

- verifikasi root checkout ML asli sebelum perubahan, branch publikasi `main`, dan keamanan perubahan lokal;
- cadangkan `SAFE_EXECUTION_PLAN.md` yang belum tercatat Git, lalu fetch dan fast-forward ML asli secara non-destruktif;
- pastikan source pada worktree audit yang perlu direkonsiliasi sudah tercakup dalam revision tujuan;
- salin dataset audit v2 dan snapshot sumber yang diabaikan Git ke checkout ML asli; cocokkan SHA-256 sumber, manifest, dan tujuan tanpa mengeksekusi model atau membuka matriks test untuk analisis;
- arsipkan run audit byte-identik pada `runs/_reference/audited-20260904-v2`; path lama dalam `run.json` dipertahankan sebagai bukti dan run ini dilarang dieksekusi;
- siapkan `.venv` baru pada ML asli dari Python 3.12 dan `requirements_lock.txt`; jangan menyalin environment lama; periksa dependency serta impor komponen tanpa fitting;
- gunakan laporan asli sebagai sumber kebenaran; buat backup ZIP seluruh source dan cocokkan hash setiap entry;
- dokumentasikan kebijakan backup dan changelog pada laporan asli; perubahan isi bab belum termasuk Step 0;
- catat status frontend sebagai referensi saja; rekonsiliasi branch/roadmap frontend dilakukan di sesi lain;
- simpan bukti pada `reports/step0_20260907`; jangan memakai `git reset --hard` atau operasi destruktif.

**GO (backend dan laporan):** ML asli berada pada `main` yang telah diselaraskan, plan/perubahan penting tersimpan, dataset dan snapshot cocok hash, environment lokal lolos pemeriksaan, dan backup laporan terverifikasi serta kebijakannya tercatat.

**STOP:** konflik/overwrite yang belum direkonsiliasi, ketidakcocokan hash artefak, dependency tidak dapat dipasang secara konsisten, atau sumber laporan ambigu. Kesiapan frontend tidak diklaim oleh GO ini.

Run final baru hanya boleh dibuat di Step 5, setelah profil aman dan protokol disetujui. Backup lokal melindungi revisi, tetapi masih berada pada disk yang sama.

### Step 1 — Hubungkan dan Validasi Sensor CPU

**Beban:** ringan, tanpa training.

Pekerjaan:

- sediakan pembaca suhu CPU nyata pada Windows;
- tulis telemetry JSON dengan format berikut:

```json
{
  "timestamp": 0,
  "cpu_c": 0.0
}
```

- pastikan timestamp diperbarui maksimal setiap 1–5 detik;
- bandingkan nilai file dengan sensor CPU package pada aplikasi monitor;
- catat baseline idle selama minimal 10 menit.

**GO:** telemetry stabil, masuk akal, dan tidak pernah lebih tua dari 10 detik.

**STOP:** sensor hanya menampilkan temperatur disk/GPU, nilainya tidak berubah, tidak masuk akal, atau sering stale.

### Step 2 — Buat Profil Aman dan Perkuat Stop/Resume

**Beban:** ringan; hanya perubahan kode, konfigurasi, dan unit test.

Pekerjaan:

- buat konfigurasi terpisah `config.safe-smoke.yaml` dan `config.safe-training.yaml` tanpa mengubah bukti run lama;
- gunakan batas pada tabel profil keselamatan;
- perkuat resume baseline dari file progress;
- tambahkan pencatatan timestamp, temperatur, RAM, alasan stop, dan durasi sesi;
- tambah test untuk telemetry stale, stop temperatur, session timeout, dan resume baseline;
- jalankan test ML saja, tanpa real-data training.

**GO:** seluruh test lulus dan simulasi guard berhenti sesuai batas.

**STOP:** guard dapat dilewati, telemetry tidak tervalidasi, atau resume merusak hasil sebelumnya.

### Step 3 — Audit Final Dataset dan Praregistrasi Protokol

**Beban:** ringan, tanpa training.

Dataset target:

```text
C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\data\builds\audited-20260904-v2
```

Pekerjaan:

- verifikasi hash dataset dan `feature_schema_id`;
- kunci metrik utama `macro_average_precision`;
- kunci aturan pemilihan model sebelum test;
- tetapkan seed utama dan seed sensitivitas;
- tetapkan eksperimen scaffold;
- tetapkan strategi imbalance yang diizinkan;
- tetapkan aturan penentuan trial budget berdasarkan waktu dan suhu smoke test, bukan berdasarkan skor test;
- MLSMOTE tetap opsional dan tidak diaktifkan tanpa keputusan eksplisit.

**GO:** keputusan tertulis lengkap dan test set tetap belum dibuka.

**STOP:** masih ada keputusan metodologi yang akan berubah berdasarkan hasil test.

### Step 4 — Smoke Test Real-Data Maksimal 5 Menit

**Beban:** rendah–sedang, terkontrol.

Ikuti probe yang telah dikunci pada [PREREGISTRATION.md](PREREGISTRATION.md), bagian 5: total 300 detik aktif untuk kedua learner, sembilan label-fit selesai per learner agar estimasi valid, tanpa seleksi berdasarkan skor. Ukur ulang kondisi termal dan pastikan suhu mulai <75°C. Jika tidak memenuhi batas, hentikan sebelum fitting.

Pekerjaan:

- buat run khusus smoke yang terpisah dari run final;
- gunakan `safe-smoke`: 1 thread, maksimal 5 menit, stop 80°C/5 detik;
- jalankan sebagian workload real-data untuk menguji sensor, callback, RAM, dan stop otomatis;
- hasil skor smoke tidak digunakan untuk memilih model;
- lakukan cooldown setelah sesi.

**GO:** tidak ada sensor stale, RAM aman, temperatur stabil, dan proses berhenti bersih.

**STOP:** temperatur menyentuh batas, naik terlalu cepat, sensor putus, sistem tidak responsif, atau muncul gejala hardware abnormal.

### Step 5 — Bekukan Budget Final dan Inisialisasi Run Baru

**Beban:** ringan, tanpa training.

Pekerjaan:

- gunakan data durasi dan suhu Step 4 untuk menetapkan jumlah trial yang realistis;
- keputusan hanya berdasarkan resource, bukan performa test;
- buat run final baru menggunakan `safe-training`;
- simpan snapshot konfigurasi, source hash, dependency, split, dan keputusan eksperimen.

**GO:** run final masih kosong dari hasil test dan seluruh konfigurasi sudah immutable.

**STOP:** konfigurasi run tidak cocok dengan dataset atau source saat ini.

### Step 6 — Baseline XGBoost

**Beban:** sedang.

Pekerjaan:

- jalankan hanya baseline XGBoost;
- maksimal 20 menit per sesi;
- monitor temperatur/RAM selama proses;
- jika session limit tercapai, cooldown lalu resume pada sesi terpisah;
- validasi tiga strategi: none, class weight, dan random oversampling.

**GO:** seluruh fold dan strategi baseline XGBoost selesai serta artefaknya valid.

**STOP:** guard aktif, temperatur tidak kembali normal setelah cooldown, atau artefak tidak konsisten.

### Step 7 — Baseline LightGBM

Sama dengan Step 6, tetapi hanya untuk LightGBM. Jangan menjalankan Step 6 dan Step 7 bersamaan.

### Step 8 — Tuning XGBoost Bertahap

**Beban:** sedang dan berulang.

Pekerjaan:

- jalankan tambahan 1 trial per sesi pada awal tuning;
- setelah setiap trial, catat status, durasi, suhu maksimum, RAM, dan skor CV;
- cooldown sebelum sesi berikutnya;
- setelah stabil, batch boleh dinaikkan maksimal 2 trial per sesi selama tetap berada dalam batas 20 menit;
- completed/pruned/failed trial tetap disimpan dan dilaporkan.

**GO:** jumlah trial yang telah dipraregistrasikan selesai.

**STOP:** satu trial tidak pernah bisa selesai dalam batas aman atau failure berulang menunjukkan masalah kode/resource.

### Step 9 — Tuning LightGBM Bertahap

Sama dengan Step 8, tetapi dijalankan terpisah setelah sesi XGBoost selesai dan laptop sudah dingin.

### Step 10 — Final Fitting Kedua Kandidat

**Beban:** sedang.

Pekerjaan:

- fit XGBoost dan LightGBM pada sesi terpisah;
- gunakan kandidat terbaik berdasarkan CV;
- gunakan holdout threshold yang sudah ditentukan;
- manfaatkan resume per label jika session limit aktif;
- jangan membuka test set.

**GO:** kedua kandidat memiliki manifest final lengkap.

**STOP:** salah satu kandidat belum selesai, konfigurasi berubah, atau manifest tidak lengkap.

### Step 11 — Eksperimen Sensitivitas Sebelum Final Test

**Beban:** sedang dan dapat dibagi menjadi beberapa sesi.

Pekerjaan:

- jalankan seed sensitivity sesuai pra-registrasi; gunakan partisi development dan rounds tetap pada PREREGISTRATION.md, bukan membangun ulang outer test;
- jalankan scaffold sensitivity sesuai pra-registrasi;
- dokumentasikan keterbatasan scaffold overlap dan label missingness;
- jangan mengubah model utama berdasarkan hasil test karena test belum dibuka.

**GO:** seluruh analisis tambahan yang diwajibkan sudah tersedia atau keterbatasan compute telah didokumentasikan sebelum test.

### Step 12 — Freeze Selection dan Evaluasi Test Satu Kali

**Beban:** ringan–sedang.

Pekerjaan:

- hash dan bekukan kedua manifest;
- pilih kandidat berdasarkan macro AP validasi;
- evaluasi test XGBoost dan LightGBM satu kali;
- laporkan macro/micro AP, macro/micro F1, precision, recall, ROC-AUC, Hamming loss, subset accuracy, serta metrik dan support per label;
- jangan melakukan tuning ulang berdasarkan hasil test.

**GO:** metrik final lengkap, dapat ditelusuri, dan tidak terjadi test leakage.

**STOP:** manifest berubah atau salah satu model belum dikunci.

### Step 13 — Ekspor ONNX, Parity, dan Benchmark Desktop

**Beban:** ringan–sedang.

Pekerjaan:

- ekspor model terpilih untuk seluruh 25 label;
- verifikasi parity Python-native terhadap ONNX;
- verifikasi checksum, threshold, input/output tensor, label order, dan feature schema;
- ukur ukuran bundle, load time, first inference, median/p95 latency, dan RAM desktop;
- jangan mengganti bundle frontend jika parity belum lulus.

**GO:** seluruh label lolos parity dan bundle tervalidasi.

### Step 14 — Integrasi Frontend dan API

**Beban:** ringan, jangan dilakukan bersamaan dengan training.

Pekerjaan sesuai roadmap:

- **P0.3:** ubah inferensi agar mengembalikan seluruh skor, threshold, dan `selected`;
- **P0.4:** pindahkan frontend dari Gradio ke REST `POST /fingerprint`;
- deploy API dengan `feature_spec.json` yang sama dengan bundle;
- impor bundle baru memakai importer tervalidasi;
- jalankan TypeScript, Jest, lint, asset verifier, dan contract tests;
- perbaiki lint environment Jest dan evaluasi warning inline style.

**GO:** API, frontend, dan bundle mempunyai `feature_schema_id` yang sama serta seluruh test lulus.

### Step 15 — Android Build dan Device Validation

**Beban:** sedang–berat; wajib terpisah dari training.

Pekerjaan:

- pasang NDK 27.1.12297006;
- build hanya ABI perangkat target terlebih dahulu;
- lakukan fresh install dan upgrade-cache test;
- uji inferensi seluruh label, API downtime, cancellation, storage, dan jaringan;
- verifikasi parity pada perangkat;
- ukur latency, peak RAM, ukuran APK, dan stabilitas;
- siapkan release signing hanya setelah debug build stabil.

**GO:** APK berhasil dan bukti perangkat lengkap.

**STOP:** build menyebabkan resource pressure abnormal, ONNX runtime crash, parity gagal, atau suhu tidak terkendali.

### Step 16 — Finalisasi Laporan Skripsi

**Beban:** ringan.

Pekerjaan:

- perbarui abstrak dan abstract;
- ganti tabel placeholder Bab IV dengan hasil final;
- jelaskan parameter terbaik, trial status, waktu komputasi, per-label metrics, sensitivity, parity, Android latency/RAM, dan keterbatasan;
- ubah Bab V dari simpulan sementara menjadi simpulan final;
- lengkapi nama dosen, NIDN, tanggal, kata pengantar, Turnitin, dan formulir bimbingan;
- kompilasi, jalankan static checks, dan render seluruh PDF untuk QA visual.

**GO:** naskah menjawab seluruh rumusan masalah hanya dengan bukti yang tersedia dan tidak mengklaim prediksi campuran atau validasi sensori.

## Fallback Jika Laptop Tidak Stabil

Jika smoke test atau satu trial tidak dapat berjalan dalam batas konservatif:

1. jangan menaikkan batas temperatur;
2. kurangi thread hanya jika analisis menunjukkan beban masih terlalu tinggi;
3. periksa ventilasi dan kondisi pendinginan tanpa membongkar laptop jika tidak mempunyai kompetensi servis;
4. pindahkan training berat ke komputer kampus, server, atau cloud CPU;
5. tetap gunakan dataset, dependency lock, seed, dan artefak run yang sama agar eksperimen reproducible.

## Progress Tracker

| Step | Status | Bukti/artefak | Catatan |
| ---: | --- | --- | --- |
| 0 | GO | reports/step0_20260907/completion.json | Backend dan laporan stabil; frontend read-only |
| 1 | GO | reports/step1_20260907/baseline-validation.json; SENSOR_SETUP.md | Sensor/baseline 610 detik valid; suhu 80,125–85,875°C, training STOP |
| 2 | GO | reports/step2_20260907/completion.json; RUNTIME_SAFETY.md | 90 tes lulus; profil, log, lock, dan resume; tanpa real-data training |
| 3 | GO | reports/step3_20260908/completion.json; PREREGISTRATION.md | 85 checksum, audit notebook dan protokol terkunci; 4 jam/model |
| 4 | STOP | reports/step4_20260908/attempt-03/completion.json | Baseline 610 detik: 71,750–85,625°C, akhir 84,875°C; 0 fitting nyata |
| 5 | PERSIAPAN SELESAI, MENUNGGU SMOKE | reports/step5_20260908/attempt-20260908-213401/completion.json | 47 snapshot nonoperasional; guard live lulus. Jumlah trial/run final belum dibuat |
| 6 | BELUM DIMULAI | — | Baseline XGBoost |
| 7 | BELUM DIMULAI | — | Baseline LightGBM |
| 8 | BELUM DIMULAI | — | Tuning XGBoost |
| 9 | BELUM DIMULAI | — | Tuning LightGBM |
| 10 | BELUM DIMULAI | — | Final fit |
| 11 | BELUM DIMULAI | — | Sensitivity |
| 12 | BELUM DIMULAI | — | Final test |
| 13 | BELUM DIMULAI | — | ONNX/parity |
| 14 | BELUM DIMULAI | — | Frontend/API |
| 15 | BELUM DIMULAI | — | Android |
| 16 | BELUM DIMULAI | — | Laporan |

## Log Eksekusi

Tambahkan satu entri setelah setiap step. Jangan menghapus entri lama.

```text
Tanggal/waktu:
Step:
Persetujuan pengguna:
Perintah atau tindakan:
Durasi:
Suhu idle / maksimum / akhir:
RAM maksimum:
Hasil GO/STOP:
Artefak:
Perubahan file:
Catatan dan batasan:
Step berikutnya yang boleh diusulkan:
```

### 7 September 2026 — Step 0

- Persetujuan: pengguna meminta mulai Step 0; kemudian mengizinkan lanjut Step 1 setelah verifikasi penutup.
- Tindakan: backup plan/laporan, fast-forward ML asli, rekonsiliasi artefak dengan SHA-256, pembuatan environment dari lock, pemeriksaan dependency/impor, dan kebijakan backup/changelog laporan.
- Durasi: pekerjaan utama sekitar 11:37–11:46 WIB; penutupan administratif setelah interupsi pada sekitar 13:44 WIB. Instalasi 161,18 detik; probe impor 3,03 detik. Jeda interupsi tidak dihitung sebagai compute.
- Suhu idle/maksimum/akhir: tidak tersedia; sensor belum terhubung. Tidak ada workload training.
- RAM: peak probe impor 225,09 MiB; bukan peak seluruh step. RAM bebas saat probe sekitar 9,09 GiB. Sampling pip launcher tidak mewakili proses anak.
- Hasil: **GO untuk backend dan laporan**. 53 berkas laporan lama tidak berubah; 29 artefak cocok hash; 109 dependency pin cocok, pip check dan 15 impor lulus.
- Artefak: `reports/step0_20260907/`, `.local-backups/step0-20260907-114024/`, dataset/snapshot, run `_reference`, `.venv`, serta `BACKUP_POLICY.md` dan `CHANGELOG.md` laporan.
- Batasan: backup satu disk, frontend read-only, source bab tidak diubah, training/test evaluation belum dimulai, belum commit/push.
- Step berikutnya: Step 1 sudah diizinkan pengguna; Step 2 dan seterusnya menunggu izin terpisah.
### 7 September 2026 — Step 1 (persiapan, STOP sebelum baseline)

- Persetujuan: pengguna meminta lanjut Step 1 setelah Step 0.
- Tindakan: mencari monitor/namespace sensor Windows, menyiapkan LibreHardwareMonitor 0.9.6 dari rilis resmi dan memverifikasi SHA-256, mengekstrak installer PawnIO bawaan dan memverifikasi Authenticode, menyiapkan konfigurasi monitor serta adapter CSV ke telemetry JSON.
- Hambatan: review persetujuan otomatis **menolak instalasi PawnIO melalui RunAs/UAC**, karena driver kernel menambah akses sistem persisten dan persetujuan Step 1 dinilai belum spesifik mencakupnya. Perintah instalasi tidak dieksekusi. Jangan mencoba ulang/menjalankan secara tidak langsung sebelum izin spesifik pengguna.
- Durasi: persiapan sekitar 13:44 WIB sampai timestamp penutupan pada `reports/step1_20260907/status.json`; baseline nyata **0 detik**. Final test runner 0,484 detik.
- Suhu idle/maksimum/akhir: **tidak tersedia**.
- RAM: peak working set proses tes 44,37 MiB; bukan peak seluruh Step 1 atau aplikasi monitor.
- Validasi: 17 tes lulus (13 adapter, 4 guard lama), Ruff lulus. Percobaan awal tes gagal karena permission direktori temp lama; pengulangan pada direktori unik berhasil tanpa perubahan permission.
- Hasil: **STOP untuk Step 1**; persiapan software selesai, tetapi sensor nyata dan baseline minimal 10 menit belum tervalidasi.
- Artefak/perubahan: `src/cpu_telemetry.py`, `tests/test_cpu_telemetry.py`, `SENSOR_SETUP.md`, `.gitignore`, `.local-tools/LibreHardwareMonitor-0.9.6/`, `reports/step1_20260907/`, dan plan ini.
- Batasan: fixture tes sintetis bukan pembacaan hardware. Monitor belum dijalankan; guard/profil/resume/model tidak diubah; training dan evaluasi test tidak dilakukan. Frontend dan isi laporan tidak diubah. Belum commit/push.
- Tindakan berikutnya: minta izin eksplisit untuk instalasi driver PawnIO yang tertera di `SENSOR_SETUP.md`, lalu lanjutkan validasi sensor dan baseline Step 1. Step 2 belum diizinkan.

### 7 September 2026 — Step 1 (lanjutan setelah izin driver, GO sensor/baseline)

- Persetujuan: pengguna menjawab **boleh** untuk instalasi driver PawnIO dan mengonfirmasi penghentian aktivitas berat, meja keras, serta ventilasi terbuka. Permintaan `continue` melanjutkan penutupan Step 1.
- Instalasi: PawnIO versi installer 2.1.0.0, tanda tangan valid, exit 0 pada 13:59:10 WIB; driver Running, StartMode Manual. Penolakan review otomatis sebelumnya sudah terselesaikan oleh izin spesifik ini.
- Tindakan: membetulkan lokasi pengaturan aplikasi menjadi `LibreHardwareMonitor.config`, menjalankan monitor, mencocokkan `Core (Tctl/Tdie)` `/amdcpu/0/temperature/2` dengan CSV dan ResourceGuard, lalu merekam baseline baru setelah konfirmasi pengguna.
- Durasi baseline: **610 detik**, 14:12:04–14:22:14 WIB; timestamp sumber mencakup 609 detik. Pengamatan awal terpisah juga berdurasi 610 detik dan tidak dicampur. Penutupan administratif sekitar 19:30 WIB setelah jeda sesi; jeda bukan durasi komputasi.
- Suhu awal/minimum/maksimum/rata-rata/akhir: **83,875 / 80,125 / 85,875 / 83,654 / 81,000°C**.
- Telemetry: 305 sampel sumber unik, 608 observasi collector; gap sumber maksimum 3 detik, umur maksimum 2,994 detik. Nilai berubah, timestamp berurutan, dan tidak stale selama pencatatan.
- RAM: peak RSS collector **21,07 MiB**; RAM sistem bebas minimum **9,87 GiB**. Peak RAM monitor selama baseline tidak disampling; nilai sebelum baseline tidak dianggap peak seluruh sesi.
- Beban latar: rata-rata CPU **17,23%**, 60 observasi terakhir **12,21%**. Baseline menggambarkan kondisi tanpa aktivitas berat pengguna dengan layanan latar/chat/monitor aktif, bukan idle laboratorium mendekati 0% CPU. Suhu ruangan dan kecepatan kipas tidak diukur.
- Hasil: **GO Step 1 untuk sensor dan pencatatan baseline**. **STOP kesiapan training** karena seluruh suhu baseline melebihi batas stop safe-smoke 80°C; perlu pengukuran ulang kondisi termal sebelum workload tersebut.
- Verifikasi tambahan: 17 tes sebelumnya tetap berlaku karena source adapter tidak berubah; ResourceGuard menerima data live dan menolak kedua file setelah collector selesai/invalidasi. Monitor tidak lagi berjalan saat verifikasi akhir; waktu penutupannya tidak diamati.
- Artefak: `reports/step1_20260907/README.md`, `baseline-validation.json`, `idle-attempt-01.jsonl`, summary, arsip CSV/hash, screenshot sensor, bukti instalasi/izin, dan `SENSOR_SETUP.md`.
- Batasan: tidak ada training atau evaluasi test. Tidak ada perubahan frontend/isi laporan, guard/profil/resume/model, BIOS, fan control, atau pengaturan keamanan. Belum commit/push.
- Step berikutnya: **Step 2** (profil aman dan tes stop/resume, beban ringan), menunggu instruksi tersendiri. Jangan meneruskan otomatis atau mengartikan GO sensor sebagai izin training.
### 7 September 2026 — Step 2 (GO implementasi dan tes sintetis)

- Persetujuan: pengguna menjawab **oke lanjut** setelah usulan Step 2. Scope hanya kode, profil, dokumentasi backend, dan tes ringan.
- Tindakan: backup sumber awal; tambah safe-smoke/safe-training; validasi profil/sensor sebelum data dimuat; pencatatan JSONL resource; lock per run; stop yang tidak dapat direset pada sesi yang sama; checkpoint baseline per strategi/fold/label dengan checksum/signature; pemeriksaan callback dan penutupan sesi.
- Durasi: preflight 19:39:58 WIB; pekerjaan dan dokumentasi hingga sekitar 20:10 WIB. Tes akhir **7,61 detik** menurut pytest; runner **9,39 detik** termasuk startup. Penutupan administratif dilanjutkan pada 8 September sekitar 09:20 WIB setelah jeda; jeda bukan durasi komputasi atau training.
- Suhu idle/maksimum/akhir: **tidak tersedia pada Step 2**, karena tidak ada sensor live terverifikasi yang berjalan. Angka Step 1 tidak dipakai sebagai pengukuran Step 2.
- RAM: peak RSS tersampling pytest launcher beserta anak **308,87 MiB**; RAM sistem bebas minimum **12,14 GiB**. Sampling 50 ms; penjumlahan RSS dapat menghitung shared pages lebih dari sekali dan tidak mewakili peak seluruh Step 2.
- Validasi: **90 tes ML lulus**, 18 warning dependency/feature names. Cakupan termasuk kedua callback learner, sensor invalid/stale, suhu, timeout, RAM anak, kegagalan log, lock, resume baseline kedua learner, dan resume final fitting.
- Pemeriksaan awal menemukan satu kegagalan akibat penempatan nested context manager pada tes lock; tes diperbaiki dan keseluruhan suite diulang. Bukti awal dipertahankan pada `initial-tests.xml` dan `initial-test-validation.json`.
- Ruff lulus untuk runtime/checkpoint dan tes barunya; pemeriksaan fatal Python E9/F63/F7/F82 lulus pada file integrasi yang diubah. `git diff --check` lulus.
- Preservasi: **29 artefak** dataset/snapshot/run referensi, **53 berkas laporan lama**, dan `config.yaml` cocok hash; frontend Git bersih. Pemeriksaan hanya membaca byte untuk hash, tidak membuka matriks final test. Tidak ada run operasional baru.
- Hasil: **GO Step 2 untuk implementasi dan tes sintetis**; **STOP kesiapan termal training** tetap berlaku.
- Artefak/perubahan: kedua YAML profil; `src/runtime.py`, `src/baseline_checkpoint.py`, `src/experiments.py`, `src/explain_model.py`; tes runtime/baseline/resume; README, IMPLEMENTATION_STATUS, RUNTIME_SAFETY, plan, dan `reports/step2_20260907/`. Backup `.local-backups/step2-20260907-193957/`.
- Batasan: guard kooperatif, belum ada pengukuran latency stop/termal real-data. Tidak ada training penelitian, test evaluation, perubahan frontend/isi LaTeX, commit, atau push.
- Step berikutnya yang diusulkan: **Step 3 — Audit Final Dataset dan Praregistrasi Protokol**, menunggu instruksi terpisah.

### 8 September 2026 — Step 3 (GO audit dan praregistrasi)

- Persetujuan: pengguna mengizinkan Step 3 dan memilih 4 jam aktif/model; penutupan administratif dilanjutkan setelah interupsi dengan izin 8 September.
- Waktu: mulai 2026-09-08T09:23:14.981257+07:00; audit selesai 09:31:21 WIB; verifier 09:50:34 WIB; penutupan 2026-09-08T11:33:58.604257+07:00. Jeda/interupsi tidak dihitung sebagai komputasi.
- Tindakan: audit development, verifikasi artefak, penguncian metrik/partisi/seed/imbalance/search space/budget, resolusi sensitivitas scaffold dengan rounds tetap, serta catatan gap laporan dan frontend.
- Validasi: tujuh sel kode notebook lulus, 85 berkas frozen cocok SHA-256, sembilan pemeriksaan verifier lulus; Ruff runner/verifier lulus pada validasi Step 3. Verifikasi checksum diulang saat penutupan dan lulus.
- Durasi audit: 5,672 detik; peak proses audit 171,40 MiB; RAM sistem bebas setelah audit 13,35 GiB. Angka tersebut tidak mewakili seluruh sesi.
- Suhu: tidak tersedia saat audit; tidak memakai suhu Step 1 sebagai suhu Step 3.
- Preservasi: 29 artefak tercakup verifikasi freeze; 53 berkas laporan lama cocok backup; source/tes/YAML cocok preflight; frontend bersih; tidak ada run operasional.
- Hasil: **GO Step 3** untuk audit dan praregistrasi lokal, bukan klaim optimalitas atau kesiapan termal.
- Artefak: PREREGISTRATION.md, protocols/essenza_v1_20260908/, reports/step3_20260908/ termasuk completion.json dan notebook audit.
- Batasan: tidak ada training penelitian atau test evaluation. LaTeX belum diselaraskan; runner ledger tuning dan sensitivitas belum dibuat. Baseline termal perlu diperiksa ulang. Tidak ada perubahan frontend, commit, atau push.
- Langkah berikutnya: Step 4 telah diizinkan; Step 5 belum diizinkan.

### 8 September 2026 — Step 4 (STOP sebelum smoke)

- Persetujuan: pengguna menjawab **oke lanjut** setelah usulan penutupan Step 3 dan Step 4; Step 5 tidak diizinkan.
- Tindakan: rekaman ulang sensor CPU, pemeriksaan guard live tanpa memuat dataset, dua sampel beban latar, arsip CSV, dan cleanup monitor/collector. Step 3 telah ditutup GO sebelumnya.
- Durasi rekaman: **610 detik**, 11:34:44–11:44:54 WIB. Penutupan administratif 2026-09-08T11:49:04.415733+07:00; fitting/probe aktif **0 detik**.
- Suhu awal/minimum/maksimum/rata-rata/akhir: **85.125 / 76.625 / 85.125 / 81.931 / 77.125°C**. Tidak ada dari 306 sampel unik di bawah 75°C.
- Sensor: 608 observasi, gap sumber maksimum 3 detik, umur maksimum 2,655 detik; cocok dengan arsip CSV. Identitas CPU die cocok dengan Step 1; tidak ada screenshot UI baru.
- RAM: peak collector **21.07 MiB**, sistem bebas minimum **10.74 GiB**; monitor **57.09 MiB** pada satu pengamatan, bukan peak seluruh sesi.
- Beban latar: CPU sistem rata-rata **20.80%**, 60 observasi akhir **14.55%**. Reason Security Engine Service memakai sekitar 13,34–16,14% total CPU pada dua sampel singkat; bukan diagnosis penyebab tunggal panas. Tidak ada layanan atau pengaturan keamanan/daya yang diubah.
- Guard: menolak mulai pada 84°C sebelum data dimuat. Callback saat fitting nyata masih belum divalidasi karena tidak ada fitting.
- Hasil: **STOP Step 4 karena syarat termal**, sensor recording valid. Tidak ada run operasional, label-fit, estimasi runtime trial, final training, atau test evaluation.
- Cleanup: collector exit 0; telemetry STOP ditolak guard; monitor milik sesi ini sudah ditutup memakai hak administrator setelah penutupan biasa ditolak OS; arsip CSV cocok SHA-256.
- Validasi: skrip analisis dan Ruff lulus; 85 checksum frozen, source/tes/YAML, 53 berkas laporan, dan frontend tetap utuh. Tidak ada commit/push; isi LaTeX tidak berubah.
- Artefak: reports/step4_20260908/README.md, thermal-validation.json, rekaman JSONL, CSV sumber, log guard, diagnosis beban, validation.json, completion.json; status pada README/IMPLEMENTATION_STATUS/plan diperbarui.
- Berikutnya: ulangi Step 4 setelah pemeriksaan termal memenuhi batas; jangan menaikkan batas atau lanjut Step 5 tanpa smoke yang valid dan izin terpisah.

### 8 September 2026 — Step 4 (attempt 02, STOP termal; runner siap)

- Persetujuan: pengguna memberi **oke lanjut**, lalu **continue**; scope ulang Step 4 dan persiapan runner. Step 5 tidak diizinkan.
- Pengamatan: 12:36:08–12:38:08 WIB, **120 detik**; ini pemeriksaan ulang singkat, bukan baseline baru 10 menit. Suhu min/max/akhir **76.375/80.500/77.625°C**; seluruh 60 sampel unik masih >=75°C.
- RAM pengamatan: peak collector **20.94 MiB**, RAM bebas minimum **10.20 GiB**. Suhu/peak RAM tes sukses tidak disampling.
- Guard menolak mulai pada 77,125°C sebelum data dimuat. Power plan Acer terverifikasi bertipe Balanced. Sampel beban latar rsEngineSvc.exe sekitar 8,89% total CPU; penyebab panas tidak disimpulkan.
- Implementasi ringan: run_smoke_probe.py, test_smoke_probe.py, SMOKE_RUNNER.md dan checksum tambahan di reports/step4_20260908. Satu guard 300 detik untuk kedua learner; semua unit/parameter mengikuti protokol. Tidak ada perubahan pipeline model lama.
- Tes: awal 7 lulus/1 gagal akibat resolusi timer monotonik Windows; diperbaiki memakai perf_counter untuk durasi unit. **8 tes sintetis lulus dalam 0,456 detik**; Ruff dan validate-only lulus. Tes memakai estimator palsu, tidak fitting model.
- Validasi: 85 berkas praregistrasi dan 3 file tambahan runner cocok; 14 bukti attempt awal, source/tes/YAML lama, 53 berkas laporan dan frontend tetap utuh.
- Hasil: **STOP Step 4 sebelum real-data smoke**; persiapan runner selesai. Probe aktif tetap 0 detik, estimasi runtime/jumlah trial belum tersedia.
- Cleanup: collector exit 0, telemetry STOP ditolak guard, monitor milik attempt ini sudah ditutup. Tidak ada perubahan keamanan/daya/BIOS, isi LaTeX, frontend, commit, atau push.
- Artefak: reports/step4_20260908/attempt-02/, latest.json, serta runner/dokumentasi/checksum pada folder induk. Penutupan 2026-09-08T16:31:31.904616+07:00.
- Berikutnya: selesaikan kondisi termal atau pilih resource lain, lalu ulangi Step 4. Jangan lanjut otomatis ke Step 5 atau menaikkan batas suhu.

### 8 September 2026 — Step 4 attempt 03 dan persiapan Step 5

- Persetujuan terbaru: **oke lanjut terus sampai step 5 juga**. Step 4 dan Step 5 diizinkan bersama; Step 6 dan seterusnya belum diizinkan. Status izin lama dalam log/artefak frozen tetap merupakan riwayat pada saat dibuat.
- Baseline baru: 2026-09-08T21:08:53.410799+07:00 sampai 2026-09-08T21:19:03.407614+07:00, **610 detik**. Suhu awal/minimum/maksimum/rata-rata/akhir **73,375/71,750/85,625/78,940/84,875°C**. Lima menit terakhir 79,625–85,625°C; rentang berturut-turut di bawah 75°C terpanjang 204 detik.
- Sensor valid: 304 sampel sumber unik, 608 observasi, umur maksimum 2,674 detik, gap maksimum 3 detik; seluruh observasi cocok arsip CSV. Peak RSS collector **21.09 MiB**, RAM sistem bebas minimum **12.52 GiB**; peak monitor tidak disampling.
- Guard menolak mulai pada **85,125°C** sebelum dataset dimuat. Probe aktif tetap **0 detik**, 0 label-fit, estimasi runtime null. Hasil **STOP Step 4**; kondisi termal belum memenuhi batas mulai <75°C.
- Beban latar CPU sistem rata-rata 20,93%, 60 observasi akhir 38,24%; sampel 10 detik rsEngineSvc.exe 9,04% total CPU. Power plan Balanced dan charger terhubung. Tidak disimpulkan penyebab tunggal panas; suhu ruang/kipas tidak diukur.
- Persiapan Step 5: checksum dataset/source/partisi/profil cocok, Python dan 14 dependency cocok. Aturan budget 4 jam/model tetap. **Step 5 diizinkan tetapi tertahan**; jumlah trial, run final dan ledger belum dibuat tanpa runtime smoke valid. Enforcement akumulasi waktu/trial lintas sesi masih harus disediakan sebelum tuning Step 8/9.
- Cleanup/validasi: collector exit 0; telemetry STOP ditolak guard; monitor PID 29908 milik attempt ini ditutup. 85 berkas protokol, 3 runner, 29 bukti attempt terdahulu dan 53 berkas laporan utuh; frontend bersih. Tidak ada pemuatan matriks dataset/test, perubahan source model, isi LaTeX, frontend, pengaturan sistem, commit, atau push oleh sesi ini.
- Artefak: reports/step4_20260908/attempt-03/, latest.json, reports/step5_20260908/. Penutupan 2026-09-08T21:23:28.175243+07:00.
- Berikutnya: setelah kondisi termal berubah, ukur ulang dan selesaikan Step 4 serta Step 5 dengan izin yang sudah diberikan. Jangan menaikkan batas atau mengarang runtime/jumlah trial untuk melewati gate.

### 8 September 2026 — Step 5 (snapshot calon run terpantau)

- Persetujuan: **yaudah oke sekarang kita lanjut aja step 5 tapi tetep monitor suhunya, jika ada potensi terjadi apa apa tolong langsung batalkan runnnya**. Pelaksanaan terbatas pada persiapan metadata; Step 6 dan seterusnya belum diizinkan.
- Tindakan: jalankan monitor/collector, lalu siapkan snapshot calon run nonoperasional dengan ResourceGuard. Batas suhu tetap safe-training; durasi guard dipersempit satu menit, satu thread. Tidak ada fitting atau pembukaan matriks development/test.
- Hasil persiapan: 47 snapshot cocok SHA-256; 85 berkas protokol dan Python/dependency/source cocok. candidate_run.json menyimpan partisi frozen dan profil final, training_allowed=false; budget_pending.json menyimpan jumlah trial/runtime null. Tidak ada run.json atau run operasional baru. Partisi belum diregenerasi oleh initializer final.
- Durasi aktif 21:37:19–21:37:21 WIB, **1,802 detik**. Suhu sampel guard awal/maksimum/akhir **75,375°C**; peak RSS **28,02 MiB**, RAM sistem bebas minimum **9,54 GiB**. Banyak check memakai sampel sumber yang sama; bukan jaminan pembacaan kontinu.
- Rekaman sensor terpisah: 21:34:21–21:37:21 WIB, **180 detik**, 90 sampel sumber/179 observasi, suhu 74,625–83,125°C, akhir 75,375°C. Peak collector 20,89 MiB, minimum RAM bebas seluruh rekaman 8,32 GiB. Bukan pengganti baseline 10 menit atau bukti smoke GO.
- Tidak ada stop terpicu selama persiapan. Collector exit 0 dan telemetry STOP ditolak guard. Monitor milik sesi PID 30952 ditutup; CSV diarsipkan dan cocok dengan seluruh observasi.
- Status **persiapan snapshot selesai**, tetapi **Step 5 belum GO** karena belum ada runtime smoke lengkap, jumlah trial, regenerasi partisi, budget freeze, atau run final. Step 4 tetap STOP. Tidak ada perubahan model utama, frontend, isi LaTeX, pengaturan daya/keamanan, commit, atau push.
- Artefak: reports/step5_20260908/attempt-20260908-213401/; status sebelumnya diarsipkan dalam history-20260908-212328/. Penutupan 2026-09-08T21:42:24.987767+07:00.
- Berikutnya: lengkapi smoke yang valid dan cooldown; jika dipisah per algoritme, amandemen runner lebih dahulu dengan cap 300 detik aktif gabungan tetap. Setelah itu selesaikan freeze dan inisialisasi Step 5 sesuai izin yang sudah ada.

## Prompt Singkat untuk Chat Berikutnya

Gunakan format berikut agar agent tidak menjalankan beberapa tahap sekaligus:

```text
Baca C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\SAFE_EXECUTION_PLAN.md secara penuh. Periksa kondisi terkini, lalu jalankan hanya Step <nomor>. Jangan lanjut ke step berikutnya. Setelah selesai, update Progress Tracker dan Log Eksekusi, kemudian laporkan hasil GO/STOP.
```
