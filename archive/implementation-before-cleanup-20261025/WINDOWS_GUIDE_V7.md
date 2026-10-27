# Jalankan v7 Morgan 2048 di Windows

**Pakai perfume-campus-grouped-v7-m2048-d8.zip.**
Dataset, label, pembagian data dan fitur sudah lengkap. Tidak perlu clone/preprocessing ulang.

## 1. Ekstrak ZIP

Extract All ke folder baru, misalnya Desktop\perfume-campus-grouped-v7-m2048-d8.
Buka folder berisi **RUN_GROUPED_V7.cmd**, klik kanan > Open in Terminal.
Jangan menimpa v6 atau v7 lama; run lama tidak bisa dilanjutkan dengan versi ini.

## 2. Instal sekali

Gunakan Python **3.12 64-bit**. Copy-paste di PowerShell:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_campus_v7.txt
~~~

Tunggu instalasi berhasil. Internet dibutuhkan untuk instalasi.
Jika py tidak ditemukan, pasang Python 3.12 64-bit atau minta petugas lab menyiapkannya.

## 3. Jalankan semua

~~~powershell
.\RUN_GROUPED_V7.cmd
~~~

Otomatis: pemeriksaan singkat, tuning, validasi, training final, evaluasi,
tabel perbandingan, lalu ZIP hasil.

| Kondisi untuk XGBoost dan LightGBM | Input | Training |
|---|---|---|
| A | Morgan 2048 | Baseline |
| B | Morgan 2048 + 8 (2056 fitur) | Baseline |
| C | Morgan 2048 | Optuna 8 jam |
| D | Morgan 2048 + 8 (2056 fitur) | Optuna 8 jam |

**8 kandidat, 4 studi Optuna. 32 jam hanya tuning**, belum termasuk validasi
lima seed dan fitting final; total bisa beberapa hari. Tidak perlu command per model.

Profil PC kampus: CPU 6 thread, model berurutan, tanpa GPU.
Sediakan minimal 10 GB ruang kosong; 20 GB atau lebih lebih nyaman untuk arsip.
Runner memeriksa RAM dan disk. Jangan biarkan PC sleep; hindari pekerjaan berat lain.

## 4. Berhenti atau lanjut

Tekan Ctrl+C dan tunggu proses keluar. Untuk lanjut:

~~~powershell
.\RUN_GROUPED_V7.cmd
~~~

Gunakan PC/folder yang sama. Jangan hapus runs/.venv atau mengganti kode,
konfigurasi dan dependency di tengah run. Trial terputus bisa diulang sebagai trial
baru; pekerjaan selesai dan budget tercatat dipertahankan.

## 5. Bawa pulang hasil

Setelah **SELESAI**, salin ZIP terbaru di folder **campus-results**:
**perfume-grouped-v7-m2048-d8-results-TANGGAL-JAM.zip**.
Model, dataset, skor, konfigurasi dan checksum ada di dalamnya.

Untuk mengemas hasil sementara, hentikan training dengan rapi, lalu:

~~~powershell
.\COLLECT_GROUPED_V7_RESULTS.cmd
~~~

Lock setelah crash dipulihkan lewat launcher; Ctrl+C dan tunggu keluar sebelum collect.
Jangan collect ketika training aktif.

Ringkasan: runs/perfume-five-grouped-v7-m2048-d8/RESULTS_SUMMARY.md.
Rencana: DESCRIPTOR_COMPARISON_V7_PLAN.md.
Agenda setelah hasil selesai: AFTER_V7_REPORT_UPDATE.md.
