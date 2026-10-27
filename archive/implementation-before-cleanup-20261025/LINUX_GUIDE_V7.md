# Jalankan v7 Morgan 2048 di Linux

Gunakan **perfume-campus-grouped-v7-m2048-d8.zip** yang sama dengan Windows.
Dataset, fitur dan pembagian data sudah ada.

## 1. Ekstrak ZIP

Ekstrak ke folder baru lewat file manager. Open in Terminal di folder yang
berisi **RUN_GROUPED_V7.sh**. Jangan menimpa v6 atau v7 lama.

## 2. Instal sekali

Gunakan Python **3.12**:

~~~bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements_campus_v7.txt
~~~

Tunggu instalasi berhasil. Jika Python/venv belum tersedia pada Ubuntu/Debian:

~~~bash
sudo apt update
sudo apt install python3.12 python3.12-venv libgomp1
~~~

Nama/ketersediaan paket bergantung distro. Kalau tidak ditemukan atau tidak punya
sudo, minta petugas lab menyiapkan Python 3.12 dan libgomp. Jangan mengganti
versi dependency acak. Internet diperlukan untuk instalasi.

## 3. Jalankan semua

~~~bash
bash RUN_GROUPED_V7.sh
~~~

| Kondisi untuk XGBoost dan LightGBM | Input | Training |
|---|---|---|
| A | Morgan 2048 | Baseline |
| B | Morgan 2048 + 8 (2056 fitur) | Baseline |
| C | Morgan 2048 | Optuna 8 jam |
| D | Morgan 2048 + 8 (2056 fitur) | Optuna 8 jam |

8 kandidat, 4 studi baru: **32 jam tuning**, ditambah validasi lima seed
dan fitting final; total dapat beberapa hari. CPU 6 thread, berurutan, tanpa GPU.
Sediakan minimal 10 GB disk kosong; 20 GB atau lebih lebih nyaman.
Jangan biarkan PC sleep.

## 4. Berhenti dan resume

Ctrl+C dan tunggu proses keluar. Lanjut di PC/folder yang sama:

~~~bash
bash RUN_GROUPED_V7.sh
~~~

Jangan hapus runs/.venv atau mengganti kode, protokol, dependency.
Jangan resume run v7 lama 1024 dengan paket ini.

## 5. Ambil hasil

Setelah **SELESAI**, bawa ZIP terbaru di **campus-results**:
**perfume-grouped-v7-m2048-d8-results-TANGGAL-JAM.zip**.
Model, dataset, skor, konfigurasi dan checksum sudah disertakan.

Untuk mengemas hasil sementara setelah training benar-benar berhenti:

~~~bash
bash COLLECT_GROUPED_V7_RESULTS.sh
~~~

Lock setelah crash dipulihkan lewat launcher; hentikan rapi sebelum collect.
Ringkasan: runs/perfume-five-grouped-v7-m2048-d8/RESULTS_SUMMARY.md.
Rencana: DESCRIPTOR_COMPARISON_V7_PLAN.md.
Agenda laporan: AFTER_V7_REPORT_UPDATE.md.
