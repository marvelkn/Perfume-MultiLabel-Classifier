# Menjalankan V9 di Linux

1. Salin **perfume-campus-grouped-v9-bits-d8.zip** ke disk lokal, lalu ekstrak.
2. Buka terminal di folder yang berisi **RUN_GROUPED_V9.sh**.
3. Pastikan **Python 3.12**, modul **venv**, dan pustaka **OpenMP/libgomp** tersedia.
4. Jalankan:

```bash
bash RUN_GROUPED_V9.sh
```

Script membuat `.venv`, memasang dependency, menjalankan seluruh eksperimen,
lalu membuat ZIP hasil. Data sudah lengkap; internet diperlukan untuk instalasi
dependency pertama saja. Tidak perlu menjalankan preprocessing atau perintah per model.

Jika Python/venv/libgomp belum tersedia pada Ubuntu/Debian yang menyediakan
Python 3.12, minta petugas lab menyiapkannya. Contoh perintah administrator:

```bash
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv libgomp1
```

Nama paket mengikuti distribusi; jangan mengganti versi Python secara sembarang
jika paket 3.12 tidak tersedia. Native Linux belum diuji di PC lab saat paket disiapkan;
runner memakai komponen lintas platform dan sintaks Bash diperiksa.

**64 jam hanya total tuning**; baseline, CV 5 seed, snapshot 8 jam, dan fitting
final menambah waktu. CPU 6 thread, model berurutan; GPU tidak digunakan.
Sediakan sekitar 20 GB disk kosong dan cegah sleep/restart/logout yang mematikan proses.

Untuk berhenti: **Ctrl+C**, tunggu proses keluar. Untuk resume pada PC/folder sama:

```bash
bash RUN_GROUPED_V9.sh
```

Jangan mengganti kode, dependency, CPU/GPU atau menghapus `runs/` di tengah run.
Budget dan checkpoint dipertahankan. Tidak ada pemeriksaan suhu manual.

Setelah **SELESAI**, bawa ZIP terbaru dari **campus-results/**.
Untuk mengemas hasil sementara setelah proses berhenti:

```bash
bash COLLECT_GROUPED_V9_RESULTS.sh
```

Ringkasan: `runs/perfume-five-grouped-v9-bits-d8/RESULTS_SUMMARY.md`.
Rancangan: [V9_EXPERIMENT_PLAN.md](V9_EXPERIMENT_PLAN.md).
**V7 tetap hasil utama; V9 eksplorasi tambahan.**
