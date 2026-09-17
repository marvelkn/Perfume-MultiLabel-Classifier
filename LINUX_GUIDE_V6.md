# Jalankan v6 di Linux

Gunakan Python **3.12**, folder lokal baru, dan paket `perfume-campus-grouped-v6.zip`.
Dataset sudah tersedia. Internet hanya dibutuhkan untuk instalasi dependency.

```bash
unzip perfume-campus-grouped-v6.zip -d perfume-campus-grouped-v6
cd perfume-campus-grouped-v6
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements_training.txt
bash RUN_GROUPED_V6.sh
```

Jika `python3.12` atau modul `venv` belum tersedia, minta pengelola lab memasangnya
sesuai distribusi Linux. Jika LightGBM melaporkan `libgomp.so.1` tidak ditemukan,
pasang runtime OpenMP distribusi tersebut (Ubuntu/Debian: `libgomp1`). Jangan
mengganti versi Python/dependency untuk melewati kegagalan instalasi.

Tuning C/D dua algoritma membutuhkan **32 jam**, ditambah validasi dan fitting
seluruh A/B/C/D. Alokasikan sekitar **2–3 hari atau lebih**. CPU 6 thread, berjalan
berurutan; GPU tidak digunakan. Sediakan minimal 10 GB disk dan RAM memadai.
Nonaktifkan sleep. Biarkan sesi terminal tetap hidup; bila lab sudah menyediakan
`tmux`, kamu dapat menjalankan proses di dalam sesi tmux agar tidak bergantung
pada jendela terminal. Tetap jangan mematikan PC atau membiarkannya sleep.

Melihat progres dari terminal lain:

```bash
cat runs/perfume-five-grouped-v6/progress.json
```

Setelah interupsi, pastikan Python lama berhenti, gunakan PC/folder yang sama:

```bash
bash RUN_GROUPED_V6.sh
```

Jangan menjalankan ulang perintah instalasi untuk resume. Jangan mengubah kode,
dependency, atau memindahkan run antara Windows dan Linux di tengah training.
Trial selesai disimpan; trial belum selesai dapat gugur dan waktu terpakainya
tetap dihitung. Budget dicatat tiap 15 detik.

Hasil otomatis: `campus-results/perfume-grouped-v6-results-*.zip`.
Jika koleksi perlu diulang sesudah training berhenti:

```bash
bash COLLECT_GROUPED_V6_RESULTS.sh
```

Rincian eksperimen, hasil, dan batas klaim sama dengan [WINDOWS_GUIDE_V6.md](WINDOWS_GUIDE_V6.md).
