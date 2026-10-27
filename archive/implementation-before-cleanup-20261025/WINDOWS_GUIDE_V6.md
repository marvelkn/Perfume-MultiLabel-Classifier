# Jalankan v6 di Windows

## 1. Ekstrak ZIP

Salin `perfume-campus-grouped-v6.zip` ke PC lab, klik kanan > Extract All.
Gunakan folder baru, jangan menimpa v3/v4/v5. Buka folder yang berisi
`RUN_GROUPED_V6.cmd`, klik kanan area kosong > Open in Terminal (PowerShell).

Dataset sudah lengkap: lima sumber Pyrfume, hasil preprocessing, 109 label,
split kelompok, notebook, dan ringkasan pembanding versi lama. Tidak perlu clone,
download dataset, preprocessing ulang, atau menyalin model lama.

## 2. Instal sekali

Pastikan Python **3.12 64-bit** tersedia. Copy-paste:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_training.txt
```

Internet diperlukan saat memasang dependency. Jika instalasi gagal, perbaiki
pesan instalasinya dahulu; jangan mengganti versi paket secara acak.

## 3. Mulai seluruh pekerjaan

```powershell
.\RUN_GROUPED_V6.cmd
```

Satu perintah menjalankan pemeriksaan data singkat, empat studi Optuna baru,
validasi semua A/B/C/D, fitting model, evaluasi, perbandingan hasil lama, dan ZIP hasil.
Tidak ada training percobaan atau pengaturan sensor suhu yang perlu kamu lakukan.

### Yang berjalan otomatis

| Kombinasi | Perlakuan |
|---|---|
| XGBoost A/B | Baseline tetap 100 pohon; dievaluasi ulang |
| XGBoost C/D | Masing-masing 8 jam tuning |
| LightGBM A/B | Baseline tetap 100 pohon; dievaluasi ulang |
| LightGBM C/D | Masing-masing 8 jam tuning |

Keempat studi dimulai baru. Resep terbaik sebelumnya dicoba sebagai trial pertama,
dihitung ulang di dalam budget. Trial/skor historis tidak diimpor. Pencarian
memperluas pohon hingga 1.600, learning rate hingga 0,005, dan proporsi fitur
hingga 0,30. Delapan kandidat mendapat lima pengulangan grouped CV dengan lima
fold per label. Pemilihan model dan threshold dikunci sebelum evaluasi test.

**Tuning saja 32 jam.** Validasi dan fitting menambah waktu; alokasikan sekitar
**2–3 hari atau lebih**, tergantung CPU dan parameter terpilih. Ini estimasi,
bukan jaminan. CPU 6 thread, model berjalan bergantian. GPU tidak digunakan.
Sediakan minimal 10 GB ruang kosong, RAM memadai (profil lab 32 GB), listrik
stabil, nonaktifkan sleep, dan biarkan terminal terbuka.

## 4. Lihat progres atau lanjutkan setelah terputus

Progres tuning tampil kira-kira setiap menit; tahap CV/fitting dapat berjalan
lebih lama tanpa pesan. Status terakhir ada di `runs/perfume-five-grouped-v6/progress.json`
dan riwayat di `events.jsonl`. Untuk membaca progres dari terminal kedua:

```powershell
Get-Content .\runs\perfume-five-grouped-v6\progress.json
```

Untuk berhenti teratur gunakan Ctrl+C sekali, tunggu proses selesai berhenti.
Untuk melanjutkan, buka folder yang sama pada **PC yang sama**, lalu:

```powershell
.\RUN_GROUPED_V6.cmd
```

Jangan instal ulang, mengubah kode/dependency, menghapus run, atau menjalankan
dua launcher bersamaan. Trial yang belum selesai dapat gugur; waktu yang sudah
dipakai tetap tercatat. Saat crash mendadak, pencatatan budget dapat tertinggal
sekitar satu interval heartbeat (15 detik). Trial selesai tetap tersedia.
CV dan fitting melanjutkan checkpoint; run lengkap tidak dilatih ulang.

## 5. Bawa pulang hasil

Setelah muncul `SELESAI`, bawa ZIP terbaru:

```text
campus-results/perfume-grouped-v6-results-*.zip
```

Jika training selesai tetapi pembuatan ZIP gagal, jalankan:

```powershell
.\COLLECT_GROUPED_V6_RESULTS.cmd
```

Ringkasan mudah dibaca ada di `runs/perfume-five-grouped-v6/RESULTS_SUMMARY.md`.
Tabel lengkap: `comparison_all.csv`, `ablation_deltas.csv`, `comparison_history.csv`.
Model dan threshold tersedia; `inference_bundle.json` menunjuk kandidat terpilih.

## Makna hasil

V6 menguji pencarian yang lebih luas; skor lebih tinggi tidak dijamin. Test set
pernah diperiksa pada versi lama, sehingga hasil tetap **eksploratori**. Lima
pengulangan CV ini bukan nested CV atau pengganti test independen. Data tanpa label
dan pseudo-label belum ditambahkan ke training. Tahap berikutnya adalah pelaporan
dan penguncian model, bukan otomatis membuat putaran tuning baru.

Panduan Linux: [LINUX_GUIDE_V6.md](LINUX_GUIDE_V6.md).
