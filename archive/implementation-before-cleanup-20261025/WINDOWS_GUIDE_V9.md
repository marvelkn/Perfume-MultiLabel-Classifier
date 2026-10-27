# Menjalankan V9 di Windows

## 1. Salin lalu ekstrak

Salin **perfume-campus-grouped-v9-bits-d8.zip** ke disk lokal PC kampus.
Klik kanan > **Extract All**. Masuk ke folder yang berisi **RUN_GROUPED_V9.cmd**.
Gunakan folder baru; jangan gabungkan dengan v7 atau v8.

Data mentah lima sumber, data bersih, 109 label, fitur, dan split sudah ada.
Tidak perlu clone GitHub, download dataset atau menjalankan preprocessing.

## 2. Pastikan Python tersedia

Gunakan **Python 3.12 64-bit** beserta Python Launcher.
Kalau belum terpasang, minta petugas lab memasangnya atau gunakan installer
[Python untuk Windows](https://www.python.org/downloads/windows/).

## 3. Jalankan satu file

**Klik dua kali RUN_GROUPED_V9.cmd.** Script akan memasang dependency pertama
kali, lalu menjalankan seluruh eksperimen dan membuat ZIP hasil.
Internet diperlukan saat instalasi pertama saja.

Kalau memakai PowerShell, buka terminal di folder hasil ekstrak lalu:

```powershell
.\RUN_GROUPED_V9.cmd
```

Otomatis: pemeriksaan integritas data singkat, baseline, tuning, validasi,
fitting final, evaluasi test dan pengemasan. Tidak ada pemeriksaan suhu manual
atau command terpisah per model.

**4 studi × 16 jam = 64 jam tuning.** Validasi 5 seed, baseline dan fitting
final dihitung terpisah. Analisis konfigurasi jam ke-8 juga membutuhkan CV
jika konfigurasinya berbeda dari jam ke-16. Total dapat berlangsung beberapa hari.
Pastikan PC tidak sleep/restart otomatis dan sediakan sekitar 20 GB disk kosong.

Profil: **CPU 6 thread**, model berurutan. Kompatibel dengan PC kampus i7-8700K,
RAM 32 GB. GPU tidak digunakan; jangan mengganti CPU/GPU atau parameter di tengah run.

## 4. Kalau berhenti atau terputus

Tekan **Ctrl+C**, lalu tunggu proses berhenti. Untuk melanjutkan, jalankan file
yang sama pada **PC dan folder yang sama**. Hasil yang selesai dipertahankan;
trial/fold yang terputus mungkin dihitung ulang. Budget tuning tidak direset.
Setelah crash, launcher memulihkan lock jika proses sebelumnya sudah tidak aktif.
Jangan menghapus `runs`, `.venv`, database Optuna atau mengganti dependency.

Progres terakhir bisa dibaca di `runs/perfume-five-grouped-v9-bits-d8/progress.json`.
Saat CV/fitting, baris terminal tidak selalu berubah setiap detik; hasil CV
disimpan bertahap per label dan model final disimpan per label.

## 5. Bawa pulang hasil

Setelah muncul **SELESAI**, salin ZIP terbaru di **campus-results/**:
`perfume-grouped-v9-bits-d8-results-TANGGAL-JAM.zip`.
ZIP berisi model, prediksi, Optuna, tabel skor, data dan konfigurasi.

Untuk menyimpan hasil sementara setelah training dihentikan dengan rapi:

```powershell
.\COLLECT_GROUPED_V9_RESULTS.cmd
```

Ringkasan utama: `runs/perfume-five-grouped-v9-bits-d8/RESULTS_SUMMARY.md`.
Perbandingan 8 vs 16 jam: `budget_comparison.csv` pada folder run yang sama.

V7 tetap dikunci. Jangan menyalin hasil V9 ke rilis aplikasi atau laporan utama
sebelum hasilnya diaudit.
