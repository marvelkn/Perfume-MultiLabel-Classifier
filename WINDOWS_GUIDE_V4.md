# Panduan Grouped-v4 di PC Kampus

Gunakan **`perfume-campus-grouped-v4.zip`** dan ekstrak ke folder baru. Paket ini
melanjutkan Optuna XGBoost dan LightGBM dari grouped-v3 dengan tambahan budget
yang sama. Jangan menimpa folder v3 karena grouped-v3 tetap menjadi hasil utama.

Paket sudah berisi lima sumber Pyrfume, dataset grouped siap pakai, empat studi
Optuna v3, dan dua incumbent v3. Tidak perlu clone GitHub, menyalin hasil lama,
atau menjalankan preprocessing manual.

## Windows

1. Ekstrak ZIP ke folder baru.
2. Buka folder hasil ekstraksi.
3. Klik kanan area kosong, lalu pilih **Open in Terminal**.
4. Jalankan tiga perintah berikut satu per satu:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_training.txt
.\RUN_GROUPED_V4.cmd
```

## Linux

```bash
unzip perfume-campus-grouped-v4.zip -d perfume-campus-grouped-v4
cd perfume-campus-grouped-v4
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements_training.txt
bash ./RUN_GROUPED_V4.sh
```

Gunakan Python **3.12**. XGBoost dan LightGBM memakai CPU dengan 6 thread dan
berjalan bergantian. GPU tidak diperlukan.

## Yang dikerjakan otomatis

1. Memeriksa checksum dataset, split kelompok struktur, protokol, dan seed v3.
2. Melanjutkan xgb_C dan xgb_D masing-masing selama 2 jam tambahan.
3. Melanjutkan lgbm_C dan lgbm_D masing-masing selama 2 jam tambahan.
4. Menilai empat kandidat baru dengan lima repeated grouped cross-validation.
5. Membandingkan kandidat baru dengan incumbent XGBoost dan LightGBM v3.
6. Memilih satu kandidat per algoritma berdasarkan mean macro AUPRC validasi.
7. Membekukan pemilihan sebelum membuka test set.
8. Melatih final hanya jika kandidat baru mengalahkan incumbent pada validasi.
9. Menghitung enam metrik utama, threshold validasi, perbandingan berpasangan
   antarseed, serta bootstrap 10.000 kali pada hasil per label.
10. Membuat ZIP hasil secara otomatis.

Tambahan tuning aktif adalah **4 jam per algoritma** atau **8 jam total**. Total
proses diperkirakan sekitar **12–20 jam**, bergantung pada parameter yang dipilih
Optuna dan kecepatan CPU. Sambungkan PC ke listrik dan nonaktifkan sleep.

## File hasil

Run lengkap ditandai oleh:

```text
runs/perfume-five-grouped-v4/complete.json
```

Bawa pulang ZIP terbaru:

```text
campus-results/perfume-grouped-v4-results-*.zip
```

Jika training selesai tetapi ZIP belum dibuat, jalankan:

```powershell
.\COLLECT_GROUPED_V4_RESULTS.cmd
```

Pada Linux:

```bash
bash ./COLLECT_GROUPED_V4_RESULTS.sh
```

## Jika proses terputus

Pastikan proses Python lama sudah berhenti, lalu jalankan launcher yang sama dari
folder yang sama. Checkpoint trial, CV, dan model yang selesai akan digunakan lagi.
Jangan mengubah kode, dataset, dependency, atau nama folder run.

Jalankan v4 **satu kali sampai selesai**. Jangan mengulang lalu memilih hasil test
tertinggi. Grouped-v3 tetap menjadi hasil utama karena test set telah diperiksa
sebelum v4; grouped-v4 adalah analisis lanjutan dengan budget yang lebih besar.
