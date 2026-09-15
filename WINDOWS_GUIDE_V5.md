# Panduan Grouped-v5 di PC Kampus

Gunakan **`perfume-campus-grouped-v5.zip`** dan ekstrak ke folder baru. V5 adalah
audit tuning terakhir untuk kondisi D pada XGBoost dan LightGBM. Jangan menimpa
folder v3 atau v4.

Paket sudah berisi lima sumber Pyrfume, dataset grouped, studi Optuna D dari v4,
serta incumbent v3. Tidak perlu clone GitHub, preprocessing, atau menyalin hasil
lama secara manual.

## Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_training.txt
.\RUN_GROUPED_V5.cmd
```

## Linux

```bash
unzip perfume-campus-grouped-v5.zip -d perfume-campus-grouped-v5
cd perfume-campus-grouped-v5
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements_training.txt
bash ./RUN_GROUPED_V5.sh
```

Gunakan Python **3.12**. Kedua algoritma memakai CPU dengan 6 thread dan berjalan
bergantian.

## Yang dijalankan

1. Memeriksa checksum dataset, seed v4, database Optuna, dan protokol.
2. Melanjutkan XGBoost D selama 4 jam baru.
3. Melanjutkan LightGBM D selama 4 jam baru.
4. Menyimpan pemakaian budget setiap 15 detik.
5. Menjalankan primary CV dan lima repeated grouped CV.
6. Membandingkan kandidat baru dengan incumbent v3.
7. Membekukan seleksi sebelum test set dibuka.
8. Menghitung metrik, threshold validasi, perbandingan seed, dan bootstrap per label.
9. Membuat ZIP hasil secara otomatis.

Tuning tambahan adalah **8 jam total**. Keseluruhan proses diperkirakan **11–17
jam**. Sambungkan PC ke listrik dan nonaktifkan sleep.

## Hasil yang dibawa pulang

Run lengkap:

```text
runs/perfume-five-grouped-v5/complete.json
```

ZIP hasil:

```text
campus-results/perfume-grouped-v5-results-*.zip
```

Jika ZIP belum dibuat setelah training selesai:

```powershell
.\COLLECT_GROUPED_V5_RESULTS.cmd
```

Pada Linux:

```bash
bash ./COLLECT_GROUPED_V5_RESULTS.sh
```

## Jika proses terputus

Pastikan proses Python lama sudah berhenti, lalu jalankan launcher yang sama dari
folder yang sama. Budget hanya dihitung sampai heartbeat terakhir, sehingga sisa
waktu tetap dapat dilanjutkan. Jangan mengubah kode, dependency, atau folder run.

Jalankan v5 satu kali sampai selesai. Setelah v5, hentikan penambahan tuning agar
tidak terus menyesuaikan eksperimen terhadap validation/test yang sama.
