# Panduan Menjalankan ML di PC Kampus

Gunakan file **`perfume-campus-grouped-v3.zip`**. Ekstrak ke folder baru.
Jangan menimpa folder atau hasil `grouped-v2`; hasil v2 tetap disimpan sebagai
eksperimen pertama.

Paket v3 sudah berisi kode, lima sumber Pyrfume, dan dataset siap pakai:
**6.686 molekul, 5.353 data latih, 1.333 data uji, dan 109 label**.
Tidak perlu clone GitHub dan tidak perlu menjalankan preprocessing manual.

Paket disetel untuk PC kampus **Intel Core i7-8700K, RAM 32 GB**.
XGBoost dan LightGBM memakai **6 thread CPU** dan berjalan bergantian.
GPU tidak dipakai agar kedua algoritma dibandingkan pada kondisi komputasi yang sama.

## Windows

1. Ekstrak ZIP ke folder baru.
2. Buka folder hasil ekstraksi.
3. Klik kanan area kosong, lalu pilih **Open in Terminal**.
4. Jalankan tiga perintah ini satu per satu:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_training.txt
.\RUN_PERFUME_FIVE.cmd
```

Jika Python 3.12 tidak ditemukan, instal Python 3.12 64-bit atau minta bantuan
pengelola laboratorium. Jangan memakai versi Python lain karena versi dependency dikunci.

## Linux

Buka terminal di lokasi ZIP, lalu jalankan:

```bash
unzip perfume-campus-grouped-v3.zip -d perfume-campus-grouped-v3
cd perfume-campus-grouped-v3
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements_training.txt
bash ./RUN_PERFUME_FIVE.sh
```

Jika `python3.12`, `venv`, atau `unzip` tidak tersedia, minta pengelola laboratorium
memasangnya.

## Proses yang berjalan otomatis

1. Memeriksa checksum dataset dan kebocoran kelompok struktur.
2. Menjalankan XGBoost dan LightGBM pada kondisi A-D.
3. Memberi bobot kelas yang dihitung dari bagian data latih setiap fold dengan
   aturan yang sama untuk kedua algoritma.
4. Menjalankan Optuna selama **2 jam untuk C** dan **2 jam untuk D** pada setiap
   algoritma, sehingga total tuning aktif adalah **4 jam per algoritma**.
5. Memilih satu kondisi terbaik dari tiap algoritma memakai macro AUPRC validasi.
6. Membandingkan dua finalis dengan **5 kali grouped cross-validation**, masing-masing
   maksimum 5 fold.
7. Memilih threshold tiap label dari prediksi out-of-fold validasi.
8. Membekukan pemilihan model, baru membuka data uji.
9. Menyimpan enam metrik utama pada threshold 0,5 dan hasil threshold validasi
   sebagai analisis tambahan.
10. Membuat ZIP hasil di folder `campus-results`.

| Kondisi | Fitur | Parameter model |
| --- | --- | --- |
| A | Morgan 1.024 bit | baseline 100 pohon |
| B | Morgan + 5 deskriptor RDKit | baseline 100 pohon |
| C | Morgan 1.024 bit | hasil Optuna |
| D | Morgan + 5 deskriptor RDKit | hasil Optuna |

Optuna memakai batas **waktu aktif**, bukan lagi berhenti setelah 15 trial.
Total proses diperkirakan sekitar **9-12 jam**, karena fitting baseline, validasi,
repeated CV, dan fitting final berada di luar delapan jam tuning. Sambungkan PC ke
listrik, tutup aplikasi berat, dan nonaktifkan sleep sementara.

## Tanda selesai dan file yang dibawa pulang

Proses lengkap ditandai oleh:

```text
runs/perfume-five-grouped-v3/complete.json
```

Ambil ZIP terbaru yang namanya diawali:

```text
campus-results/perfume-grouped-v3-results-
```

ZIP hasil berisi model, trial Optuna, metrik tiap fold, prediksi out-of-fold,
threshold validasi, hasil test, dataset, split, dan versi kode.

Jika training selesai tetapi ZIP hasil belum dibuat, jalankan:

```powershell
.\COLLECT_CAMPUS_RESULTS.cmd
```

Pada Linux:

```bash
bash ./COLLECT_CAMPUS_RESULTS.sh
```

## Jika proses terputus

Pastikan proses Python lama sudah berhenti, lalu jalankan launcher yang sama dari
folder yang sama. Runner akan melanjutkan artefak yang sudah selesai dan
membersihkan lock lama secara otomatis. Jangan mengubah kode, dependency, dataset,
atau nama folder run.

Jalankan protokol v3 ini **satu kali sampai selesai**. Jangan menjalankan berkali-kali
lalu memilih hasil test tertinggi karena itu membuat hasil penelitian bias.

## Catatan interpretasi

Enam metrik utama tetap dihitung pada threshold **0,5** agar konsisten dengan
perbandingan yang sudah dikunci. Threshold hasil validasi dilaporkan terpisah untuk
menunjukkan pengaruh keputusan multi-label.

Karena data uji yang sama sudah pernah dibuka pada eksperimen v2, hasil v3 adalah
analisis lanjutan. Hasil v2 tetap menjadi evaluasi konfirmatori awal dan tidak boleh
dihapus.
