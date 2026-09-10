# Panduan PC Kampus - Windows atau Linux

Gunakan hanya **perfume-campus-grouped-v2.zip**. Jangan gunakan
`essenza-campus-overlay.zip`, `RUN_CAMPUS_ALL.cmd`, atau dataset `perfume-five-v1`.

Paket sudah berisi kode, lima sumber Pyrfume, dan dataset siap pakai:
**6.686 molekul, 5.353 data latih, 1.333 data uji, 109 label**.
Anda tidak perlu clone GitHub dan tidak perlu menjalankan preprocessing manual.

## Pilih salah satu sistem operasi

### A. Windows

1. Salin ZIP ke PC kampus lalu ekstrak ke folder baru.
2. Buka folder hasil ekstraksi.
3. Klik kanan area kosong, lalu pilih **Open in Terminal** atau buka PowerShell di folder itu.
4. Salin dan jalankan tiga perintah berikut satu per satu:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_training.txt
.\RUN_PERFUME_FIVE.cmd
```

Jika perintah pertama menyatakan Python 3.12 tidak ditemukan, instal Python 3.12 64-bit
atau minta bantuan pengelola laboratorium. Jangan lanjut memakai versi Python lain.

### B. Linux

1. Salin ZIP ke PC kampus.
2. Buka Terminal pada folder tempat ZIP berada.
3. Jalankan perintah berikut satu per satu:

```bash
unzip perfume-campus-grouped-v2.zip -d perfume-campus-grouped-v2
cd perfume-campus-grouped-v2
python3.12 --version
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements_training.txt
bash ./RUN_PERFUME_FIVE.sh
```

Perintah `python3.12 --version` harus menampilkan Python 3.12.x. Jika `python3.12`,
`venv`, atau `unzip` tidak tersedia, minta pengelola laboratorium memasangnya.

## Apa yang berjalan otomatis?

Launcher menjalankan urutan berikut:

1. memeriksa checksum dataset dan memastikan tidak ada kelompok bocor;
2. menjalankan XGBoost kondisi A-D;
3. menjalankan LightGBM kondisi A-D;
4. melakukan tuning Optuna untuk kondisi C dan D;
5. membuat model final dan mengevaluasi data uji;
6. membuat ZIP hasil di folder `campus-results`.

| Kondisi | Fitur | Pengaturan |
| --- | --- | --- |
| A | Morgan 1.024 bit | baseline 100 pohon |
| B | Morgan + 5 deskriptor | baseline 100 pohon |
| C | Morgan 1.024 bit | Optuna |
| D | Morgan + 5 deskriptor | Optuna |

XGBoost dan LightGBM berjalan bergantian dengan dua thread. Tuning maksimal empat jam
per algoritma. Baseline, validasi ulang, fitting final, dan evaluasi berada di luar waktu tuning,
sehingga total proses dapat melebihi delapan jam. Jangan tutup terminal dan matikan mode sleep.

## Setelah selesai

Proses lengkap ditandai oleh file:

```text
runs/perfume-five-grouped-v2/complete.json
```

Ambil ZIP terbaru dari folder:

```text
campus-results
```

ZIP tersebut berisi model, parameter Optuna, hasil cross-validation, prediksi, metrik,
dataset, split, dan informasi versi kode. Bawa ZIP itu kembali untuk dianalisis.

Jika launcher selesai tetapi ZIP hasil belum ada, jalankan:

```powershell
.\COLLECT_CAMPUS_RESULTS.cmd
```

atau pada Linux:

```bash
bash ./COLLECT_CAMPUS_RESULTS.sh
```

## Jika proses terputus

Jalankan launcher yang sama dari folder yang sama. Proses akan melanjutkan artefak yang sudah ada.
Jangan mengubah kode, dataset, konfigurasi, dependensi, atau nama folder run.

Jika komputer mati mendadak, pastikan tidak ada proses Python lama yang masih berjalan. Setelah itu,
hapus hanya file berikut sebelum menjalankan ulang:

```text
runs/perfume-five-grouped-v2/.alignment.lock
```

Jangan hapus folder run atau database Optuna.

## Catatan metode

Semua molekul tetap digunakan. Struktur tanpa stereokimia yang sama atau fingerprint Morgan identik
ditempatkan dalam kelompok yang sama. Kelompok tidak boleh tersebar antara data latih dan uji,
atau antara data latih dan validasi. Seluruh kondisi XGBoost dan LightGBM memakai pembagian yang sama.

Lima sumber relevan dengan bahan pewangi, tetapi tidak diklaim berisi bahan parfum secara eksklusif.
Fokus penerapan penelitian berada pada prediksi aroma molekul dalam konteks parfum.
