# 03 — DATA ANGKA SIAP TRAINING: MORGAN 2048 + 8 DESKRIPTOR

**Ini data akhirnya, belum model yang sudah dilatih.** File train dan test sudah dipasangkan dengan target 109 label dan nomor molekul yang tepat.

| File | Isi |
|---|---|
| `train_2048_8.npz` | Data latihan: X **(5353,2056)** dan y **(5353,109)** |
| `test_2048_8.npz` | Data pengecekan: X **(1333,2056)** dan y **(1333,109)** |
| `target_train_109_label.csv` | Nomor molekul, SMILES, dan 109 kolom jawaban 0/1 train; gampang dibuka di Excel |
| `target_test_109_label.csv` | Bentuk yang sama untuk test |
| `descriptors.csv` | Delapan deskriptor untuk semua 6.686 molekul, disalin dari file fitur asli |
| `features_semua_molekul_2048_8.npz` | Morgan dan deskriptor semua molekul, disalin persis dari fitur utama |
| `labels_109.json` | Nama 109 aroma dalam urutan kolom y |
| `nama_fitur_2056.json` | Nama kolom X: 2048 posisi Morgan, lalu delapan deskriptor |
| `pengaturan_eksperimen.json` | Salinan pengaturan A–D, Optuna, validasi dan threshold dari jalur aktif |

## Isi file train/test NPZ

| Nama di dalam file | Artinya |
|---|---|
| `X` | Angka masukan model: Morgan 2048 diikuti delapan deskriptor, tipe float32 |
| `y` | Jawaban aroma 0/1, 109 kolom. Ini sama dengan `truth` pada file target asli |
| `morgan` | Morgan saja, 2048 kolom; untuk skenario A/C |
| `descriptors` | Delapan angka deskriptor dalam urutan tetap |
| `row_index`, `source_row` | Penunjuk molekul agar barisnya bisa dilacak ke records |
| `group_id` | ID kelompok pada pembagian yang sudah dipakai penelitian |
| `labels` | Nama aroma sesuai urutan kolom y |
| `feature_names`, `descriptor_names` | Nama fitur sesuai urutan kolomnya |

**Skenario A/C memakai `morgan`; B/D memakai `X`.** Nomor molekul, group_id, SMILES, dan nama sumber bukan kolom fitur yang ditambahkan ke X. CSV target menampilkan SMILES untuk pencarian, bukan supaya seluruh CSV dimasukkan ke model.

## Coba buka tanpa training

Jalankan contoh ini dari folder `DATA_UNTUK_BELAJAR`:

```python
import numpy as np

with np.load("03_SIAP_TRAINING_2048_8/train_2048_8.npz", allow_pickle=False) as data:
    print("Fitur:", data["X"].shape)       # (5353, 2056)
    print("Target:", data["y"].shape)      # (5353, 109)
    print("Nama aroma:", data["labels"][:5])
```

Ini cuma membaca file, bukan melatih model. Gunakan test untuk mengecek setelah pilihan model/pengaturan ditetapkan, bukan ikut melatih atau memilih parameter.

## Apa yang dilakukan saat membuat paket ini?

Fitur Morgan 2048 dan delapan deskriptor **diambil dari file fitur utama yang sudah tersimpan**. Target dan indeks train/test diambil dari file data yang sudah dipakai penelitian, lalu dipasangkan dengan fitur utama pada nomor baris yang sama.

File train/test lama tidak disalin mentah sebagai data final di folder ini karena masih membawa fitur versi dasar. File **train_2048_8.npz** dan **test_2048_8.npz** di sini adalah paket baru dengan fitur yang benar. Tidak ada perubahan molekul, label, grup, split, atau penghitungan fitur ulang.

Semua baris tetap ada, termasuk **11 train dan 4 test** yang targetnya semua nol setelah pemilihan 109 label. Nol berarti aroma tidak tercatat, bukan kepastian bahwa manusia tidak akan mencium aroma itu.

Paket ini bisa dibaca untuk belajar/training baru, tetapi bukan pengganti otomatis loader penelitian. Jalur program utama dan hasil model lama tidak diubah. Tidak ada training atau prediksi yang dijalankan saat paket disiapkan.
