# Data buat belajar — sebelum preprocessing sampai siap training

Folder ini berisi **salinan data penelitian**, supaya kamu nggak perlu mencari ke banyak tempat.
File aslinya tetap ada. Notebook dan program utama masih membaca lokasi asli, bukan otomatis berpindah ke folder ini.

## Mulai dari file yang mana?

| Kalau mau melihat... | Buka ini |
|---|---|
| Bentuk data sebelum dibersihkan | [01_SEBELUM_PREPROCESSING](01_SEBELUM_PREPROCESSING/) |
| Daftar molekul setelah dibersihkan dan digabung | [records.csv](02_SETELAH_PREPROCESSING/records.csv) |
| Target aroma 0/1 pada data latihan | [target_train_109_label.csv](03_SIAP_TRAINING_2048_8/target_train_109_label.csv) |
| Delapan angka deskriptor | [descriptors.csv](03_SIAP_TRAINING_2048_8/descriptors.csv) |
| Semua angka X dan y yang siap dimuat untuk training | [train_2048_8.npz](03_SIAP_TRAINING_2048_8/train_2048_8.npz) |

```text
01_SEBELUM_PREPROCESSING    13.337 catatan aroma dari lima sumber
           ↓ cek identitas, struktur dan label
                           11.777 diterima; 1.560 ditolak
           ↓ gabungkan catatan molekul yang sama
02_SETELAH_PREPROCESSING     6.686 molekul dalam records.csv
           ↓ pakai pembagian dan 109 label yang sudah ditetapkan
03_SIAP_TRAINING_2048_8      train: 5.353 molekul; test: 1.333 molekul
```

**Ini bukan hasil eksperimen baru.** Tidak ada training, tuning, prediksi model, atau penghitungan fitur molekul ulang ketika paket ini dibuat. Fitur dan targetnya diambil dari hasil yang sudah tersimpan, lalu dipasangkan sesuai nomor baris yang sama.

## Contoh yang gampang ditunjukkan ke dosen: vanilin

1. Buka `01_SEBELUM_PREPROCESSING/goodscents/molecules.csv`, cari CID **1183** atau nama **vanillin**.
2. Buka `goodscents/stimuli.csv`, cari Stimulus **121-33-5** yang terhubung ke CID tadi.
3. Buka `goodscents/behavior.csv`, cari Stimulus yang sama. Label GoodScents-nya: **sweet;vanilla;creamy;chocolate**.
4. Buka `02_SETELAH_PREPROCESSING/records.csv`, cari SMILES **COc1cc(C=O)ccc1O** atau `source_row` **5737**.
5. Di hasil gabungan, labelnya **caramel, chocolate, creamy, gourmand, powdery, sweet, vanilla**, dari lima sumber. Ini catatan dataset, bukan skor prediksi model.

SMILES sebelum/sesudah bisa berbeda tulisannya, tetapi strukturnya sama. RDKit menyeragamkan penulisannya.

## Cara buka

- CSV bisa dibuka lewat VS Code atau Excel. Kalau di Excel semua isinya masuk satu kolom, gunakan **Data → From Text/CSV** dan pilih pemisah **koma**.
- JSON berisi catatan aturan, penolakan, atau daftar pembagian. Buka di editor teks.
- NPZ berisi array NumPy, jadi dibuka dengan Python, bukan Excel. Cara baca tanpa training ada di README folder 03.
- Untuk penjelasan sambil melihat kode, buka notebook [01](../01_Sumber_Data.ipynb), [02](../02_Preprocessing_dan_Audit_Label.ipynb), dan [03](../03_Dataset_dan_Fitur.ipynb).

`ASAL_DAN_VERIFIKASI.json` mencatat asal file, file yang disalin persis, file tampilan yang dibuat, dan hasil pengecekan. Salinan mentah dan hasil cleaning dicocokkan memakai SHA-256, yaitu sidik isi file.
