# 02 — SETELAH PREPROCESSING

**Buka records.csv dulu.** Ada **6.686 molekul** setelah identitas dicocokkan, struktur/label diperiksa, dan catatan molekul yang sama digabung.

| File | Gunanya |
|---|---|
| `records.csv` | Satu baris per molekul: nomor, SMILES, daftar label, dan asal sumber |
| `audit.json` | Hitungan baris diterima dan ditolak tiap sumber |
| `rejected_records.json` | Daftar catatan yang ditolak dan alasannya |
| `label_mappings.csv` | Istilah sumber dipasangkan ke nama aroma apa |
| `unmapped_descriptors.csv` | Istilah yang belum cocok dengan daftar target |
| `label_selection.csv` | Daftar istilah yang lolos/tidak lolos syarat dukungan pada train |
| `train_label_support.csv` | Jumlah positif/negatif dan fold tiap aroma terpilih |
| `provenance.json` | Catatan label/molekul ini berasal dari baris sumber mana |
| `groups.csv` | Kelompok molekul yang dipertahankan dalam pembagian |
| `splits.json` | Indeks train/test dan fold awal per aroma; file ini besar karena menyimpan semua daftar indeks |
| `pembagian_train_test.csv` | Tampilan sederhana: setiap molekul masuk train atau test |

Semua file asli di atas disalin persis. **pembagian_train_test.csv** adalah tampilan baru dari pembagian yang sama, bukan split baru.

Daftar label di `records.csv` adalah hasil gabungan sebelum dibatasi ke 109 target model. Jadi daftar teks itu tidak selalu sama dengan kolom target final. Target 109 yang benar-benar dipakai ada di folder 03.

Grup dan pembagian tidak dibuat ulang. Riwayat grupnya memakai struktur tanpa stereo atau Morgan 1024 yang identik; ini catatan asal pembagian, bukan fitur training utama. Fitur model utama tetap Morgan 2048, dengan/tanpa 8 deskriptor sesuai skenario.
