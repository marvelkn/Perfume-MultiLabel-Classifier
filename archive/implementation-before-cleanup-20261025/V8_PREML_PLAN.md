# V8: data Suh 10 sumber, fitur v7, pre-ML dahulu

Tanggal: 1 Oktober 2026. Instruksi terakhir pengguna mempertahankan
**Morgan 2048 + 8 deskriptor**, menggantikan rencana awal menyamakan seluruh fitur Suh.

## Status keputusan 1 Oktober 2026

**V7 tetap dikunci sebagai hasil utama penelitian.** V8 adalah eksplorasi/audit data Suh sepuluh sumber untuk memahami perbedaan metode dan kualitas data. Paket V8 saat ini berhenti sebelum training dan tidak mengganti hasil, dataset, atau narasi performa v7.

## Tujuan dan batas klaim

Gunakan arsip data Suh dan ikuti preprocessing yang dapat diperiksa. Pertahankan
representasi v7 untuk eksperimen berikutnya: 2056 fitur dari SMILES. Karena fitur
berbeda, klaim yang tepat adalah:

> Data berasal dari arsip terkurasi Suh. Tahap pembersihan dan encoding yang
> dapat diverifikasi direproduksi. Representasi fitur dimodifikasi menjadi
> Morgan 2048 dan delapan deskriptor. Efek tuning akan dibandingkan dengan
> baseline yang dilatih ulang memakai data, fitur, split, dan evaluasi yang sama.

Tidak boleh mengatakan seluruh pre-ML identik dengan Suh, atau selisih skor
publik Suh semata-mata disebabkan Optuna.

## Data yang disertakan

Arsip asli [OSF Suh](https://osf.io/vfru6/) versi 1, ketiga SHA-256 dicocokkan
dengan metadata OSF pada 1 Oktober 2026. File asli tetap utuh:

- Dataset1_data_preparation.xlsx: checkpoint sumber dan kurasi.
- Dataset2_final_w_molblock.csv: data final dengan MolBlock.
- author_code.ipynb: bukti kode; **tidak dijalankan sebagai notebook**.
- download_manifest.json dan license.json: asal data dan CC BY 4.0.

Sepuluh sumber pada workbook: Arctander, AromaDB, FlavorDB (odor), Flavornet,
GoodScents/TGSC, IFRA, Leffingwell, Sharma A, Sharma B, Sigma.
IFRA Fragrance Ingredient Glossary April 2020 juga dipakai sebagai acuan kurasi
pada sheet 1. Ini tidak disebut dataset ke-11.

Tidak mengunduh ulang Pyrfume main sebagai pengganti arsip. Commit unduhan
historis dan seluruh tindakan kurasi pakar **tidak diketahui**.
Dataset bersifat odor molekuler, bukan seluruhnya bahan khusus parfum.

## Urutan yang dapat diverifikasi

| Tahap | Implementasi/bukti |
| --- | --- |
| Penggabungan sumber | Artikel menjelaskan CID; workbook menyimpan sumber dan checkpoint. Kode otomatis lengkap tidak tersedia. |
| Prioritas IFRA/pemetaan pakar | Gunakan hasil penulis yang diarsipkan. Tidak membuat kamus sendiri agar jumlahnya terlihat sama. |
| Pembuatan MolBlock | Kode sel 0: MolFromSmiles, AddHs, EmbedMolecule, UFFOptimizeMolecule, MolToMolBlock. Seed tidak eksplisit. V8 memakai MolBlock tersimpan untuk pembanding, tidak meregenerasi. |
| Deduplikasi | Drop duplicate string SMILES, keep first; sama seperti sel 2. Tidak canonical dedup tambahan. |
| Nilai kosong | Hapus SMILES/MolBlock kosong setelah dedup; slot aroma kosong dibiarkan kosong. Bukan label ODORLESS. |
| Encoding | Huruf besar, urutan label alfabetis, membership 0/1. Kode sel 2 tidak memanggil MultiLabelBinarizer meskipun artikel menyebutnya; hasil diverifikasi terhadap helper asli. |
| Fitur V8 | Morgan radius 2, 2048 bit, chirality false, dari SMILES; 8 RDKit, urutan v7, float32, digabung jadi 2056. |
| Fitur pembanding | ST1024 dari MolBlock, MD8 dari SMILES, FG20; dihitung terpisah hanya untuk audit. |
| Scaling/normalisasi | Tidak ada pada kode acuan; V8 juga tidak menambahkannya. |
| Feature selection | Tidak ada seleksi fitur tambahan. >=30 adalah pemilihan label untuk ringkasan, bukan seleksi bit fitur. |
| Rare strata dan split | Diagnosis saja. Tidak memperbaiki atau mengadopsi split yang bermasalah secara otomatis. |
| Training dan tuning | Tidak termasuk implementasi V8 tahap ini. |

Urutan delapan deskriptor V8: MolWt, MolLogP, NumHDonors, NumHAcceptors,
TPSA, NumRotatableBonds, HeavyAtomCount, RingCount.
Jenisnya sama dengan MD Suh; urutan kolom MD Suh berbeda dan disimpan eksplisit.

## Titik berhenti dan keputusan yang diperlukan

Lihat hasil terukur di **data/builds/suh-ten-v8-m2048-d8-preml/HASIL_AUDIT_V8.md**,
atau salinan dalam ZIP pada **v8_reference/preml/HASIL_AUDIT_V8.md**.

1. Tentukan perbaikan indeks sumber yang akan didokumentasikan sebagai koreksi,
   bukan pura-pura reproduksi persis kode asli.
2. Tentukan split yang sama untuk semua baseline/tuned. Split acuan mengandung
   fingerprint/struktur identik lintas train-test; split grouped lebih ketat
   tetapi harus disebut perbedaan dari Suh. Belum diimplementasikan di V8.
3. Putuskan data multikomponen. Suh mempertahankannya, v7 menolaknya. V8 saat ini
   hanya menghitung dan menyimpan untuk audit, tidak memberi label layak training.
4. Tinjau satu perbedaan SMILES/MolBlock dan dukungan label. Jangan memperbaiki,
   menghapus, atau menebak struktur tanpa keputusan yang terdokumentasi.
5. Bekukan target utama, definisi metrik, split dan protokol ML sebelum training.

**Tidak ada bypass flag yang otomatis menyalakan training.** ML baru dibuat
setelah temuan ditinjau. Hasil audit boleh lengkap walaupun ready_for_ml=false.

## Eksperimen ML berikutnya (konsep, belum dijalankan)

| Kandidat | Fitur | Peran |
| --- | --- | --- |
| XGBoost baseline | 2056 yang sama | Pembanding tanpa Optuna |
| XGBoost tuned | 2056 yang sama | Menguji efek tuning XGBoost |
| LightGBM baseline | 2056 yang sama | Pembanding tanpa Optuna |
| LightGBM tuned | 2056 yang sama | Menguji efek tuning LightGBM |

Suh sudah menggunakan XGBoost dan LightGBM, selain Random Forest. Pemilihan dua
model ini bukan algoritma baru. Kode Suh memakai n_estimators=100; XGBoost tidak
menetapkan class_weight, sedangkan LightGBM class_weight='balanced'. Parameter
lain bergantung default library historis yang tidak disertakan.

Untuk isolasi tuning, pertahankan weighting, data, fitur, split dan ambang sama
dalam pasangan baseline/tuned. Jika ingin kontrol yang sama persis resep Suh,
pisahkan dari kontrol berbobot proyek dan jelaskan perbedaannya. Tidak otomatis
menyalin semua setting v7 ke data baru.

Metrik acuan: accuracy biner per label, AUROC, average precision (ditulis AUPRC
oleh Suh), specificity, precision, recall. F1, Hamming loss dan metrik mikro
boleh tambahan. Ringkasan kode Suh memakai 100 label dengan >=30 slot kemunculan
pada checkpoint penuh. Jangan menyebutnya seleksi label train-only.
Kode ringkasan menyebut nama CSV lain yang tidak didistribusikan; sumber file
itu belum dapat diverifikasi. Daftar 100 dari Dataset2 cocok dengan jumlah
yang dilaporkan, tetapi itu bukan bukti isi kedua file identik.

Data pernah terpapar eksperimen lama dan artefak Suh sudah memiliki skor test.
Jangan menganggap pembagian baru otomatis menjadi validasi independen.
Model selection dan threshold kelak hanya memakai development/OOF; test tidak
dipakai mencari konfigurasi yang terlihat paling baik.

## Struktur paket

~~~text
RUN_PREML_V8.cmd / RUN_PREML_V8.sh
WINDOWS_GUIDE_V8.md / LINUX_GUIDE_V8.md
V8_PREML_PLAN.md
requirements_preml_v8.txt
v8/
  prepare.py
  test_preml.py
  protocol.json
  source_lock.json
  package.py
data/reference/suh2025/
  Dataset1_data_preparation.xlsx
  Dataset2_final_w_molblock.csv
  author_code.ipynb
  download_manifest.json
  license.json
notebooks/04_suh_v8_preml.ipynb
v8_reference/preml/          hasil audit penyiapan, untuk pembanding
data/builds/...              dibuat saat runner di komputer lab dijalankan
~~~

## Sumber

Suh, J., Hong, Y., & Park, C. (2025). A comparative study of machine learning
models on molecular fingerprints for odor decoding. Communications Chemistry,
8, 278. [Artikel](https://www.nature.com/articles/s42004-025-01651-7).
[Data/kode asli](https://osf.io/vfru6/).
