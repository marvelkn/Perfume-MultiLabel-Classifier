# 2. Dari SMILES menjadi fitur dan target

**Tujuan belajar:** kamu bisa menunjukkan bentuk satu molekul sebelum dan setelah diubah menjadi angka, menjelaskan delapan deskriptor, dan membedakan angka input dengan angka jawaban.

## 2.1 SMILES itu tulisan struktur

**SMILES** adalah cara menuliskan struktur molekul sebagai teks. Contoh pendek: `CCO` menyatakan etanol. Huruf dan simbolnya menyatakan atom, ikatan, cabang, dan cincin. Dasarnya dibahas dalam [Weininger, 1988](https://doi.org/10.1021/ci00057a005).

Contoh yang kita ikuti adalah **vanillin**. Salinan data mentah GoodScents memiliki CID `1183`, nama `vanillin`, dan SMILES:

```text
COC1=C(C=CC(=C1)C=O)O
```

Setelah dibaca RDKit dan ditulis dalam bentuk kanonik, catatan gabungan menggunakan:

```text
COc1cc(C=O)ccc1O
```

Keduanya adalah penulisan struktur yang sama dalam contoh ini. **Kanonik** berarti perangkat lunak memilih penulisan yang konsisten. Jadi teks berubah tidak otomatis berarti molekulnya berubah. Di kode preprocessing, `Chem.MolToSmiles(mol, isomericSmiles=True)` mempertahankan informasi stereokimia yang tersedia; Morgan yang dipakai model kemudian dikonfigurasi tanpa kiralitas.

Lihat bukti aktualnya di [molecules.csv GoodScents, baris 102](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/02_Data/01_Data_Mentah/goodscents/molecules.csv:102>) dan [records.csv hasil preprocessing, baris 5739](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/02_Data/02_Data_Bersih/records.csv:5739>). Nomor `source_row` vanillin adalah **5737**; ini berbeda dari nomor baris teks karena ada header dan penomoran dimulai dari nol.

## 2.2 Library apa yang mengerjakan bagian ini?

Di [features.py, baris 2](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/features.py:2>):

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
```

| Nama | Kerjanya di file ini |
|---|---|
| **NumPy**, disingkat `np` | Menyimpan angka dalam array dan menggabungkan kolom fitur |
| **RDKit Chem** | Membaca SMILES menjadi objek molekul dan memeriksa komponennya |
| **Descriptors** | Menghitung delapan sifat molekul |
| **rdFingerprintGenerator** | Membuat generator Morgan dan menghasilkan fingerprint |

`Chem.MolFromSmiles(smiles)` mengubah teks menjadi objek molekul. Fungsi `molecule()` pada baris 12–16 menolak input yang gagal dibaca atau terdiri atas lebih dari satu komponen. Ini aturan implementasi, jadi tidak perlu dipaksakan menjadi rumus.

## 2.3 Apa isi 2.048 angka Morgan?

Bayangkan struktur molekul sebagai jaringan: atom adalah titik dan ikatan adalah garis. Morgan memeriksa lingkungan sekitar setiap atom, lalu memperluasnya sampai **radius 2**, yaitu sampai dua langkah ikatan dari pusat atom. Identitas pola-pola itu dipetakan ke posisi dalam vektor biner. **Biner** berarti isinya 0 atau 1.

Kode aktual, [features.py, fungsi calculate_morgan, baris 29](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/features.py:29>):

```python
generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=RADIUS, fpSize=N_BITS, includeChirality=False)
return np.asarray([generator.GetFingerprintAsNumPy(molecule(smi, i))
                   for i, smi in enumerate(smiles)], dtype=np.uint8)
```

`RADIUS=2` dan `N_BITS=2048` ditetapkan pada baris 6–7. `includeChirality=False` berarti informasi kiralitas tidak dimasukkan ke fingerprint. Untuk satu SMILES, hasilnya satu baris dengan 2.048 kolom; untuk sepuluh SMILES, bentuknya `(10, 2048)`.

Contoh **ilustrasi delapan posisi**, bukan fingerprint vanillin asli:

```text
[0, 1, 0, 0, 1, 0, 1, 0]
```

Angka 1 berarti ada pola yang terpetakan ke posisi tersebut. Angka 0 berarti posisi itu tidak aktif. **Bit ke-5 bukan “fitur aroma vanilla”.** Posisi fingerprint mewakili hasil pemetaan pola struktur. Beberapa pola bisa bertabrakan di posisi yang sama; ini disebut *hash collision*. Karena itu satu bit juga tidak selalu berarti satu substruktur unik. Dasar circular fingerprint dijelaskan oleh [Rogers dan Hahn, 2010](https://pubmed.ncbi.nlm.nih.gov/20426451/); penggunaan Morgan di RDKit dijelaskan dalam [dokumentasi resmi RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html).

## 2.4 Delapan deskriptor itu apa?

Deskriptor adalah **angka ringkasan sifat struktur**. Nilai berikut berasal dari file hasil aktual, bukan contoh karangan: [descriptors.csv, baris 5739](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/02_Data/03_Siap_Training/descriptors.csv:5739>).

| Urutan | Nama kode | Cara membayangkannya | Vanillin |
|---|---|---|---:|
| 1 | `MolWt` | Bobot molekul berdasarkan bobot atom rata-rata | 152,149 |
| 2 | `MolLogP` | Estimasi kecenderungan molekul berada pada lingkungan lemak dibanding air | 1,2133 |
| 3 | `NumHDonors` | Jumlah donor ikatan hidrogen menurut aturan RDKit | 1 |
| 4 | `NumHAcceptors` | Jumlah akseptor ikatan hidrogen menurut aturan RDKit | 3 |
| 5 | `TPSA` | Ringkasan luas permukaan polar, satuan Å² | 46,53 |
| 6 | `NumRotatableBonds` | Jumlah ikatan yang dinilai bisa berotasi | 2 |
| 7 | `HeavyAtomCount` | Jumlah atom selain hidrogen | 11 |
| 8 | `RingCount` | Jumlah cincin menurut perhitungan RDKit | 1 |

Kamu belum perlu menghitung TPSA atau logP manual. Kode memakai fungsi RDKit, bukan rumus buatan sendiri. Untuk sidang, pahami nama, arti, sumber perhitungan, dan urutannya.

Di `calculate()` baris 19–26, kode membuat tempat untuk `(jumlah molekul, 8)`, membaca tiap molekul, lalu menjalankan fungsi yang namanya ada di `DESCRIPTORS8`. `getattr(Descriptors, name)(mol)` berarti “ambil fungsi deskriptor bernama `name`, lalu hitung untuk `mol`”. Nilai tak hingga atau tidak valid menyebabkan error; tidak diam-diam diganti angka lain.

Pada kondisi B/D, fungsi `features()` baris 41–45 menempelkan delapan angka itu setelah Morgan. Maka:

```text
1 molekul: 2.048 bit Morgan + 8 angka deskriptor = 2.056 fitur
X training: 5.353 baris × 2.056 kolom
X test:     1.333 baris × 2.056 kolom
```

`X` adalah nama umum untuk matriks input. “Matriks” cukup dibayangkan sebagai tabel angka. Kondisi A/C memakai Morgan saja, sehingga 2.048 kolom. Urutan fitur saat prediksi harus sama dengan saat training.

## 2.5 Target 0/1 berbeda dari fitur 0/1

Sekarang pindah ke **jawaban yang ingin dipelajari**, yaitu target `Y`. Misalnya kita tampilkan empat dari 109 label:

| Molekul | chocolate | sweet | vanilla | citrus |
|---|---:|---:|---:|---:|
| Vanillin | 1 | 1 | 1 | 0 |

Dalam proyek ini **1 = label tercatat setelah pemetaan dan penggabungan sumber**. **0 = label tidak tercatat sebagai target positif**. Nilai 0 belum membuktikan bahwa semua orang pasti tidak akan mencium aroma itu. Data anotasi bisa tidak lengkap.

Rumus ringkasnya:

```text
y(i,j) = 1 jika label j ada pada daftar label molekul i; selain itu 0.
```

`i` adalah nomor molekul, `j` nomor label. Ini persis aturan pada [dataset.py, fungsi encode_frozen_records, baris 103](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Kode/dataset.py:103>):

```python
target = np.asarray([[int(name in row) for name in meta["labels"]]
                     for row in records.labels], dtype=np.uint8)
```

`name in row` memeriksa keanggotaan label; `int(True)` menjadi 1 dan `int(False)` menjadi 0. Daftar serta urutan 109 label mengikuti registri yang sudah dibekukan. Fungsi audit aktif ini **tidak memilih ulang label**. Pengaturan asalnya mensyaratkan sedikitnya 30 positif pada training dan tidak menetapkan batas “ambil 109 teratas”.

Fitur menjawab **“strukturnya seperti apa?”**, target menjawab **“label apa yang dicatat untuk contoh ini?”**. Bentuk target training adalah `(5353, 109)`. Banyak angka 1 dalam satu baris diperbolehkan.

## 2.6 Urutan praktik dan persiapan sidang

1. Ikuti vanillin pada CSV mentah dan `records.csv` yang ditautkan di atas.
2. Baca [03_Dataset_dan_Fitur.ipynb](<C:/Users/ACER/Documents/Marvel/Skripsi/Project Skripsi/Perfume-MultiLabel-Classifier/KODE_ML/03_Dataset_dan_Fitur.ipynb>) sambil melihat `features.py`.
3. Buka CSV deskriptor dan `target_train_109_label.csv` dalam folder siap training. Keduanya punya `source_row` agar molekulnya bisa dicocokkan. CSV target memiliki kolom identitas tambahan; **109** mengacu pada kolom target aromanya.

**Latihan:** tiga molekul memakai kondisi D. Berapa ukuran `X` dan `Y`? Jawaban: **`X=(3,2056)` dan `Y=(3,109)`**. Jika satu molekul punya tiga label tercatat, baris targetnya mengandung tiga angka 1, bukan satu angka 3.

**Kalau dosen bertanya:**

- **“Kenapa SMILES tidak langsung masuk ke XGBoost?”** “Classifier menerima fitur numerik. RDKit mengubah struktur SMILES menjadi Morgan dan deskriptor dengan pengaturan yang tetap.”
- **“Delapan tambahannya juga bit?”** “Bukan. Delapan deskriptor adalah angka sifat molekul; beberapa berupa hitungan dan beberapa pecahan.”
- **“Target nol berarti pasti tidak beraroma itu?”** “Nol berarti label tersebut tidak tercatat sebagai positif dalam dataset. Keterbatasan kelengkapan anotasi tetap ada.”

**Cek paham sebelum lanjut:** tunjuk satu baris vanillin, jelaskan kenapa ada 2.056 fitur tetapi 109 target, dan bedakan bit Morgan dengan target aroma tanpa menyebut keduanya sebagai hal yang sama.
