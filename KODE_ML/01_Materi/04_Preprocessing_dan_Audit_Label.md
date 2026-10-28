# 04. Dari catatan sumber menjadi data bersih

## Tujuan dan urutan proses

Buka notebook **02_Preprocessing_dan_Audit_Label.ipynb**, `03_Kode/preprocessing.py` dan **02_Data/02_Data_Bersih/**. Targetnya: kamu bisa menjelaskan apa yang dibuang, apa yang dirapikan, dan apa yang tetap menjadi batasan data.

`merge_sources()` menerima tabel lima sumber. Untuk setiap baris: cari SMILES, periksa molekul, ambil istilah aroma sesuai format sumber, petakan ke istilah IFRA, lalu gabungkan anotasi untuk molekul yang sama. Ini aturan pengolahan data, jadi tidak perlu dipaksakan menjadi rumus ML.

## Pemeriksaan struktur

Potongan asli di `merge_sources()`:

```python
mol = Chem.MolFromSmiles(smiles)
if mol is None or mol.GetNumAtoms() == 0:
    reason = "invalid_smiles"
elif len(Chem.GetMolFrags(mol)) != 1:
    reason = "multifragment"
```

`Chem` berasal dari RDKit. `MolFromSmiles` mencoba membaca teks menjadi struktur. `None` berarti pembacaan gagal. `GetMolFrags` menghitung komponen yang terpisah. Data dengan beberapa komponen dikeluarkan karena penelitian dibatasi pada satu molekul terhubung; ini bukan model untuk campuran aroma.

Kalau identitasnya hilang, baris ditolak sebelum blok ini. Alasan penolakan disimpan di `rejected_records.json`, jadi bukan dibuang tanpa jejak.

| Alasan penolakan | Jumlah baris aktual |
|---|---|
| Identitas tidak tersedia | 482 |
| Beberapa fragmen | 313 |
| Tidak ada istilah label | 213 |
| Ada istilah, tidak ada yang terpetakan | 552 |
| SMILES invalid / behavior invalid | 0 / 0 |
| Total | 1.560 |

Ini hitungan baris sumber, bukan 1.560 molekul unik. Total 482 + 313 + 213 + 552 = 1.560.

## Label dirapikan, bukan diprediksi saat cleaning

`normalize()` di `source_identity.py` mengubah huruf menjadi kecil, merapikan tanda baca dan spasi. Contoh ilustrasi: `Sweet Notes` menjadi `sweet notes`. `map_labels()` lalu mencoba mencocokkan istilah dengan taksonomi IFRA.

```python
term = normalize(token)
mapped = term if term in taxonomy else ALIASES.get(term)
```

`token` adalah satu istilah dari sumber. `taxonomy` adalah daftar istilah acuan. Kalau istilahnya sudah ada, langsung dipakai. Kalau belum, kode mencari aturan di `ALIASES`. Contoh aturan aktual: `fruit` -> `fruity`, `cream` -> `creamy`, `powder` -> `powdery`. Istilah yang tetap gagal tidak ditebak dengan fuzzy matching.

Bedakan tiga angka ini: **31 aturan alias**, **184 istilah IFRA**, dan **109 target model**. 31 bukan jumlah label. 184 adalah daftar kandidat istilah hasil pembacaan IFRA. 109 baru didapat setelah seleksi dukungan pada data training.

## Baris diterima juga bisa kehilangan sebagian label

Misalnya secara ilustrasi satu baris punya `sweet` dan istilah yang tidak dikenali. Baris masih diterima karena `sweet` berhasil dipetakan, tetapi istilah lain tidak masuk target. Jadi 'baris diterima' tidak berarti seluruh informasi labelnya ikut.

Audit aktual `06_Audit_Label/summary.json` mencatat **45.689 kemunculan istilah** yang mencapai mapper: 37.940 exact, 1.255 alias, 6.494 tidak terpetakan. Istilah gagal itu terdiri dari 498 bentuk berbeda. Ada **3.898 baris diterima** yang sebagian istilahnya tidak terpetakan. Itu berbeda dari 552 baris yang ditolak karena tidak ada satu pun label terpetakan.

Ada juga 1.230 kemunculan istilah yang berhasil dipetakan tetapi labelnya tidak masuk 109 target karena dukungan training tidak cukup. Jangan mencampur kegagalan mapping dengan seleksi target. Lihat `unmatched_terms.csv` dan `taxonomy_selection.csv` untuk mengecek rincian.

Aturan alias sudah diperiksa konsistensi eksekusinya, tetapi kesetaraan makna seluruh istilah belum divalidasi pakar aroma. Ini batas penelitian yang perlu kamu sampaikan dengan jujur.

## Molekul yang sama digabung

Potongan asli berikut memakai himpunan, jadi label yang berulang tidak ditambahkan dua kali:

```python
canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
merged[canonical].update(labels)
origins[canonical].add(source)
```

`canonical` adalah penulisan SMILES standar dari RDKit. `merged` menyimpan gabungan label; `origins` menyimpan asal sumber. Hasilnya **6.686 baris**, satu baris per canonical SMILES. Vanillin di records.csv, `source_row` 5737, memiliki gabungan label: caramel, chocolate, creamy, gourmand, powdery, sweet, vanilla. Daftar ini lebih panjang daripada satu catatan GoodScents karena informasi beberapa sumber digabung.

## Target 0/1 dibentuk setelah pemilihan label

Protokol memilih label dengan sedikitnya **30 positif dan 2 negatif pada train**. Aturan ini ada di `asal_kohor.py`, fungsi `encode_and_split`; registry-nya sekarang dibekukan. Dari 184 kandidat, tersisa 109. Angka 30 adalah keputusan protokol, bukan jaminan statistik bahwa semua label pasti bagus.

Dalam kode `encode_frozen_records`, target dibentuk dengan memeriksa keanggotaan label:

```python
target = np.asarray([[int(name in row) for name in meta["labels"]]
                     for row in records.labels], dtype=np.uint8)
```

`name` adalah nama label, `row` kumpulan anotasi satu molekul. Ada -> `True` -> 1; tidak ada -> `False` -> 0. Urutan kolom mengikuti registry label, bukan urutan skor tertinggi. Bentuk seluruh target adalah **(6686, 109)**.

Nol berarti anotasi label tidak tercatat setelah pengolahan, bukan bukti sensoris bahwa aroma tersebut mustahil ada. Setelah hanya memakai 109 label, ada 11 baris train dan 4 test yang semua targetnya nol. Keduanya dipertahankan sesuai data beku.

## Memastikan notebook memang memeriksa data nyata

`review_frozen_preprocessing()` mengulang cleaning di memori, memanggil `verify()`, lalu membandingkan hasil encoding dengan artefak beku. Ia tidak menimpa dataset atau membuat split baru. Inputnya snapshot + config; outputnya records, target, label, split dan tabel audit untuk dipelajari.

Latihan: pilih satu baris `label_mappings.csv` dengan method alias dan satu baris `unmapped_descriptors.csv`. Jelaskan kenapa hasilnya berbeda. Lalu buka `target_train_109_label.csv` dan tunjukkan satu angka 1 serta satu angka 0.

Pertanyaan dosen: 'Apakah label nol berarti molekulnya tidak punya aroma itu?' Jawaban: 'Dalam dataset, nol berarti label itu tidak tercatat pada target terpilih. Karena anotasi tidak lengkap dan ada harmonisasi istilah, saya tidak menyamakan nol dengan kepastian ketiadaan aroma.'

Pertanyaan: 'Mengapa cleaning perlu diaudit?' Jawaban: 'Cleaning menentukan molekul dan label yang dipelajari model. Saya menyimpan alasan penolakan, asal baris dan hasil mapping supaya perubahan jumlah data dapat ditelusuri.'

Checklist: [ ] bisa menjumlahkan alasan penolakan; [ ] memahami exact/alias/unmapped; [ ] membedakan 31/184/109; [ ] memahami arti target nol.
