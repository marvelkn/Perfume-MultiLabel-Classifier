# 03. Dataset awal bentuknya seperti apa?

## Tujuan dan file yang dibuka

Di akhir bab ini kamu bisa memperlihatkan data sebelum preprocessing dan menjelaskan bagaimana satu anotasi aroma ditemukan SMILES-nya. Mulai dari notebook **01_Sumber_Data.ipynb** dan folder **DATA_UNTUK_BELAJAR/01_SEBELUM_PREPROCESSING/**.

Jangan membayangkan dataset awal sudah berupa satu tabel rapi berisi 2056 angka dan 109 target. Snapshot awal punya **15 CSV dari lima sumber**. Tiap sumber punya tabel molekul, tabel penghubung stimulus, dan tabel anotasi aroma.

| Tabel | Isinya | Perannya |
|---|---|---|
| molecules.csv | Identitas molekul dan SMILES | Mengetahui struktur molekul |
| stimuli.csv | Stimulus dan penghubung CID/CAS | Menyambungkan anotasi ke molekul |
| behavior.csv | Anotasi aroma | Mengetahui label yang dicatat sumber |

Arctander memakai nama `behavior_1_sparse.csv`. Nama kolom tidak seragam, jadi cara membaca anotasinya mengikuti sumbernya. GoodScents memakai teks yang dipisahkan titik koma; IFRA punya tiga kolom descriptor; Leffingwell memakai kolom 0/1; Sigma memakai nilai True/False. Aturan ini benar-benar ada di `research/preprocessing.py`, fungsi `tokens_for`.

## Mengapa ada penghubung identitas?

Misalnya tabel aroma cuma menulis Stimulus `121-33-5`. Kode perlu tahu stimulus itu menunjuk molekul yang mana, lalu mengambil SMILES-nya. CID adalah ID molekul PubChem; CAS adalah nomor registrasi zat; Stimulus adalah ID sampel dalam sumber. Ketiganya bukan fitur yang dipelajari model.

Contoh nyata GoodScents untuk vanillin:

| Langkah | Nilai dalam data |
|---|---|
| behavior | Stimulus 121-33-5; sweet;vanilla;creamy;chocolate |
| stimuli | Stimulus 121-33-5 terhubung ke CID 1183 |
| molecules | CID 1183, vanillin, COC1=C(C=CC(=C1)C=O)O |
| Sesudah digabung | Anotasi dan SMILES berada pada baris yang sama |

Untuk mencarinya, buka CSV GoodScents lalu cari CID `1183` di molecules dan `121-33-5` di stimuli/behavior. Bisa dibuka sebagai teks atau CSV; ini belum tabel fitur training.

## Kode asli: menyambungkan tabel

Buka `research/source_identity.py`, fungsi `_attach_smiles`. Potongan berikut benar-benar bagian implementasinya:

```python
frame = beh.merge(bridge, on="Stimulus", how="left", validate="many_to_one")
return frame.merge(lookup, on="__cid__", how="left", validate="many_to_one")
```

`beh` adalah tabel anotasi. `bridge` menghubungkan Stimulus ke CID. `lookup` menghubungkan CID ke SMILES. `merge` berarti menyambungkan tabel memakai kunci yang sama. `left` mempertahankan baris anotasi; identitas yang gagal ditemukan tetap terlihat supaya bisa diaudit. `many_to_one` memastikan banyak baris anotasi tidak tersambung ke beberapa identitas yang bertentangan.

Sigma punya jalur khusus `_attach_sigma` dan `_cas_bridge`: alias CAS harus menunjuk satu identitas yang bisa dipastikan dari sumber lain. Kode tidak asal memilih salah satu saat identitasnya konflik. Kasus yang tidak terselesaikan dicatat dan dikeluarkan dari data yang diterima.

## Snapshot dan config itu untuk apa?

`research/config/sources.json` mencatat lima sumber, nama CSV, commit snapshot, aturan dukungan label dan split. Commit adalah penanda versi data. `load_sources()` membaca snapshot lokal dan membandingkan SHA256 file dengan manifest. SHA256 bisa dibayangkan sebagai sidik jari isi file: perubahan isi akan membuat nilainya berubah.

Ini membantu memastikan data yang diperiksa sama dengan arsip penelitian. Tidak ada unduhan otomatis ketika notebook belajar membaca snapshot. Konfigurasi fitur model saat ini ada di `research/config/experiment.json`; jangan menyamakan seluruh config dengan pengaturan fingerprint.

## Alur jumlah data yang benar

| Tahap | Jumlah | Satuan |
|---|---|---|
| Baris anotasi awal | 13.337 | Baris sumber sebelum cleaning |
| Baris diterima | 11.777 | Baris yang lolos aturan cleaning |
| Baris ditolak | 1.560 | Baris dengan alasan penolakan tercatat |
| Sesudah molekul yang sama digabung | 6.686 | Molekul unik dalam kohor |

13.337 bukan jumlah molekul unik. Satu molekul bisa muncul di beberapa sumber atau beberapa catatan. Penggabungan dilakukan berdasarkan SMILES canonical yang masih mempertahankan informasi stereo.

Ini bukti yang dipakai: `02_SETELAH_PREPROCESSING/audit.json`, `records.csv` dan `provenance.json`. Provenance menjawab 'baris bersih ini berasal dari sumber/baris yang mana?'. Mengetahui asal data membantu kamu menjawab dosen tanpa menyebut hasil gabungan sebagai data mentah.

## Latihan dan pertanyaan sidang

Latihan: buka tiga CSV GoodScents. Cari vanillin dan tulis urutan Stimulus -> CID -> SMILES. Setelah itu buka records.csv dan cari canonical SMILES vanillin. Jangan mengharapkan teks SMILES mentah selalu identik karena penulisannya sudah dikanonisasi.

Pertanyaan: 'Apakah datanya hasil mencium parfum sendiri?' Jawaban: 'Tidak. Penelitian memakai anotasi dari lima sumber Pyrfume. Yang diprediksi adalah beberapa label untuk satu molekul berdasarkan struktur, bukan aroma campuran parfum atau hasil uji sensoris sendiri.'

Pertanyaan: 'Mengapa tidak langsung gabung dari nama senyawa?' Jawaban: 'Nama bisa memiliki sinonim atau penulisan berbeda. Implementasi menggunakan penghubung Stimulus, CID dan CAS yang diperiksa konsistensinya untuk mendapatkan SMILES.'

Pertanyaan: 'Lima sumber berarti semuanya bahan parfum?' Jawaban: 'Tidak otomatis. Lima sumber menyediakan anotasi molekul; cakupan penggunaannya perlu dibatasi dan tidak membuktikan semua molekul hanya dipakai dalam parfum.'

Checklist: [ ] bisa menunjukkan tiga jenis CSV; [ ] bisa menelusuri vanillin; [ ] membedakan baris anotasi dan molekul unik; [ ] tahu kenapa identitas bisa ditolak.
