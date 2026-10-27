# Audit asal label dan alias — 26 Oktober 2026

Audit ini membaca snapshot lima sumber yang dipakai model utama. Tidak menambah
alias, menukar label, membuat split baru, atau melatih model.

## Asal 109 label

1. `taxonomy_from_ifra` membaca tiga kolom Descriptor dari IFRA. Nilainya
   dinormalisasi, dideduplikasi, diurutkan: **184 istilah**. Ini bukan daftar
   109 label yang ditulis manual.
2. `map_labels` menormalkan token setiap sumber. Jika istilah ada di taxonomy,
   ia diterima langsung. Kalau tidak, baru dicari di **31 aturan ALIASES**.
   Target alias juga wajib ada dalam taxonomy. **17 alias terpakai** dalam
   data yang mencapai mapper; 14 aturan lainnya tidak teramati.
3. Lima sumber digabung menurut SMILES canonical. **13.337 baris sumber**
   menghasilkan **11.777 baris diterima**, lalu **6.686 molekul unik**.
   **1.560 baris ditolak** dengan alasan yang tersimpan.
4. Sesudah outer split, kode asli menghitung dukungan label hanya pada **5.353
   baris train**. Syaratnya minimal **30 positif** dan **2 negatif** di train.
   Hasilnya **109 label**, tanpa batas top-N. **75 label** lainnya masing-masing
   hanya punya **1–28 positif di train**. Jadi 109 adalah hasil aturan atas data,
   kemudian disimpan sebagai registry beku agar urutan kolom model konsisten.

Audit menghitung ulang langkah 4 memakai indeks train beku, bukan mengacak ulang.
Daftar pilihan, urutan 109 label, dan matriks target cocok dengan artefak tersimpan.

## Pemisahan satuan hitung

| Ukuran | Jumlah |
|---|---:|
| Semua kemunculan token di seluruh baris mentah | 47.406 |
| Istilah berbeda setelah normalisasi di semua baris | 704 |
| Kemunculan token pada baris yang mencapai mapper produksi | 45.689 |
| Istilah berbeda yang mencapai mapper | 699 |
| Kemunculan token dipetakan secara exact setelah normalisasi | 37.940 |
| Kemunculan token dipetakan melalui alias | 1.255 |
| Kemunculan token tidak terpetakan | 6.494 |
| Istilah berbeda tidak terpetakan | 498 |
| Istilah berbeda yang berhasil dipetakan | 201 |
| Tujuan taxonomy berbeda setelah pemetaan | 184 |
| Label akhir yang memenuhi dukungan train | 109 |

Kemunculan token bukan jumlah molekul. Satu molekul bisa muncul di beberapa
sumber dan punya banyak token. **6.494 / 45.689 = sekitar 14,2% token yang
mencapai mapper gagal dipetakan**, bukan 14,2% molekul.

Data dari baris tanpa identitas valid/multikomponen tidak mencapai mapper
produksi. Audit tambahan membaca tokennya untuk transparansi; token tersebut
tidak dimasukkan ke training.

## Apa yang terjadi pada label tidak dikenal?

`map_labels` mencatat token ke penghitung `unmapped`, tetapi tidak memasukkannya
ke himpunan label hasil. Jika masih ada label lain yang diterima, baris tetap
dipakai. Ini terjadi pada **3.898 baris sumber yang diterima**. Jika semua
token gagal dipetakan, seluruh baris ditolak sebagai `unmapped_labels`.

Pencatatan memang sudah ada di kode asli melalui `unmapped_descriptors.csv`.
Namun, jika hanya membaca matriks target akhir, label yang tidak terpetakan
tidak kelihatan. Notebook perlu menampilkan audit ini, bukan menyimpulkan
bahwa semua label sudah cocok.

Di antara 39.195 kemunculan token yang cocok, **1.230** menuju label taxonomy
yang kemudian tidak memenuhi batas dukungan train. Ini berbeda dari **6.494**
token yang gagal pencocokan. Alasan pengecualian tidak boleh digabung.

Seleksi label tidak menghapus molekul lagi. Akibatnya terdapat **11 baris train
dan 4 baris test** dengan nol pada seluruh 109 kolom target. Artinya label
tercatatnya berada di luar 109 target yang dipilih, bukan bukti bahwa molekul
itu tidak berbau. Secara umum nilai 0 berarti label itu tidak tercatat dalam
target gabungan, bukan hasil pengukuran sensoris yang memastikan aroma absen.

## Masalah yang masih ada, dan batas audit

Kapitalisasi, spasi berlebih, dan tanda baca sudah dinormalisasi. Terdapat
**3.433 kemunculan token** pada mapper yang berubah bentuk lewat normalisasi;
contohnya `Floral` menjadi `floral`. Tetapi normalisasi ini tidak membuktikan
bahwa makna antaristilah sudah sama.

| Istilah gagal | Kemunculan | Cara membaca temuannya |
|---|---:|---|
| meaty | 430 | Tidak ada tujuan exact dalam taxonomy IFRA; jangan dipaksa jadi savoury. |
| winey | 407 | Tidak ada tujuan exact; jangan otomatis dipaksa jadi fermented atau rum. |
| caramellic | 292 | Kandidat untuk tinjauan makna terhadap caramel; bukan otomatis alias sah. |
| camphoreous | 180 | Kandidat tinjauan istilah terhadap camphoraceous. |
| musky | 106 | Kandidat tinjauan istilah terhadap musk like. |
| caramelic | 81 | Kandidat tinjauan terhadap caramel. |
| odorless | 183 | Konsep tidak berbau berbeda dari deskriptor aroma positif. |

Tidak dilakukan fuzzy matching. Kemiripan ejaan saja tidak cukup untuk
menentukan kesamaan makna. Bahkan bila nanti dipakai, fungsinya sebaiknya
menghasilkan kandidat untuk ditinjau, bukan langsung mengubah target.
Audit ini memverifikasi pelaksanaan aturan dan rekonsiliasi data; **belum
memvalidasi semantik setiap istilah dengan anotator/ahli atau gold standard**.
Mengubah alias/taxonomy akan mengubah target dan perlu eksperimen baru; hasil
terkunci tidak boleh diberi label seolah berasal dari aturan yang sudah diubah.

## Berkas audit lengkap

- `summary.json`: ringkasan hitungan dan status verifikasi.
- `per_source.csv`: attrition dan hasil mapper per sumber.
- `taxonomy_selection.csv`: seluruh 184 label, dukungan train, pilihan final,
  dan alasan 75 pengecualian.
- `aliases.csv`: seluruh 31 aturan, target, dan berapa kali digunakan.
- `token_inventory.csv`: token asli, normalisasi, hasil pemetaan, alasan
  penggunaan/pengecualian, jumlah, dan cakupan barisnya.
- `unmatched_terms.csv`: seluruh 498 istilah yang gagal, jumlah, sumber,
  dan penulisan mentahnya.
- `mappings.csv` / `unmapped_by_source.csv`: penghitung asli mapper.
- `rejected_rows.csv`: seluruh 1.560 baris yang ditolak dan alasannya.
- `raw_row_status.csv`: alasan tiap baris sumber sebelum penggabungan.

`audit_labels.py::audit_labels()` hanya menghasilkan objek di memori. Penyimpanan
tabel dilakukan eksplisit oleh runner/notebook, sehingga fungsi ini tidak
mengubah dataset asli.
