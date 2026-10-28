# 01 — SEBELUM PREPROCESSING

Ini salinan **15 CSV sumber**, tiga file untuk masing-masing GoodScents, IFRA, Leffingwell, Arctander, dan Sigma. Isinya disalin persis, tanpa dibersihkan atau diganti.

| File | Isi umumnya |
|---|---|
| `molecules.csv` | Identitas molekul, SMILES dan informasi kimianya |
| `stimuli.csv` | Penghubung identitas catatan dengan molekul |
| `behavior.csv` | Catatan label aroma |
| `arctander_1960/behavior_1_sparse.csv` | Catatan label Arctander; nama file sumbernya memang berbeda |

Format label antar-sumber berbeda. GoodScents/Arctander memakai teks dengan pemisah titik koma; IFRA punya tiga kolom descriptor; Leffingwell memakai 0/1; Sigma punya kolom indikator True/False.

Angka **13.337** adalah jumlah catatan pada tabel behavior lima sumber, bukan penjumlahan semua baris pada 15 CSV. Satu molekul bisa muncul di beberapa sumber.

Sumber di sini adalah salinan arsip Pyrfume yang digunakan penelitian. Artinya “sebelum preprocessing proyek kita”, bukan klaim data mentah hasil eksperimen laboratorium yang belum pernah dikurasi pihak mana pun.

`pengaturan_sumber.json` berisi daftar file yang dipakai proyek. `manifest_sumber_asli.json` berisi asal URL dan sidik isi CSV sumber.
