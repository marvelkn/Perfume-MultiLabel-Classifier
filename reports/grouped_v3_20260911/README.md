# Amendment eksperimen grouped-v3

Dataset, 109 label, fitur, dan pembagian kelompok tidak diubah dari grouped-v2.
Eksperimen v3 hanya memperbaiki metode training dan evaluasi.

| Masalah pada v2 | Perbaikan v3 |
| --- | --- |
| XGBoost tanpa class weighting, LightGBM memakai `class_weight=balanced` | Keduanya memakai `sample_weight` seimbang dari bagian latih setiap fold |
| LightGBM berhenti setelah 15 trial sebelum 4 jam | C dan D masing-masing memakai budget aktif 2 jam; batas trial hanya guard 10.000 yang tidak mengikat |
| Hanya rata-rata fold tersimpan | Metrik setiap fold dan prediksi out-of-fold ikut disimpan |
| Pemilihan akhir bergantung pada satu susunan fold | Satu finalis per algoritma dibandingkan lagi pada lima seed grouped CV |
| Threshold 0,5 menghasilkan jumlah prediksi label yang sangat berbeda | Threshold per label dipilih dari OOF validasi; hasilnya dilaporkan sebagai analisis tambahan |

Enam metrik utama tetap menggunakan threshold 0,5. Data uji baru dibuka setelah
semua kandidat, finalis, repeated CV, threshold, dan pilihan model dibekukan.

Karena split uji yang sama telah diperiksa pada grouped-v2, grouped-v3 dinyatakan
sebagai analisis lanjutan. Hasil grouped-v2 tetap menjadi evaluasi konfirmatori awal.
