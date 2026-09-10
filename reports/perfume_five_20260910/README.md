# Revisi lima sumber — hasil verifikasi

Preprocessing aktif: alignment/perfume_data.py. Config: alignment/perfume_config.json.
Notebook: notebooks/02_perfume_five_preprocessing.ipynb. Runner ML: alignment/experiments.py.
Panduan: WINDOWS_GUIDE.md di root. Perintah PC kampus: RUN_PERFUME_FIVE.cmd.

- Tepat lima sumber: GoodScents, IFRA 2019, Leffingwell, Arctander 1960, Sigma-Aldrich 2014.
- 6.686 molekul; 5.318 training; 1.368 test. 109 label terpilih dari 184 istilah IFRA.
- Morgan 1.024 atau Morgan + 5 deskriptor = 1.029 fitur. Tidak ada irisan SMILES kanonik.
- Notebook: 9/9 sel PASS, dua grafik diperiksa. Durasi sekitar 20 detik, peak working set sekitar 487 MiB.
- 21 tes PASS; mencakup fit nyata empat pohon pada data sintetis untuk kedua library dan trial Optuna.
- Input build: INPUTS_VERIFIED; training skala penelitian dan evaluasi prediksi test belum dimulai.
- Laporan: sitasi, referensi, pasangan environment, brace dan gambar lulus pemeriksaan statis. PDF belum dikompilasi.

## Batas konteks
Cao & Ling (2022), DOI 10.3390/app12199716, ditambahkan sebagai konteks parfum.
Artikel itu tidak dijadikan pembenaran memilih lima arsip atau mengganti input model menjadi sensor.
Suh tetap acuan struktur–aroma. Angka artikelnya tidak dibandingkan langsung dengan dataset berbeda.
Lima sumber relevan bahan pewangi juga mencakup flavor; penggunaan parfum setiap molekul belum diverifikasi.

## File laporan yang perlu diunggah ke Overleaf

- CHANGELOG.md
- pustaka.bib
- assets/pics/perfume_five_sources.png
- assets/pics/perfume_five_training_labels.png
- src/1.Awal/07a-Abstrak.tex
- src/1.Awal/07b-Abstract.tex
- src/2.Isi/Bab1.tex
- src/2.Isi/Bab2.tex
- src/2.Isi/Bab3.tex
- src/2.Isi/Bab4.tex
- src/2.Isi/Bab5.tex

Laporan asli: C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael

## Bukti

- execution.json: hasil eksekusi notebook.
- tests.xml: hasil pytest terbaru (tes sebelumnya gagal hanya karena folder output belum dibuat, lalu diperbaiki).
- latex_validation.json: validasi source laporan dan daftar perubahan.
- report_before.json: checksum source sebelum revisi.
- data/builds/perfume-five-v1/audit.json: alasan eksklusi sumber secara rinci.

Backup sebelum revisi: .local-backups/perfume-five-20260910-095152/before-revision.zip.
Frontend tidak diedit. Model/API lama belum diganti. Perubahan lokal belum dipush; ZIP kampus lama belum memuat revisi.
