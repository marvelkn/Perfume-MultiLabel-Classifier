# Kajian fondasi dan metodologi
Tanggal: 9 September 2026. Bukti lokal: notebook dan hasil audit pada folder ini.

## Formulasi penelitian

Penelitian mempelajari pemetaan struktur satu molekul ke beberapa deskriptor aroma. Saat training, struktur dan anotasi diketahui. Saat inferensi, hanya struktur dimasukkan; model menghasilkan skor per label dan label berdasarkan ambang. Skor bukan komposisi parfum, intensitas sensoris, atau jaminan probabilitas terkalibrasi.

Nama molekul berasal dari metadata/lookup, bukan target klasifikasi. Molekul baru, campuran, rasio bahan, dan ketahanan parfum memerlukan formulasi masalah serta data tersendiri.

Bab I laporan asli sudah menetapkan batas satu molekul. Sebaliknya, features_and_ui_design.md frontend baris 40 masih memuat rancangan klaim campuran dan ketahanan delapan jam. App.js pada checkout asli masih memanggil mockPredictFromSmiles. Selesainya preprocessing tidak membuktikan integrasi Android selesai. Frontend hanya dibaca sebagai referensi.

## Kontribusi yang dapat dipertanggungjawabkan

Suh et al. sudah memakai Pyrfume dan RDKit; penggunaannya saja tidak dapat diklaim sebagai kebaruan. Lima sumber proyek adalah subset keluarga sumber yang disebut paper, bukan protokol kurasi identik.

Kontribusi yang dituju lebih tepat berupa kurasi yang dapat ditelusuri, representasi yang dispesifikasikan, perbandingan XGBoost–LightGBM dengan validasi dan budget eksplisit, evaluasi manfaat tuning, serta implementasi aplikasi yang menjaga kesamaan fitur training/inferensi.

Optuna merupakan framework optimasi yang sudah dipublikasikan. Bukti manfaatnya harus berasal dari perbandingan untuned baseline vs tuned pada data, fitur, target, dan split yang sama. Klaim manfaat gabungan lima sumber atau tambahan deskriptor memerlukan ablation khusus. Keunggulannya belum dapat disimpulkan sebelum eksperimen selesai.

## Justifikasi lima sumber

Pemilihan dapat dijelaskan secara purposif berdasarkan kesesuaian unit analisis dan target: anotasi aroma terkait identitas kimia, identitas dapat ditelusuri ke struktur, istilah dapat diharmonisasikan, dan snapshot dapat direproduksi. Tidak semua koleksi Pyrfume mengukur target sama.

Ini penjelasan metodologis yang sesuai implementasi. Belum ada tabel screening seluruh arsip dalam bukti audit ini. Jangan menyatakan kelima sumber satu-satunya yang valid atau terbaik.

| Sumber | Molekul unik diterima dalam sumber | Molekul eksklusif terhadap empat sumber lain | Pasangan molekul-label eksklusif |
| --- | ---: | ---: | ---: |
| GoodScents | 4.203 | 1.681 | 9.723 |
| Leffingwell | 3.522 | 1.157 | 7.640 |
| Arctander | 2.376 | 1.117 | 3.721 |
| Flavornet | 483 | 51 | 196 |
| Sigma | 704 | 0 | 738 |

Kontribusi dihitung pada taksonomi awal, sebelum pemilihan 25 target. Pasangan eksklusif bukan kelas label baru. Jumlah molekul antarsumber tidak boleh langsung dijumlahkan karena overlap. Nol molekul eksklusif Sigma berkaitan pula dengan join CAS yang bergantung pada sumber lain.

## Draf paragraf Bab III

Data penelitian dihimpun dari lima arsip Pyrfume, yaitu GoodScents, Leffingwell, Arctander, Sigma, dan Flavornet. Pemilihan sumber dibatasi pada kesesuaian dengan tugas klasifikasi deskriptor aroma berbasis struktur molekul serta ketersediaan jalur pemetaan identitas kimia yang dapat diaudit. Data pada sumber terpilih selanjutnya diseleksi pada tingkat rekaman, sehingga keberadaan suatu rekaman dalam arsip tidak otomatis menjadikannya sampel penelitian. Rekaman yang tidak memperoleh identitas molekul, memiliki struktur tidak valid atau multifragmen, atau tidak menghasilkan label yang terpetakan tidak dimasukkan ke dataset gabungan. Seluruh sumber diambil dari revisi arsip yang sama dan diverifikasi melalui checksum.

Penggabungan dilakukan berdasarkan canonical isomeric SMILES. Anotasi dari identitas sama digabungkan untuk membentuk target multi-label dan asal anotasi tetap dicatat. Taksonomi awal mengacu pada Leffingwell dan istilah dari sumber lain dipetakan melalui normalisasi, pencocokan langsung, serta daftar sinonim eksplisit. Pemilihan lima sumber dimaksudkan untuk memperoleh cakupan struktur dan anotasi yang relevan dalam lingkup penelitian, bukan untuk menyatakan seluruh sumber lain tidak layak atau kombinasi ini telah terbukti optimal.

## Gap dan tindakan

1. **Log pengeluaran belum lengkap pada builder produksi.** Notebook merekonsiliasi 739 baris tambahan dengan alasan no_mapped_labels. Integrasi ke builder nanti perlu memeriksa dampak pada manifest/protokol.
2. **Harmonisasi semantik.** Istilah seperti powdery, jasmin, camphoraceous, dan musky tidak terpetakan. Tinjau padanan dengan rujukan/pakar; jangan sekadar memakai kemiripan tulisan.
3. **Label negatif semu.** Tidak tercatat diperlakukan nol; perlu menjadi keterbatasan laporan. Sebanyak 381 development all-zero bukan berarti tidak berbau.
4. **Cakupan kimia.** RDKit valid belum berarti volatil atau representatif bahan parfum. Range MolWt development 4,003–1.297,128 perlu pemeriksaan domain sebelum menambah aturan eksklusi.
5. **Batas 25 kelas.** Ini pembatasan tugas ke label dengan dukungan relatif besar; tidak mencakup seluruh aroma. Minimum 30 bukan bukti semua kelas seimbang.
6. **Generalisasi.** Deduplikasi identitas menghindari duplikasi molekul antarsplit, tetapi scaffold/kemiripan struktur masih bisa beririsan.
7. **Baseline.** Paper bukan baseline numerik langsung bila data, target, fitur, dan evaluasi berbeda. Baseline eksperimen utama dibangun pada protokol proyek sendiri.
8. **Aplikasi.** Hasil model molekul harus dibedakan dari katalog parfum dan rancangan pencampuran. Klaim ketahanan/campuran tidak didukung target dataset ini.

## Referensi

- Suh, Hong, dan Park (2025). [A comparative study of machine learning models on molecular fingerprints for odor decoding](https://www.nature.com/articles/s42004-025-01651-7).
- Hamel et al. (2024). [Pyrfume: A window to the world's olfactory data](https://www.nature.com/articles/s41597-024-04051-z).
- Akiba et al. (2019). [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902).
- [Snapshot Pyrfume yang digunakan](https://github.com/pyrfume/pyrfume-data/tree/8054ea98ed675005ec10e67359902f500e4911b0).

Metadata sumber membantu menjelaskan relevansi; checksum snapshot lokal membuktikan versi tabel yang diproses. Audit ini tidak mengubah laporan LaTeX atau frontend; draf paragraf disediakan untuk revisi akademis berikutnya.
