# Analisis hasil grouped-v4

Tanggal audit: 15 September 2026.

## Status integritas

Run selesai dan dapat dipakai sebagai hasil eksploratori. Tidak ada lock aktif,
136 artefak yang dibekukan cocok dengan checksum, dan ZIP hasil berisi 256 file
dengan status lengkap serta tidak korup.

## Hasil utama

Kondisi D tetap menjadi kondisi terbaik untuk kedua algoritma. Kandidat hasil
continuation tidak mengalahkan incumbent grouped-v3, sehingga model final tetap
`xgb_D_v3` dan `lgbm_D_v3`.

| Model | Repeated-CV AUPRC | SD | Test AUPRC | Test AUROC | Accuracy | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost D | 0,29543 | 0,00163 | 0,30578 | 0,83918 | 0,93290 | 0,23019 | 0,47057 |
| LightGBM D | 0,29427 | 0,00152 | 0,30369 | 0,83920 | 0,92378 | 0,21076 | 0,50183 |

XGBoost unggul tipis pada AUPRC, accuracy, specificity, dan precision. LightGBM
unggul tipis pada AUROC dan lebih jelas pada recall. Selisih repeated-CV AUPRC
XGBoost minus LightGBM adalah 0,00116 dengan CI t 95% 0,00050–0,00182; XGBoost
lebih tinggi pada seluruh lima seed. Pada test set, bootstrap per label memberi
selisih 0,00209 dengan CI 95% -0,00282–0,00689. Karena interval test melewati nol,
keunggulan test XGBoost belum kuat pada tingkat label.

## Dampak RDKit

Penambahan lima deskriptor RDKit pada kondisi D menaikkan repeated-CV AUPRC
dibanding kondisi C sebesar 0,03005 pada XGBoost dan 0,02335 pada LightGBM.
Efek fitur jauh lebih besar daripada selisih antaralgoritma.

## Audit Optuna

| Studi | Trial awal lengkap | Trial baru lengkap | Perbaikan objective terbaik | Status |
| --- | ---: | ---: | ---: | --- |
| xgb_C | 10 | 6 | +0,000182 | Membaik tipis, tetap jauh di bawah D |
| xgb_D | 10 | 5 | 0 | Incumbent tetap terbaik |
| lgbm_C | 43 | 38 | +0,000164 | Membaik tipis, tetap jauh di bawah D |
| lgbm_D | 52 | 0 | 0 | Budget hilang akibat interupsi |

Database `lgbm_D` mencatat satu trial gagal setelah seed v3, tetapi tidak ada trial
baru yang selesai. Mekanisme v4 mencadangkan seluruh sisa waktu di awal sesi; jika
proses terputus, seluruh reservasi dianggap terpakai. Karena itu `lgbm_D` tercatat
14.400 detik walaupun tambahan trial efektifnya nol. Hal ini tidak merusak metrik
atau model yang sudah dibekukan, tetapi membuat klaim bahwa kedua D mendapat
continuation efektif yang sama menjadi tidak tepat.

## Threshold dan keterbatasan

Threshold validasi menaikkan macro F1 XGBoost dari 0,30129 menjadi 0,32104 dan
LightGBM dari 0,28851 menjadi 0,32380. Hasil threshold ini tetap analisis sekunder;
enam metrik pada threshold 0,5 menjadi pelaporan utama.

Sebanyak 13 dari 109 label hanya memiliki kurang dari 10 contoh positif pada test
set. Nilai per label tersebut tidak stabil. Contohnya, AUPRC `popcorn` mencapai
0,91667 tetapi hanya berdasarkan tiga contoh positif. Klaim utama sebaiknya memakai
agregat seluruh label dan menyebut keterbatasan label langka.

## Keputusan

Grouped-v4 valid sebagai bukti bahwa kondisi D dan manfaat fitur RDKit stabil,
serta bahwa perbedaan XGBoost–LightGBM kecil. Grouped-v4 belum menjadi akhir tuning
karena `lgbm_D` tidak menerima trial tambahan yang selesai. V5 dibatasi pada kondisi
D untuk kedua algoritma, memakai tambahan empat jam per algoritma dan checkpoint
budget berkala. C tidak diteruskan karena selisih D sudah besar dan konsisten.

Grouped-v3 tetap menjadi hasil perbandingan utama. V4 dan V5 harus ditulis sebagai
analisis eksploratori karena test set telah dilihat sebelum eksperimen tersebut.
