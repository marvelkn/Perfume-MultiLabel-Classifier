# Bahan bacaan yang dipakai

Mulai dari bab01 materi belajar sebelum membaca paper lengkap. Tujuannya memahami hubungan metode, bukan menghafalkan seluruh artikel.

| Bacaan | Fokus baca |
|---|---|
| [Suh 2025, salinan PDF lokal](Suh_2025_paper.pdf) | Struktur -> anotasi aroma; 10 sumber; perbandingan fitur/algoritma |
| [Saini dan Ramanathan 2022](https://doi.org/10.1038/s41598-022-18086-y) | Multi-label, Binary Relevance dan Classifier Chains |
| [XGBoost, Chen dan Guestrin 2016](https://arxiv.org/abs/1603.02754) | Penjumlahan pohon, objective dan regularisasi |
| [LightGBM, Ke dkk. 2017](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) | Histogram dan pertumbuhan leaf-wise |
| [Optuna, Akiba dkk. 2019](https://arxiv.org/abs/1907.10902) | Trial dan pencarian hyperparameter |
| [RDKit Morgan generator](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html) | radius, jumlah bit dan chirality |
| [scikit-learn Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) | AP, berbeda dari luas trapezoid PR |

## Laporan yang benar

Laporan sumber diidentifikasi dari **SKRIPSI.tex** yang memanggil Bab1-Bab5 dan pustaka.bib dalam folder 2521_skripsi_Marvel_Kevin_Nathanael. Bukan menebak dari PDF paling baru. Salinan baca-saja bab dan bibliografinya ada di **laporan_snapshot/**.

- Bab1: masalah, tujuan dan batasan.
- Bab2: penelitian terdahulu dan dasar teori.
- Bab3: protokol dan rancangan eksperimen.
- Bab4: hasil dan pembahasan.
- Bab5: simpulan dan saran.

File .tex bisa dibuka sebagai teks. Salinan ini untuk belajar dan tidak dibuat menjadi proyek LaTeX baru. Bila source laporan direvisi, salinan tidak otomatis berubah. Audit 27 Oktober mencatat tambahan batas data serta status aplikasi yang perlu dibedakan dari dokumen backend lama. Tidak ada revisi laporan dalam tugas ini.

**requirements_research_asli.txt** adalah salinan daftar dependency penelitian, bukan requirements baru untuk dipasang secara terpisah. Gunakan petunjuk environment di bab09 dan README KODE_ML.
