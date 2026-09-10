# Audit preprocessing — 9 September 2026

Buka [notebook utama](../../notebooks/00_preprocessing_audit.ipynb). Isinya 25 sel (12 Markdown, 13 kode), tabel tahapan data, empat grafik, contoh struktur molekul, dan output eksekusi tersimpan.

## Menjalankan ulang

Dari root repo backend asli, gunakan PowerShell:

~~~powershell
.\.venv\Scripts\python.exe reports\preprocessing_20260909\execute_notebook.py
~~~

Executor menjalankan sel Python berurutan dalam proses baru dengan interpreter proyek dan merekam tabel/PNG ke notebook. Ini bukan pengujian kernel Jupyter interaktif; nbformat memvalidasi format notebook. Tidak perlu memasang dependensi tambahan untuk perintah ini. Untuk menjalankan sel secara interaktif di editor Jupyter, pilih kernel dengan dependensi proyek yang sesuai.

Output tersimpan dapat dibaca tanpa menjalankan ulang. Eksekusi ulang memperbarui notebook dan artefak pada folder audit ini. Dataset, snapshot, source produksi, dan paket transfer kampus tidak ditulis oleh notebook.

## Hasil

- 12.907 baris behavior → 11.401 baris anotasi diterima → 6.703 molekul unik.
- 482 baris tidak memperoleh identitas, 285 struktur invalid/multifragmen, 739 baris tanpa label terpetakan.
- 113 label taksonomi awal → 25 target terpilih dari development.
- 5.383 development / 1.320 test; 2.053 fitur.
- 381 molekul development memiliki nol positif pada 25 target; ini bukan kelas odorless.
- Sigma menyumbang 738 pasangan molekul–label eksklusif pada taksonomi awal, meskipun tidak menyumbang molekul eksklusif.

## Bukti

- [Kajian fondasi dan rumusan Bab III](METHODOLOGY_REVIEW.md).
- [Ringkasan assertion](audit_summary.json).
- [Log eksekusi, durasi, dan memori proses](execution.json).
- [Penerimaan/pengeluaran baris tiap sumber](source_attrition.csv).
- [Kontribusi sumber](source_contributions.csv) dan [irisan molekul](source_overlap_counts.csv).
- [739 baris tanpa label terpetakan](no_mapped_label_records.csv).
- [Istilah tidak terpetakan](unmapped_descriptors.csv).
- [Distribusi label development](development_label_distribution.csv).
- [Rentang fitur development](development_descriptor_summary.csv).

Pemeriksaan mencakup kesamaan provenance, rejected records, descriptor tidak terpetakan, indeks split, seluruh nilai X/Y development, serta checksum artefak beku dan 15 sumber. Array test tersimpan tidak dideserialisasi; label mentah diproses untuk mereproduksi split awal. Tidak ada training atau tuning.

PASS berarti rekonstruksi konsisten. PASS tidak menyatakan kurasi sempurna, anotasi lengkap, lima sumber optimal, atau kinerja model terbukti. Audit ini tidak mengubah protokol maupun amendment kampus.
