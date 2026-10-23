# Pembersihan backend ML — 23 Oktober 2026

Folder yang dibersihkan: `C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier`.

**Selesai.** Pembersihan hanya mengenai hasil yang dapat dibuat ulang.
Berdasarkan inventaris ukuran logis sebelum/sesudah, ukuran turun dari
**31.440.829.399 ke 25.974.010.709 byte** (berkurang **5.466.818.690 byte**,
sekitar **5,09 GiB**). Jumlah file turun dari **212.702 ke 46.153**
(berkurang **166.549** secara netto; satu panduan kode baru ditambahkan).
Penghematan ruang fisik disk dapat berbeda bila filesystem memakai deduplikasi
atau hard link.

## Yang dihapus

- Delapan salinan virtual environment `.venv`/`.venv-conda` di dalam
  paket hasil V2, V3, V4, V5, V6, V7 2048+8, dan V9 yang run-nya sudah selesai.
- Tiga folder pengujian/pemeriksaan paket sementara di `.local-tools`
  dan satu `reports/backend_alignment_20260921/test-tmp-01`.
- Cache otomatis `__pycache__`, `.pytest_cache`, dan sejenisnya
  di luar environment utama.

Semua **233 direktori** dalam daftar pembersihan diverifikasi sudah tidak ada.
Sebelum penghapusan, target dicocokkan terhadap **63 manifest**
dengan **5.338 entri file terlindungi**; tidak ada target yang
memuat file yang tercatat dalam manifest tersebut.
Semua target berada di bawah direktori backend yang disebut di atas.
Delapan paket yang environment-nya dibersihkan memiliki `complete.json`
dan tidak mempunyai lock training.

## Yang dipertahankan dan alasannya

- `.venv` di root proyek dan dependency V8: masih berguna untuk audit/inferensi.
- Seluruh `alignment/`, `src/`, `config.yaml`, notebook dan protokol:
  file bernama versi lama masih terlibat dalam import/fitting V7 dan V9.
- Sumber data, fitur, split, hasil CV/test, model `.joblib`,
  database dan trial Optuna, hasil audit, manifest, dan rilis Python.
- Arsip `.rar` dan backup lokal: kesetaraan arsip dengan hasil ekstraksi
  belum diverifikasi menyeluruh. Arsip tidak dihapus hanya berdasarkan nama.
- Repo frontend dan file laporan LaTeX di luar backend tidak disentuh.

## Dokumen yang diperbarui

- [MULAI_BACA_KODE.md](../../MULAI_BACA_KODE.md): peta notebook/Python
  preprocessing, Morgan+RDKit, Optuna, training, validasi, evaluasi, dan inferensi.
- [alignment/README.md](../../alignment/README.md) dan status awal
  `README.md`, `PROJECT_CONTEXT.md`, `PROJECT_OVERVIEW.md`,
  `SAFE_EXECUTION_PLAN.md`: menandai V9 sudah selesai,
  sambil mempertahankan riwayat status versi lama.

## Pemeriksaan setelah pembersihan

`verification.json` mencatat import modul utama, tautan panduan dan
keberadaan input/hasil V7/V9. `inventory_before_summary.json` dan
`inventory_after_summary.json` menyimpan ukuran dan jumlah file;
`cleanup_plan.json` dan `cleanup_execution.json` menyimpan daftar target
dan hasil pengecekan akhirnya. Proses ini tidak menjalankan training.

