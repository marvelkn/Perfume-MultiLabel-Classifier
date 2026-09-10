# Validasi split kelompok v2

Keputusan: pertahankan kohort; cegah overlap kelompok struktur/fitur.
Tidak melakukan clustering berdasarkan label atau trimming molekul.

- 6.686 molekul dan seluruh anotasi sumber sama persis (perbandingan bytes dengan v1).
- 5.363 kelompok; kelompok terbesar 38 molekul.
- 5.353 training, 1.333 test; 109 label, seluruhnya 5 fold.
- Perubahan target berdasarkan frekuensi training: narcissus masuk, vetiver keluar.
- Morgan identik lintas training-test: 0; Morgan + 5 lintas training-test: 0.
- Morgan identik lintas training-validation di seluruh 545 fold: 0.
- Pengujian tidak melakukan fitting penelitian atau membuka test.npz.
- Semua kondisi memakai split yang sama; preflight menghitung ulang kelompok dari struktur.
- Overlap kemiripan parsial/scaffold tidak dihilangkan oleh aturan kesetaraan ini.
- Riwayat audit sebelum perbaikan: notebooks/03_perfume_data_quality_audit.ipynb.

File validation.json adalah pemeriksaan independen; execution.json adalah hasil notebook.
Rancangan outer memakai GroupShuffleSplit 20% kelompok seed 42 tanpa label; inner memakai
StratifiedGroupKFold per label. k hanya dikurangi bila kelas tidak layak; tanpa fallback split acak.

Referensi resmi:
https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
