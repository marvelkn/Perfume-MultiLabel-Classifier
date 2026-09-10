# Pipeline aktif: perfume-five-grouped-v2

Preprocessing: perfume_data.py; konfigurasi: perfume_config.json.
Pengelompokan dan pemeriksaan split: grouped_splits.py.
Training XGBoost/LightGBM + Optuna: experiments.py. Metrik: metrics.py.
Notebook: ../notebooks/02_perfume_five_preprocessing.ipynb.
Ikuti ../WINDOWS_GUIDE.md dan RUN_PERFUME_FIVE.cmd.

6.686 molekul tetap; 5.353 training, 1.333 test, 109 target, 5.363 kelompok.
Morgan identik atau struktur identik tanpa stereokimia disatukan secara transitif.
GroupShuffleSplit seed 42 dan StratifiedGroupKFold maksimal 5 per label.
Semua kondisi A-D memakai indeks yang sama. Tidak ada pencarian seed memakai skor test.
Dataset/protokol/grup diverifikasi ulang sebelum fitting; data versi lama ditolak.
Narcissus menggantikan vetiver dalam target terpilih karena frekuensi training baru.
Semua molekul dan anotasi dalam records.csv tetap sama seperti v1.

Dataset: data/builds/perfume-five-grouped-v2; hasil: runs/perfume-five-grouped-v2.
COLLECT_CAMPUS_RESULTS.cmd mengemas hasil dan provenance.
data.py/protocol.json dan konfigurasi API lama hanya konteks historis.
Audit notebook 03 adalah bukti temuan pada v1 sebelum perbaikan, bukan dataset aktif.
