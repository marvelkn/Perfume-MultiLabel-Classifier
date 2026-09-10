# Windows Guide — lima sumber, tuning sampai model final

Protokol aktif: perfume-five-v1 (10 September 2026).
Gunakan file repository yang sudah direvisi ini. ZIP campus-transfer-full-20260909 lama belum memuat revisi ini.

## 1. Persiapan PC kampus
Salin repository terbaru, termasuk folder alignment, src, notebooks, requirements_training.txt,
reports/campus_20260909/full/amendment.json dan (opsional) data/builds/perfume-five-v1 serta
data/snapshots/perfume-five-v1. Jangan menyalin .venv dari laptop; buat environment di PC kampus.
Clone GitHub saja belum cukup karena perubahan lokal ini belum dipush.

Buka PowerShell di folder repository:
~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_training.txt
~~~
Jika Python 3.12 belum tersedia, instal Python 3.12 64-bit terlebih dahulu.

## 2. Satu perintah untuk preprocessing, tuning, fitting, evaluasi
~~~powershell
.\RUN_PERFUME_FIVE.cmd
~~~

Runner mengunduh hanya lima sumber bila build belum ada; bila sudah ada, checksum dan bentuk datanya diverifikasi.
PC kampus Windows dipakai untuk training. Laptop asal ditolak sesuai keputusan sebelumnya.
Tidak ada sensor suhu atau cooldown wajib. Model dijalankan bergantian dengan dua thread.
Jangan tutup terminal sebelum selesai.

| Kondisi | Fitur | Parameter |
| --- | --- | --- |
| A | Morgan 1.024 bit | Baseline 100 pohon |
| B | Morgan + 5 deskriptor | Baseline 100 pohon |
| C | Morgan | Optuna |
| D | Morgan + 5 deskriptor | Optuna |

Urutan: XGBoost A–D lalu LightGBM A–D. Setiap label mempunyai model biner sendiri.
Optuna masing-masing C/D: maksimal 15 attempted trials atau 2 jam aktif.
Total tiap algoritma: maksimal 30 attempted trials atau 4 jam tuning.
Baseline, CV ulang, dan final fitting dihitung terpisah: total pekerjaan bisa lebih dari 8 jam.
Dengan 109 label dan lima fold, satu trial memerlukan 545 fitting. Budget bukan jaminan 15 trial selesai
atau konfigurasi terbaik global. Jika tidak ada trial selesai, runner berhenti dengan pesan yang jelas.

## 3. File hasil yang perlu dibawa pulang
Salin utuh:
- runs/perfume-five-v1/ (Optuna SQLite, parameter, CV, model joblib per label, prediksi dan metrik test).
- data/builds/perfume-five-v1/ (dataset, urutan label, fitur, split, checksum dan audit).
- alignment/ dan requirements_training.txt yang digunakan saat run.

Keberhasilan seluruh pipeline ditandai runs/perfume-five-v1/complete.json.
Model dipilih dari validasi; test baru dinilai setelah semua delapan kandidat dikunci.
Metrik: accuracy biner, AUROC, AUPRC = Average Precision, specificity, precision, recall.
Model final masih Python/joblib. Ekspor ONNX dan integrasi Android merupakan tahap berikutnya.

## Notebook untuk melihat proses
Notebook notebooks/02_perfume_five_preprocessing.ipynb sudah berisi hasil eksekusi dan dua grafik.
Buka melalui VS Code/Jupyter. Untuk menjalankan ulang tanpa memasang Jupyter:
~~~powershell
.\.venv\Scripts\python.exe -m pip install nbformat==5.10.4 matplotlib==3.10.6
.\.venv\Scripts\python.exe reports/perfume_five_20260910/execute_notebook.py
~~~
Untuk kernel interaktif, instal jupyterlab/ipykernel pada environment itu dan pilih .venv sebagai kernel.

## Perintah terpisah, bila diperlukan
Build baru (jangan jalankan jika build yang sama sudah ada):
~~~powershell
.\.venv\Scripts\python.exe -m alignment.perfume_data
~~~
Cek input saja, tanpa training:
~~~powershell
.\.venv\Scripts\python.exe -m alignment.experiments --check
~~~
Training/resume di PC kampus:
~~~powershell
.\.venv\Scripts\python.exe -m alignment.experiments
~~~
Untuk membangun ulang, gunakan --output dengan direktori baru, lalu --dataset dan --run baru pada runner.
Jangan mengubah kode/config/dependency saat ingin resume. Jika listrik terputus, pastikan proses lama benar-benar
berhenti sebelum menghapus file .alignment.lock pada run tersebut. Jangan hapus folder run.
Reservasi budget trial yang terputus dihitung konservatif sebagai waktu terpakai.

## Arti revisi penelitian
Lima sumber: GoodScents, IFRA 2019, Leffingwell, Arctander 1960, Sigma-Aldrich 2014.
Hasil snapshot: 6.686 molekul, 5.318 training, 1.368 test, 109 label.
Kamus awal IFRA berisi 184 deskriptor. Ambang 30 positif training menentukan label, tanpa batas 25.
Sumber tersebut relevan bahan pewangi tetapi juga berisi flavor; bukan bukti semua rekaman khusus parfum.
Baseline dan modifikasi dibandingkan pada data yang sama. Angka artikel Suh bukan pembanding langsung
karena sumber dan label penelitian ini berbeda.
