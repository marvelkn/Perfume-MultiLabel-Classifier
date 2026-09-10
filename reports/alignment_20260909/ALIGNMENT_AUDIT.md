# Audit keselarasan — 9 September 2026

**Kesimpulan:** konsep prediksi sesuai, tetapi eksperimen belum identik dan belum mendukung klaim modifikasi mengalahkan acuan.

## 1. Arsitektur dan tujuan

Kode featurize menerima satu SMILES, menolak struktur tidak valid/multifragmen, dan menghasilkan Morgan + deskriptor. experiments.fit_binary melatih satu pengklasifikasi per label; skor berasal dari predict_proba. Tidak ada proses pencampuran atau pembentukan senyawa baru.

Perbedaan material terhadap acuan mencakup sumber data, taksonomi/25 target, representasi fitur, dan tiga fold. Perbedaan ini adalah modifikasi protokol, bukan otomatis kesalahan program. Pyrfume dan RDKit sudah digunakan Suh; penyebutan keduanya sebagai kebaruan perlu dikoreksi.

**Gap pembuktian:** baseline proyek memakai data dan fitur proyek, bukan reproduksi konfigurasi Suh. Saat ini baseline hanya menyimpan AP validasi. Kandidat yang difinalisasi/test adalah hasil tuning. Belum ada baseline final dengan metrik test yang sama untuk membuktikan peningkatan pada data uji. Mengubah dataset, fitur, dan tuning sekaligus juga tidak mengisolasi penyebab perubahan skor.

**Konsistensi fitur:** config.safe-training.yaml memakai chirality aktif, sedangkan API default membaca feature_spec.json dengan chirality nonaktif untuk model historis. Saat memasang model baru, arahkan API ke spesifikasi yang sama dengan model. Jangan mengganti satu sisi saja.

API /fingerprint menghasilkan fitur, bukan langsung prediksi label. Model classifier yang mengubah fitur menjadi label. Pembagian ini sesuai rancangan hibrida dan bukan kesalahan arah input/output.

## 2. Metrik

| Metrik acuan | Kode utama proyek | Tindakan |
| --- | --- | --- |
| Accuracy | subset_accuracy | Tambahkan accuracy biner per label dan mean |
| AUROC | roc_auc per label / macro | Sudah ada; pastikan label/perataan sebanding |
| AUPRC | average_precision | Simpan AP dan PR-AUC trapesium terpisah |
| Specificity | Belum ada | Tambahkan TN/(TN+FP) |
| Precision | Per label / macro | Sudah ada |
| Recall | Per label / macro | Sudah ada |

F1, Hamming loss, dan subset accuracy tetap berguna sebagai metrik tambahan. Accuracy biner rata-rata sama dengan 1 - Hamming loss untuk matriks label lengkap tanpa pembobotan; output eksplisit menghindari salah baca subset accuracy.

Dokumentasi sklearn menyatakan AP berbeda dari PR-AUC integrasi trapesium. Artikel Suh belum menjelaskan detail integrasi, threshold, dan perataan yang cukup untuk memastikan parity kode; akses OSF gagal saat audit. Karena itu, PR-AUC tambahan tidak diklaim sebagai implementasi identik paper.

### Skrip yang disiapkan

[metric_supplement.py](metric_supplement.py) memuat fungsi alignment_metrics. Skrip menambah accuracy, specificity, PR-AUC trapesium dan confusion counts per label, memakai threshold yang sudah tersimpan. Nilai tidak terdefinisi dilaporkan null beserta jumlah label valid.

Setelah hasil kampus tersedia, jalankan dari root backend dengan mengganti path contoh:

~~~powershell
.\.venv\Scripts\python.exe reports\alignment_20260909\metric_supplement.py --model-dir "D:\hasil-kampus\experiment\xgb"
.\.venv\Scripts\python.exe reports\alignment_20260909\metric_supplement.py --model-dir "D:\hasil-kampus\experiment\lgbm"
~~~

Skrip memerlukan model_manifest.json, test_predictions.npz, test_metrics.json, serta selection_frozen.json pada parent. Ia memeriksa urutan label, hash manifest yang dibekukan, dan kesamaan metrik lama sebelum menulis alignment_metrics.json baru. File lama tidak ditimpa. Tidak menjalankan training/inferensi ulang dan tidak memilih threshold dari test.

Ini pelengkap pelaporan terpisah; belum ditambahkan ke runner/ZIP kampus. Source dan 85 berkas protokol tetap lolos verifikasi. Dataset, objektif Optuna, dan paket kampus tidak diubah. Belum ada hasil test kampus yang dihitung dalam audit ini.

## 3. Revisi laporan yang diterapkan

Bab I sekarang menyebut adaptasi, bukan algoritma baru, membatasi perbandingan angka lintas studi, dan menjelaskan aplikasi sebagai rancangan yang perlu diverifikasi.

Bab III menjelaskan baseline proyek, batas klaim tuning, taksonomi Leffingwell, aturan 30 trial/4 jam, perbedaan kebijakan laptop/kampus, serta metrik pelengkap. Integrasi aplikasi tidak lagi digambarkan seolah sudah terverifikasi pada checkout asli.

Bab IV memperbarui cakupan audit sampai 9 September, melengkapi 739 rekaman tanpa label terpetakan, memperbaiki status persiapan dan referensi run historis, serta mencatat frontend asli masih memakai mockPredictFromSmiles. Tidak ada skor model, klaim kemenangan, atau hasil Android yang dibuat-buat.

[latex_changes.json](latex_changes.json) menyimpan kalimat sebelum/sesudah dan lokasi backup. Ada backup 54 file yang diverifikasi sebelum revisi. Tidak mengubah frontend.

## 4. Perbandingan yang masuk akal

1. Untuk menilai tuning: baseline dan hasil Optuna menggunakan data, fitur, label, serta split yang sama. Skor validasi terbaik saja bisa optimistis; klaim peningkatan data uji perlu baseline final yang dikunci sebelum test.
2. Untuk menilai tambahan fitur: bandingkan Morgan saja dan Morgan + deskriptor dengan faktor lain tetap.
3. Untuk menilai lima sumber: perlu eksperimen cakupan sumber dengan populasi test dan target sama serta identitas test dikeluarkan dari seluruh sumber training.
4. Untuk klaim langsung terhadap Suh: metode acuan perlu direproduksi atau dievaluasi ulang pada kondisi uji yang sebanding. Angka artikel saja hanya pembanding kontekstual.

Tidak semua ablation harus dilakukan. Sesuaikan klaim skripsi dengan eksperimen yang benar-benar selesai.

## Referensi

- [Suh et al. (2025), paper acuan](https://www.nature.com/articles/s42004-025-01651-7).
- [Definisi AP sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
- [OSF sumber kode Suh](https://osf.io/vfru6/) — tidak berhasil diakses pada audit ini.
