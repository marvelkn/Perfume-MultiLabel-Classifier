# Hasil perapian jalur penelitian — 25 Oktober 2026

**Jalur utama sudah dipisahkan ke `research/` dan notebook belajar diselaraskan.** Hasil utama tetap arsip V7; V9 tambahan. Tidak ada training/tuning, unduhan dataset, perubahan frontend, atau perubahan model hasil penelitian.

## File aktif

| Bagian | Lokasi dari root repo |
|---|---|
| Identitas molekul/istilah | research/source_identity.py |
| Preprocessing lima sumber | research/preprocessing.py |
| Verifikasi/pemuat data beku | research/dataset.py |
| Morgan2048 + 8 deskriptor | research/features.py |
| Fold dan pemeriksaan grup | research/splits.py |
| Estimator/bobot/fit/final fit | research/training.py |
| Optuna | research/tuning.py |
| CV/OOF/ambang tambahan | research/validation.py |
| Metrik | research/metrics.py |
| Urutan delapan kandidat | research/experiment.py |
| Prediksi model tersimpan | research/inference.py |
| File/lingkungan/alamat hasil | research/runtime.py, research/results.py |
| Sumber/aturan kohor | research/config/sources.json |
| Checksum/identitas data | research/config/dataset.json |
| Metode utama/fitur/tuning | research/config/experiment.json |
| CPU | research/config/compute.json |
| Environment ML | requirements_research.txt |
| Contoh belajar | KODE_ML/_lokasi.py, 01_fitur.py sampai 05_prediksi.py |
| Notebook utama | KODE_ML/BELAJAR_KODE_ML_LENGKAP.ipynb |
| Panduan/index/status | KODE_ML/README.md, PANDUAN_SIDANG_KODE.md, MULAI_BACA_KODE.md, README.md, PROJECT_CONTEXT.md, PROJECT_OVERVIEW.md, IMPLEMENTATION_STATUS.md, SAFE_EXECUTION_PLAN.md |
| Tes tanpa fitting | research/tests/test_research_readonly.py |

File source/config tidak diubah hanya namanya: dependensi `experiments_v5.fit_final_models`, `tuning_v6`, core eksperimen lama, dan helper `src` telah diekstrak. Jalur aktif tidak mengimpor alignment atau src. Fungsi lama yang tidak diperlukan tidak ikut disalin ke paket utama.

## Alur baru

```text
snapshot lima sumber -> source_identity/preprocessing -> records beku
records + target/indeks/grup/split beku + features2048+8 -> dataset.verify
features -> training + tuning + validation -> selection_frozen
selection_frozen -> final fitting -> metrics/test -> inference bundle
```

Source baru tetap menyediakan urutan training, tetapi belum dipakai melatih ulang. Default CLI hanya pemeriksaan. Resume arsip lama ditolak sebelum ada mutasi. Run masa depan harus memakai identitas source dan folder baru; tidak mengganti hasil utama otomatis.

## Fitur dan data yang dibaca

X berasal dari `data/builds/perfume-m2048-d8-v7/features.npz`. Morgan2048 radius2 tanpa chirality; delapan deskriptor tetap dalam urutan yang sama. A/C2048 kolom, B/D2056 kolom.

Target/row_index/source_row dibaca dari `data/builds/perfume-five-grouped-v2/train.npz` dan `test.npz`. Array fitur1024+5 yang masih ada di file historis tersebut **tidak diakses loader**. Dataset fisik tidak diduplikasi atau ditulis ulang.

Preflight memeriksa checksum kontrak baru dan manifest asli, urutan records/features/target, registry109 label, ukuran/tipe/nilai fitur, groups.csv, splits.json, pemisahan grup dan coverage545 fold awal. Test truth tidak dibuka preflight; file hanya diperiksa checksum. `load_partition('test')` membuka target test secara eksplisit.

Runtime tidak lagi merekonstruksi Morgan1024, mencocokkan lima fitur lama, atau melipat2048 menjadi1024. Pemeriksaan terpisah juga menghitung ulang seluruh fitur2048+8 dari SMILES dan memperoleh nilai identik.

**Asal grup tidak berubah:** struktur tanpa stereo atau fingerprint1024 historis identik. Pemeriksaan baru menjamin grup yang sudah dibekukan tetap identik dan pemisahannya sah; bukan klaim independen bahwa grup dibentuk dari2048. Pengelompokan itu bukan scaffold split atau semua pasangan dengan Tanimoto tinggi.

Cleaning ulang untuk audit dilakukan di memori dari snapshot lokal, tanpa output dataset. Records, urutan, sumber/label, dan target hasilnya cocok dengan kohor beku:13337 rekaman sumber,11777 lolos sebelum merge,1560 ditolak,6686 molekul,184 istilah kamus,109 target. Notebook sendiri membaca hasil cleaning dan tidak menjalankan audit raw penuh saat Run All.

## Hasil pemeriksaan

| Pemeriksaan | Hasil |
|---|---|
| AST36 fungsi inti | Sama setelah normalisasi nama import; tidak mengubah logika inti |
| Konfigurasi metode utama | Environment, A–D, fitur, weighting, search space, TPE, budget, CV, threshold dan metrik sama |
| Semua fitur molekul | Morgan `(6686,2048)` dan deskriptor `(6686,8)` persis sama saat dihitung ulang |
| Target dan urutan | Cocok; train5353 ×109 dan test1333 ×109 |
| Grup/split awal | Identik;545 fold diperiksa; tidak ada overlap grup antarpartisi |
| Jalur inferensi lama vs baru | 327 skor dari3 molekul ×109 label sama persis, termasuk urutan dan keputusan |
| Model tersimpan delapan kandidat | 27904 skor =8 ×32 molekul test ×109 label; maksimum selisih0 setelah castfloat32 |
| Toleransi skor tersimpan | rtol0, atol1e-7; selisih aktual0 untuk semua8 kandidat |
| Bukti hasil utama |1199 berkas pada completion_manifest masih cocok SHA-256 |
| Data/identitas terlindungi |24 checksum tetap |
| Src/API/tes kompatibilitas |42 berkas lama yang diperiksa tidak berubah |
| Tes jalur baru |7 lulus; tidak menjalankan suite lama yang memuat fitting |
| Notebook |31 sel dieksekusi dari awal sampai akhir;765 baris tabel kode dicocokkan dengan source; output disimpan |
| Visual notebook |Halaman pembuka, tabel penjelasan, kedua grafik dan tabel8 kandidat diperiksa |

Bukti mesin: [verification.json](verification.json), [final_checks.json](final_checks.json), [baseline_predictions.json](baseline_predictions.json), [protected_hashes.json](protected_hashes.json).

Perbandingan prediksi membuktikan model tersimpan berperilaku sama pada sampel yang diperiksa. Ini **bukan** bukti model hasil training ulang pasti identik; training tidak dijalankan. Perbandingan AST mendukung kesetaraan logika, bukan pengganti eksekusi training lengkap.

## Arsip dan backup

Sebanyak **92 file** source/config/notebook/guide/launcher lama dipindahkan, bukan dimusnahkan. Rincian dan hash ada di `archive/implementation-before-cleanup-20261025/ARCHIVE_MANIFEST.json`. Termasuk alignment lama, eksplorasiV8, notebook duplikat dan panduan/launcher versi6–9.

Backup sebelum perubahan memuat175 file, termasuk perubahan lokal notebook yang sudah ada sebelumnya. ZIP dibuka ulang dan setiap checksum diverifikasi. Lokasi:

C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier\.local-backups\research-cleanup-20261025-221854

Jangan mengekstrak ZIP di atas source aktif. Pulihkan ke folder baru. Dataset/model tidak digandakan di ZIP source. Paket PC kampus dan identity.json tetap menjadi bukti source training asli; snapshot lokal sebelum refactor tidak otomatis sama dengan source saat training.

## Bagian lama yang sengaja masih ada

1. Alamat dataset dan hasil masih bernomor versi karena merupakan identitas bukti. Nilai maupun file bukti tidak ditulis ulang.
2. Grup/split tetap mengikuti keputusan historis; itu bagian metode yang dikunci.
3. `research.runtime.check_host` masih membaca `reports/campus_20260909/full/amendment.json` untuk pengecualian laptop asal. Ini batas eksekusi, bukan fitur/model lama.
4. `src`, API, konfigurasi aplikasi, Dockerfile/Procfile, serta tes integrasinya tetap di lokasi semula karena masih saling dipakai. Jalur research tidak memakainya. `config.yaml` dan `feature_spec.json` di root **bukan konfigurasi penelitian utama**.

Pemindahan luas src/API ditolak oleh pemeriksaan persetujuan otomatis karena risiko memutus caller. Sebagai alternatif aman, hanya komponen eksperimen yang sudah dilepas dari jalur utama diarsipkan. ONNX/Android tetap ditunda; tidak ada klaim API lama sudah memakai model baru.

## Yang membutuhkan eksperimen baru

- Mengganti aturan pembentukan grup menjadi berbasis2048, scaffold, atau ambang kemiripan.
- Mengubah kohor, label, makna target0, data tambahan, representasi, parameter, budget, atau evaluasi.
- Menguji kualitas model setelah training ulang dari source baru.

Perapian ini tidak memberikan bukti peningkatan performa baru. Keterbatasan anotasi, kelas langka, test yang pernah dilihat, dan CV yang bukan nested tetap berlaku.

## Perintah pemeriksaan ringan

```text
python -B -m research.experiment --check
python -B -m pytest research/tests/test_research_readonly.py -q -p no:cacheprovider
python -B KODE_ML/04_training_evaluasi.py
```

Gunakan Python environment penelitian. Jangan menjalankan seluruh suite tes lama saat hanya ingin audit tanpa training; sebagian tes historis melakukan fitting kecil.
