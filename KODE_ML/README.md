# Mulai belajar di sini

Folder **KODE_ML** sekarang menyimpan notebook bertahap **dan source utama yang dipakai program**.
Tidak perlu keluar folder untuk mencari implementasi cleaning, fitur, training, Optuna, CV, evaluasi, atau prediksi.

## Urutan notebook

| Notebook | Yang dipelajari |
|---|---|
| [01_Sumber_Data.ipynb](01_Sumber_Data.ipynb) | Baca lima sumber, tabel mentah, join Stimulus/CID/CAS ke SMILES |
| [02_Preprocessing_dan_Audit_Label.ipynb](02_Preprocessing_dan_Audit_Label.ipynb) | Cleaning, ALIASES, token gagal, 184 kandidat menjadi 109 target |
| [03_Dataset_dan_Fitur.ipynb](03_Dataset_dan_Fitur.ipynb) | Target multi-label, Morgan 2048, delapan deskriptor, pembandingan semua fitur |
| [04_Grup_Split_dan_Fold.ipynb](04_Grup_Split_dan_Fold.ipynb) | Asal grup historis, rekonstruksi split/fold dan seleksi label |
| [05_Model_dan_Bobot.ipynb](05_Model_dan_Bobot.ipynb) | XGBoost/LightGBM, satu classifier per label, bobot training |
| [06_Optuna_CV_dan_OOF.ipynb](06_Optuna_CV_dan_OOF.ipynb) | Search space, trial, lima seed, fold, OOF dan ambang tambahan |
| [07_Evaluasi_dan_Pemilihan.ipynb](07_Evaluasi_dan_Pemilihan.ipynb) | Urutan eksperimen, pemilihan kandidat, metrik delapan kandidat |
| [08_Prediksi_Molekul.ipynb](08_Prediksi_Molekul.ipynb) | Muat model tersimpan, prediksi satu SMILES, skor dan keputusan |

Setiap notebook memiliki import/input sendiri. Boleh menjalankan satu notebook dengan kernel baru,
tanpa menjalankan notebook sebelumnya. Urutan 01–08 adalah urutan belajar.

**Run All adalah mode belajar/audit:** membaca snapshot dan hasil, cleaning di memori,
menghitung fitur/metrik, merekonstruksi split, dan inferensi model tersimpan.
Ia tidak membuat trial atau melatih ulang model. Source training lengkap tetap ditampilkan;
notebook 07 menyediakan pemanggilan runner nyata dengan gerbang default `False` untuk eksperimen baru.
Saat gerbang tidak diaktifkan, hasil yang dibaca adalah **hasil penelitian tersimpan**, bukan hasil training notebook.

## Cara membuka

1. Buka folder proyek ini di VS Code/Jupyter, lalu masuk **KODE_ML**.
2. Pilih kernel `.venv` penelitian proyek; jangan memakai Python lain dengan versi ML berbeda.
3. Mulai dari notebook 01, pilih **Restart Kernel → Run All**, lalu baca penjelasan dan outputnya.
4. Notebook 02 mengaudit seluruh token dan notebook 04 merekonstruksi fold; keduanya dapat memerlukan beberapa menit.

Output pemeriksaan sudah disimpan di notebook. Untuk membaca saja, tidak perlu memasang paket atau menjalankan ulang.
Jika kernel notebook belum tersedia, pasang alat notebook dari root proyek:

```powershell
.\.venv\Scripts\python.exe -m pip install -r KODE_ML/requirements_belajar.txt
```

Linux: gunakan `.venv/bin/python`. Versi model tetap mengikuti `requirements_research.txt`;
`requirements_belajar.txt` hanya alat notebook. Validasi tugas ini memakai alat notebook di folder sementara,
tanpa mengubah environment ML penelitian.

## Mengapa masih ada file Python?

`research/` **di dalam folder ini** adalah satu-satunya source aktif penelitian.
Notebook mengimpor fungsi dari sana, menampilkan source fungsi yang sama lengkap dengan file/baris/SHA,
lalu memanggilnya pada data nyata. Ini mencegah dua implementasi berbeda antara notebook dan program.
Source tidak disalin menjadi definisi fungsi kedua di notebook.

Sel `tampilkan_sumber(...)` benar-benar membaca fungsi yang diimpor. Outputnya berisi kode aktual,
bukan contoh ilustrasi. Penjelasan sebelum sel menerangkan input, proses, output, dan alasan blok tersebut.
Jika source berubah, jalankan ulang notebook agar tampilan source/output ikut diperbarui.

| Source dalam KODE_ML | Notebook utama | Peran |
|---|---|---|
| research/source_identity.py | 01, 02 | Validasi molekul, join identitas, normalisasi istilah |
| research/preprocessing.py | 01, 02 | Membaca snapshot, taksonomi, alias, cleaning dan gabung molekul |
| audit_labels.py | 02 | Pemeriksaan token/alias dan rekonsiliasi tambahan; bukan perubahan mapper |
| research/dataset.py | 03 | Verifikasi data dan memasangkan X aktual dengan y/indeks beku |
| research/features.py | 03 | Morgan 2048 dan delapan angka deskriptor |
| asal_kohor.py | 04 | Kode asli grup/outer split/seleksi target untuk audit sejarah kohor |
| research/splits.py | 04 | Fold per label dan pemeriksaan pemisahan kelompok |
| research/training.py | 05 | Estimator, bobot, fit, final fit dan checkpoint |
| research/tuning.py | 06 | Ruang pencarian, TPE, budget dan resume |
| research/validation.py | 06 | CV, OOF dan pemilihan ambang tambahan |
| research/experiment.py | 07 | Urutan delapan kandidat, freeze sebelum test, hasil akhir |
| research/metrics.py | 07 | Enam metrik dan perbandingan |
| research/inference.py, research/results.py | 08 | Prediksi dan lokasi hasil utama |
| research/runtime.py | 07 | Input/output, identitas source, profil CPU dan penjagaan eksekusi |
| belajar.py | Semua | Menampilkan source; tidak menentukan hasil ML |

Folder `research/` di **root proyek** hanya berisi penghubung import,
agar perintah `python -m research.experiment` tetap memakai source di **KODE_ML/research/**.
Tidak ada salinan logika kedua di sana. File API/src aplikasi lama tidak termasuk jalur penelitian ini.

## Konfigurasi yang benar-benar dipakai

| File di research/config | Isi |
|---|---|
| sources.json | Lima sumber, commit snapshot, syarat dukungan label, kebijakan split |
| dataset.json | Kontrak ukuran/urutan dan hash artefak beku |
| experiment.json | A–D, Morgan 2048+8, Optuna, seed/fold, ambang dan metrik |
| compute.json | Profil CPU penelitian |

Keempat file tersebut dipindahkan **tanpa mengubah isinya**. 109 di kontrak dataset adalah
pemeriksaan hasil seleksi yang sudah dibekukan, bukan daftar label manual untuk proses pencocokan.

## Audit label yang perlu dipahami

**31 aturan ALIASES ≠ 184 istilah IFRA ≠ 109 kelas model.**
Taksonomi IFRA dibaca otomatis; pemilihan 109 memakai minimal 30 positif dan 2 negatif pada training.
75 label lain hanya punya 1–28 positif training. Angka 30 adalah aturan protokol penelitian,
bukan batas universal yang menjamin kualitas model.

Dari 45.689 kemunculan istilah yang mencapai mapper, 37.940 cocok exact setelah normalisasi,
1.255 melalui alias, dan **6.494 tidak terpetakan (498 istilah berbeda)**.
3.898 baris diterima tetap mempunyai sebagian token yang tidak ikut sebagai label.
Ini berbeda dari 1.230 token yang berhasil dipetakan, tetapi labelnya kemudian tidak memenuhi dukungan train.

Semua daftar lengkap ada di [hasil_audit_label](hasil_audit_label/), khususnya
[temuan audit](hasil_audit_label/LABEL_AUDIT_FINDINGS.md), [unmatched_terms.csv](hasil_audit_label/unmatched_terms.csv),
dan [taxonomy_selection.csv](hasil_audit_label/taxonomy_selection.csv).
Belum ada penambahan fuzzy/alias baru: kesetaraan aturan terbukti, validitas makna semua istilah belum terbukti.

## Data dan hasil berada di mana?

Data/model tidak diduplikasi ke folder belajar. Source pembacanya berada di sini; file besarnya tetap di:

- `../data/snapshots/perfume-five-v1/8054ea98ed675005ec10e67359902f500e4911b0/`: CSV mentah.
- `../data/builds/perfume-five-grouped-v2/`: 6.686 records, 109 target, train 5.353/test 1.333, grup dan split.
- `../data/builds/perfume-m2048-d8-v7/features.npz`: fitur model aktual 2048+8.
- Folder hasil berikut, ditentukan secara eksplisit oleh `research/results.py`:

```text
../.local-tools/campus-transfer-full-20260909/
  perfume-campus-grouped-v7-m2048-d8/
    perfume-campus-grouped-v7-m2048-d8/
      runs/perfume-five-grouped-v7-m2048-d8/
```

| Artefak hasil | Isi |
|---|---|
| comparison_all.csv | CV dan test delapan kandidat |
| selection_frozen.json | Pilihan dari validasi dan checksum sebelum evaluasi test |
| */trials.csv, recipe.json | Jejak Optuna dan parameter tiap kandidat |
| robustness/*/seed_*/ | Fold dan validasi lima seed |
| robustness/*/pooled_oof_predictions.npz | Skor OOF rata-rata lima seed pada train |
| robustness/*/thresholds.json | Ambang tambahan per label |
| */test_predictions.npz | Truth, skor dan keputusan test |
| lgbm_D/models.json, label_*.joblib | 109 classifier model terpilih |
| inference_bundle.json | Urutan fitur/label dan kebijakan inferensi |

## Kesetaraan dan batasnya

- Source inti preprocessing, fitur, estimator, weighting, tuning, CV/OOF dan metrik dipertahankan.
- Perubahan aktif terbatas pada alamat source/config dan identitas source untuk run baru.
- `asal_kohor.py` membawa tiga fungsi historis yang identik secara AST. Ia wajib menerima
  Morgan1024 **tersimpan** untuk audit asal grup; tidak membuat fitur model 1024.
- Fitur model tetap **Morgan2048+8**. Mengganti grup menjadi grup berbasis 2048 akan menjadi eksperimen baru.
- Hasil/model lama tidak ditimpa. Kesetaraan sumber dan keluaran yang diperiksa tidak sama dengan
  pembuktian bahwa training ulang seluruh eksperimen akan menghasilkan angka bit-identik.
- Nol target berarti tidak tercatat. Ada 11 baris train dan 4 test dengan seluruh target nol setelah seleksi.
- Test pernah dilihat dan CV bukan nested CV. ONNX/Android model baru tetap ditunda.

Lihat [HASIL_VERIFIKASI.md](HASIL_VERIFIKASI.md) untuk pemeriksaan dan daftar perubahan,
serta [PANDUAN_SIDANG_KODE.md](PANDUAN_SIDANG_KODE.md) untuk latihan menjelaskan metode.
