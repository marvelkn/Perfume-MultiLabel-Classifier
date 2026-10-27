# V9: perbandingan panjang Morgan dan budget tuning

Ditetapkan **17 Oktober 2026** atas persetujuan pengguna. Disiapkan untuk PC kampus.
V7 tetap hasil penelitian yang dikunci; V8 tetap audit sepuluh sumber Suh.
**V9 memakai lima sumber dan data beku V7, bukan dataset V8.**

## Pertanyaan eksperimen

1. Apakah Morgan 1024 atau 2048 lebih baik ketika keduanya ditambah delapan deskriptor?
2. Apakah tambahan tuning dari 8 menjadi 16 jam membantu AP validasi?
3. Bagaimana tradeoff AP, precision, recall dan F1 pada ambang 0,5 dan ambang OOF?

| Kandidat untuk setiap algoritma | Input | Penyiapan parameter |
|---|---|---|
| m1024_base | 1024 + 8 = 1032 fitur | Baseline 100 pohon |
| m2048_base | 2048 + 8 = 2056 fitur | Baseline yang sama dalam algoritma |
| m1024_tuned | 1024 + 8 = 1032 fitur | Studi Optuna baru 16 jam |
| m2048_tuned | 2048 + 8 = 2056 fitur | Studi Optuna baru 16 jam |

XGBoost dan LightGBM menghasilkan **8 kandidat utama / 4 studi baru / 64 jam tuning**.
Budget adalah pilihan sumber daya, bukan angka optimal yang dibuktikan literatur.
Baseline, repeated CV, CV snapshot 8 jam, fitting final dan pengemasan di luar budget.
Seluruh tahap berjalan otomatis; total dapat berlangsung beberapa hari.

## Yang dikontrol

- Data: GoodScents, IFRA 2019, Leffingwell, Arctander 1960, Sigma-Aldrich 2014.
- **6686 molekul, 109 label, 5353 train / 1333 test**, kelompok/split yang sudah beku.
- Morgan radius 2, tanpa kiralitas. Delapan deskriptor: MolWt, MolLogP,
  NumHDonors, NumHAcceptors, TPSA, NumRotatableBonds, HeavyAtomCount, RingCount.
- 2048 dan deskriptor memakai artefak beku V7. OR-fold 2048 menjadi 1024 diverifikasi
  identik dengan fitur 1024 beku di seluruh baris; tidak membuang atau membagi ulang data.
- Kedua ukuran memakai kelompok lama berbasis struktur/fingerprint 1024. Ini menjaga
  anggota kelompok tidak melintasi partisi pada kedua representasi, bukan scaffold split.
- Bobot kelas tetap dihitung dari training fold. Tidak ada uji bobot kelas baru, perubahan
  kamus, pemilihan label baru, pseudo-labeling, atau penggabungan sumber tambahan pada V9.
- Baseline sama dengan baseline V7; studi baru memakai rentang pencarian, resep awal generik,
  sampler TPE seed 42 dan 10 startup trials yang sama. Tidak mengimpor trial/skor/best params lama.
- CPU 6 thread per model; model/label/studi berurutan. Environment dipin seperti V7.
- Batas waktu aktif termasuk loop tuning; downtime saat pause tidak dihitung. Heartbeat 15 detik;
  setelah hard crash ada tambahan pencatatan satu interval heartbeat. Ini perkiraan waktu komputasi,
  bukan pengukuran energi. Trial yang terputus tidak dinilai sebagai trial selesai.

## Urutan otomatis dan evaluasi

1. Verifikasi checksum data/fitur, environment, split dan tidak ada kelompok lintas partisi.
2. Baseline empat kombinasi; repeated grouped CV lima seed 7, 21, 42, 84, 100.
3. Empat studi baru 16 jam. Objective tetap macro Average Precision CV pada data latih.
4. Repeated grouped CV konfigurasi tuned, lima seed × lima fold per label seperti V7.
5. Simpan konfigurasi terbaik yang **sudah selesai** pada jam ke-8 dan ke-16.
   Nilai keduanya dengan repeated CV yang sama. Jika resep sama, gunakan hasil CV yang sama
   dan catat pemakaian ulang. Jika tidak ada trial selesai pada 8 jam, tulis tidak tersedia.
   Snapshot 8 jam tidak menjadi kandidat utama dan tidak dievaluasi pada test.
6. Bekukan seluruh konfigurasi, hasil CV, pilihan kandidat, dan ambang OOF sebelum test.
7. Fit delapan kandidat pada seluruh train, evaluasi test, simpan model/prediksi/tabel, buat ZIP.

Pemilihan kandidat memakai mean macro AP validasi; exact tie memakai urutan kandidat
di protocol_v9.json. Ambang utama **0,5**; ambang tambahan per label memaksimalkan F1
pada rata-rata OOF lima seed, grid 0,05–0,95 langkah 0,05. Test tidak menentukan ambang.
AP/AUROC tetap sama pada kedua ambang. Tuning AP tidak menjamin F1 ikut meningkat.
Metrik: accuracy biner rata-rata per label, AUROC, AP, specificity, precision, recall,
macro-F1, dan jumlah label prediksi per molekul. AP bukan AUPRC trapezoidal atau accuracy.

## CPU dan GPU

DxDiag PC yang diberikan: i7-8700K (6 core/12 thread), RAM 32 GB, GTX 1080 Ti 11 GB.
XGBoost 2.1 mendukung CUDA dengan device=cuda dan tree_method=hist. GTX 1080 Ti
tercantum sebagai compute capability 6.1. Namun LightGBM 4.6 memakai backend GPU
OpenCL atau CUDA yang berbeda; build CUDA didukung di Linux dan memerlukan build sesuai.

**Paket V9 memilih CPU untuk kedua algoritma** agar instalasi Windows sederhana dan
perbandingan budget memakai profil yang sama. Tidak ada fallback CPU/GPU diam-diam.
GPU tidak otomatis menaikkan kualitas model, dan kecepatannya belum diukur pada 5353
baris × 109 model biner ini. Eksperimen GPU memerlukan benchmark serta profil/protokol
terpisah; jangan mengganti device di tengah studi dengan budget yang sama.

Sumber: [XGBoost GPU](https://xgboost.readthedocs.io/en/release_2.1.0/gpu/index.html),
[LightGBM 4.6 installation](https://lightgbm.readthedocs.io/en/v4.6.0/Installation-Guide.html),
[NVIDIA legacy GPUs](https://developer.nvidia.com/cuda/gpus/legacy),
[Optuna TPE](https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.TPESampler.html).

## Batas kesimpulan

- Kontras baseline mengisolasi bit pada parameter sama; kontras tuned membandingkan
  representasi setelah pencarian dengan budget sama, parameter terbaik dapat berbeda.
- Snapshot 8/16 jam berasal dari satu studi berkelanjutan. CV lanjutan adalah bukti
  pengembangan yang berkaitan dengan pencarian, bukan nested/independent evaluation.
- Test lama pernah dilihat. V9 eksploratori; jangan mengulang sampai test terlihat bagus.
- Lima seed/fold berkorelasi; jangan menganggapnya 25 eksperimen independen atau mengklaim
  signifikansi hanya karena rata-rata lebih tinggi. Laporkan semua delta dan penurunan.
- Data tidak eksklusif parfum; label 0 adalah anotasi tidak tercatat, belum tentu negatif sensoris.
- Model/label langka tidak otomatis terpercaya karena tuning lebih lama. Skor belum dikalibrasi.
- V7 dan rilis aplikasi tidak diganti otomatis; tidak ada ONNX/deployment/LaTeX baru dalam paket ini.

## Berkas hasil utama

- RESULTS_SUMMARY.md dan comparison_all.csv: delapan kandidat dan metrik.
- paired_feature_cv.json / feature_deltas.csv: bit dan tuning pada fold berpasangan.
- budget_comparison.json / .csv: konfigurasi terbaik 8/16 jam dan hasil CV keduanya.
- Setiap kandidat tuned: optuna.sqlite3, trials.csv, budget.json, budget_snapshots.json.
- robustness/: OOF, hasil tiap seed/fold, dan ambang per label.
- selection_frozen.json, identity.json, completion_manifest.json: bukti penguncian/integritas.
- inference_bundle.json dan model per label: artefak riset, belum menggantikan rilis Essenza.
