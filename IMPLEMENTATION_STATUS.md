# Status implementasi — 8 September 2026

Pekerjaan sesi ini berlangsung langsung di checkout ML asli pada `C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier`. Frontend hanya referensi. Source LaTeX tetap pada folder laporan asli; isi bab belum diubah oleh Step 0–4.

| Tahap terbaru | Status dan bukti |
| --- | --- |
| Step 0 | GO: environment, rekonsiliasi 29 artefak, serta backup 53 berkas laporan terverifikasi. |
| Step 1 | GO sensor/baseline: 610 detik, suhu 80,125–85,875°C. Kesiapan termal training STOP. |
| Step 2 | GO: profil aman, log resource, lock per run, dan resume baseline tersedia; 90 tes ML sintetis lulus. |
| Step 3 | GO: audit dan [praregistrasi](PREREGISTRATION.md), 85 checksum, tujuh sel notebook dan sembilan pemeriksaan verifier lulus. |
| Step 4 | STOP sebelum smoke: 610 detik, CPU 76,625–85,125°C; 0 label-fit, guard menolak mulai. [Bukti](reports/step4_20260908/README.md). |
| Training protokol terbaru | Belum dimulai; belum ada hasil model atau test baru. |
| Berikutnya | Ulangi Step 4 ketika suhu memenuhi syarat mulai <75°C. Estimasi runtime dan jumlah trial final masih menunggu; Step 5 belum diizinkan. |

Profil aktif yang dapat dipilih adalah smoke 1 thread/3 GiB/5 menit/80°C selama 5 detik dan training 2 thread/4 GiB/20 menit/85°C selama 10 detik. Keduanya memerlukan sensor valid dan kondisi mulai yang lebih dingin. Penghentian bersifat kooperatif, sehingga unit test belum membuktikan batas termal pada workload nyata.

Rincian operasional: [SAFE_EXECUTION_PLAN.md](SAFE_EXECUTION_PLAN.md), [RUNTIME_SAFETY.md](RUNTIME_SAFETY.md), [SENSOR_SETUP.md](SENSOR_SETUP.md), dan [bukti Step 2](reports/step2_20260907/README.md). Konfigurasi historis `config.yaml` tidak diubah; arsip `runs/_reference/` tidak dieksekusi. Belum commit/push.

## Catatan historis — 4 September 2026

Bagian berikut mempertahankan keadaan saat audit awal. Angka tes, lokasi worktree, status sensor, dan daftar pekerjaan di bawah merupakan catatan **4 September**, bukan status saat ini. Urutan eksekusi terbaru mengikuti SAFE_EXECUTION_PLAN.

Perubahan dikerjakan pada branch codex/project-reliability di worktree terpisah. Frontend berbasis origin/marvel (775b123); ML berbasis a0d96cc. Checkout asli dan laporan LaTeX tidak diubah. Publikasi perubahan project diminta setelah review implementasi: ML ke main, frontend ke marvel di repository yogawyas/Essenza_Frontend. Deployment aplikasi/API dan tuning pada data nyata belum dijalankan.

## Dua belas poin dan dasar perubahan

| No. | Perubahan yang tersedia | Dasar dan batas kesimpulan |
|---|---|---|
| 1 | Dependency training/API terpisah, versi langsung dikunci, lock lingkungan teruji, source hash dan metadata eksperimen. | [pip repeatable installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/); [Breck et al., 2017](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/). Reproducibility bukan bukti peningkatan akurasi. |
| 2 | Snapshot Pyrfume ber-revision dan checksum; join menggunakan Stimulus/CID/CAS; audit identity ambiguity, source contribution, label missingness. | [Suh et al., 2025](https://www.nature.com/articles/s42004-025-01651-7) mendukung integrasi data odor lintas sumber; [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) mendukung dokumentasi provenance. Resolusi ambigu di proyek ini bersifat konservatif. |
| 3 | Satu fungsi feature extraction untuk API/training, schema memuat versi RDKit dan chirality, larangan input multifragment. Model lama dan baru memiliki kontrak berbeda. | [RDKit Morgan generator](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator). Chirality=true tersedia untuk eksperimen baru; manfaat prediktifnya belum diukur. |
| 4 | Test terpisah; inner train/early-stop/score dan threshold holdout; penilaian semua label; freeze kedua kandidat sebelum test. | [Sechidis et al., 2011](https://doi.org/10.1007/978-3-642-23808-6_10); [Cawley & Talbot, 2010](https://jmlr.org/papers/v11/cawley10a.html). Skor CV tetap dapat optimistis karena dipakai memilih trial; final test wajib terpisah. |
| 5 | Baseline tanpa resampling, class weighting, random oversampling; MLSMOTE opsional dengan bit biner dan descriptor terpisah. | [Charte et al., 2015](https://fcharte.com/assets/pdfs/2015-KBS-MLSMOTE.pdf); [resampling pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html). Distance Hamming + standardized descriptor dan voting adalah adaptasi eksplisit, bukan salinan persis algoritme asli. Keunggulan harus diuji. |
| 6 | Dua Optuna studies dengan ruang learner-specific, early stopping, pruning setelah fold, SQLite/checkpoint/resume, thread/RAM/waktu/suhu terbatas. | [Akiba et al., 2019](https://arxiv.org/abs/1907.10902); [Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754); [Ke et al., 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html); [LightGBM parameter tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html). Rentang awal bukan parameter optimal hasil penelitian. |
| 7 | AP/F1 macro dan micro, precision/recall, AUC per label, Hamming loss, subset accuracy; scaffold split dan seed terpisah; TreeSHAP seluruh label dan heatmap/MDS dengan batas interpretasi. | [Saito & Rehmsmeier, 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432); [AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html); [MoleculeNet](https://arxiv.org/html/1703.00564v3); [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874). Eksperimen stabilitas/SHAP nyata belum dijalankan. |
| 8 | Manifest per label, hash model, feature schema, threshold, tensor names; ekspor lengkap dengan native/ONNX parity; benchmark seluruh model. | [ONNX Runtime mobile](https://onnxruntime.ai/docs/tutorials/mobile/). Uji desktop tidak menggantikan parity, ukuran APK, RAM, dan latency di perangkat. |
| 9 | Gradio CPU tanpa GPU/shims, REST/Gradio memakai fungsi sama, metadata konsisten, error terstruktur, Docker entry point benar, contract tests. | [HF Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker) dan pengujian kontrak proyek. Integrasi ke layanan publik masih menunggu deployment yang sesuai. |
| 10 | Aplikasi memvalidasi schema dan checksum, cache berdasarkan bundle ID, tidak menampilkan hasil parsial, membatalkan respons lama, menserialkan inferensi/release session; contoh molekul dari PubChem. | [React Effect cleanup](https://react.dev/reference/react/useEffect), [Axios cancellation](https://axios-http.com/docs/cancellation), dan failure-mode tests proyek. Ini keputusan rekayasa; tidak memerlukan klaim jurnal tentang peningkatan ML. |
| 11 | Explorer memakai taxonomy katalog sendiri dan mean selected accord strength; ranking deterministik; storage versioned/serialized, validasi dan preservasi data rusak; exporter JSON bersumber eksplisit. | Formula ranking dan kontrak diuji langsung; tidak disebut cosine similarity. Belum ada studi pengguna yang membuktikan kualitas rekomendasi. Provenance raw katalog historis tetap belum lengkap. |
| 12 | Pengujian Python/Jest/TypeScript, asset verifier/importer, Android preflight, signing release terpisah, worker build dibatasi, docs diperbarui, notebook lama ditandai arsip. | [Breck et al., 2017](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/); [React Native release signing](https://reactnative.dev/docs/signed-apk-android). Belum ada APK baru yang lolos uji perangkat. |

## Bukti pemeriksaan

- Dataset v2: 6.703 canonical single molecules, 5.383 development, 1.320 test, 25 label, 2.053 fitur. Audit dan sumber terperinci: reports/audit_20260904/dataset_audit.json serta data/builds/audited-20260904-v2/dataset_manifest.json.
- Tidak ada identitas SMILES yang beririsan antara development dan test. Namun 1.231 dari 1.320 molekul test memiliki scaffold yang terlihat di development (termasuk scaffold kosong untuk acyclic); ini mendasari perlunya eksperimen sensitivitas scaffold. Sebanyak 381 molekul development tidak mempunyai label pada 25 target terpilih dan tetap dipertahankan.
- Versi RDKit awal 2024.09.6 menghasilkan perbedaan H-bond acceptor pada methylpyrrole dibanding training historis. RDKit 2026.03.4 cocok persis pada seluruh 5.091 baris training historis yang didukung; 170 multifragment dikecualikan. Bukti: reports/audit_20260904/legacy_feature_parity_rdkit2026.json. Ini pemeriksaan fitur, bukan validasi akurasi model.
- Model ONNX lama tetap 25 file, dengan manifest/checksum. Tidak ada model baru hasil tuning yang diganti diam-diam.
- Verifikasi akhir: 33 tes Python dan 13 tes Jest lulus; TypeScript serta pemeriksaan checksum aset lulus. ESLint pada kode aplikasi/service yang diubah tidak menemukan error; peringatan inline style pada desain lama masih ada. Bukti mesin: reports/audit_20260904/pytest.xml dan validation-results.json di frontend.
- Python tests memakai data sintetis kecil untuk fitting dan ekspor kedua learner, termasuk jalur early stopping. Mereka tidak mengukur kualitas penelitian.
- Fitting dengan callback pembatas sumber daya diuji melalui penghentian sintetis dan resume; model yang selesai dipertahankan.
- Bundle JavaScript hasil build lama dikeluarkan dari source assets; Gradle membangun bundle release dari source terkini.
- Frontend diuji dengan mock native/network. Pemeriksaan aset mencocokkan 25 ONNX dan 2.029 baris katalog.
- Notebook arsip lolos validasi struktur nbformat; tidak dieksekusi penuh karena selnya berasal dari protokol lama dan dapat menimpa artefak historis.

## Yang belum selesai dan urutan berikutnya

1. Hubungkan pembaca suhu CPU nyata. Pada Acer Aspire A515-45 / Ryzen 5 5500U ini telemetry suhu belum tersedia. Konfigurasi default menolak memulai training tanpa sensor. [AMD](https://www.amd.com/en/support/downloads/drivers.html/processors/ryzen/ryzen-5000-series/amd-ryzen-5-5500u.html) mencantumkan Tjmax 105Â°C; limit proyek 90Â°C/30 detik adalah margin operasional, bukan klaim bahwa suhu di bawahnya menjamin perangkat tidak rusak. Ikuti [panduan ventilasi Acer](https://community.acer.com/vi/kb/articles/14006-how-to-prevent-and-recognize-overheating-of-a-notebook).
2. Review audit data, pemetaan label, dan budget trial; tetapkan seed/split sensitivity sebelum test. Jalankan baseline dan kedua studi bertahap. Simpan semua hasil, termasuk gagal/pruned dan waktu komputasi. Belum ada bukti salah satu kandidat memberikan peningkatan.
3. Fit kedua kandidat, bekukan pemilihan berdasarkan CV, evaluasi test sekali; rangkum variasi antar-seed. SHAP dan scaffold sensitivity berjalan setelah model tersedia dan budget disetujui.
4. Ekspor kandidat terpilih dan jalankan parity/benchmark sebenarnya, termasuk Android. API harus memakai feature_spec dari bundle tersebut.
5. Android SDK/API 36 dan adb ditemukan; NDK 27.1.12297006 belum tersedia. Belum memulai build Gradle berat. Verifikasi kompatibilitas React Native 0.86 dengan ONNX Runtime React Native 1.19.0, instalasi baru, upgrade cache, offline/network failure, dan signing. Jangan menyatakan APK siap rilis sebelum ini selesai.
6. Temukan raw snapshot dan izin/sumber katalog historis atau sediakan CSV terverifikasi untuk exporter baru. Katalog lama dipertahankan sebagai prototipe dengan provenance yang belum lengkap.
7. Setelah angka eksperimen dan hasil perangkat sah, baru sinkronkan Bab IIIâ€“V, abstrak, tabel, dan diagram laporan. Tidak ada file laporan yang diubah pada tahap ini.

Target publikasi source adalah main di marvelkn/Perfume-MultiLabel-Classifier dan marvel di yogawyas/Essenza_Frontend. Status commit dan publikasi dapat diperiksa pada riwayat Git; publikasi source tidak berarti training, build APK, atau deployment API sudah selesai.
