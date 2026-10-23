# Status implementasi — 8 September 2026

## V7 revisi 2: Morgan 2048 disiapkan - 25 September 2026

Pengguna menetapkan **Morgan 2048 bit**; paket tambahan dipilih **8 deskriptor RDKit**.
XGBoost dan LightGBM masing-masing menjalankan A-D baru: A/C Morgan saja (2048 fitur),
B/D Morgan + 8 (2056 fitur). A/B baseline 100; C/D empat studi Optuna 8 jam/studi
(**32 jam tuning**, belum termasuk validasi dan final fit). Eksekusi di **PC kampus/PC lain**.

Delapan jenis deskriptor mengikuti acuan Suh; pemilihan ini bukan bukti paket terbaik.
Gabungan fitur dan ukuran 2048 adalah eksperimen proyek, bukan resep gabungan asli Suh.
Data 6686 molekul, 109 label, split 5353/1333 dan kelompok lama tetap.
Fitur baru terpisah di data/builds/perfume-m2048-d8-v7; hasil v6 dipertahankan.

**Training penelitian v7 belum dijalankan.** Rilis aktif tetap LightGBM D v6.
Seleksi dari mean macro AP repeated grouped CV; ambang tambahan dari prediksi OOF.
B-A menguji tambahan 8 pada baseline tetap; D-C membandingkan hasil tuning masing-masing.
V6-v7 merupakan sejarah eksperimen; B/D mengubah panjang Morgan dan paket deskriptor.
Test pernah dilihat, sehingga hasil tetap eksploratori.

Paket terbaru: **perfume-campus-grouped-v7-m2048-d8.zip**; ekstrak ke folder baru.
Run ID: perfume-five-grouped-v7-m2048-d8.
Rancangan/paket v7 lama 1024 (B/D/E/F, 5 vs 8) digantikan revisi ini dan hanya menjadi riwayat.
Panduan: [Windows](WINDOWS_GUIDE_V7.md) / [Linux](LINUX_GUIDE_V7.md).
Rencana lengkap: [DESCRIPTOR_COMPARISON_V7_PLAN.md](DESCRIPTOR_COMPARISON_V7_PLAN.md).

**Setelah hasil v7 diaudit, agenda berikutnya merombak laporan sesuai metode/hasil terbaru.**
Catatan permanen: [AFTER_V7_REPORT_UPDATE.md](AFTER_V7_REPORT_UPDATE.md).
LaTeX belum direvisi untuk hasil yang belum tersedia. ONNX/Android tetap ditunda.

## Rilis v6 yang tetap aktif — penyelarasan backend, 21 September 2026

Grouped-v6 telah selesai; **LightGBM D** dipilih berdasarkan macro AP validasi
0,302232 (XGBoost D 0,298848). AP test masing-masing 0,313921 dan 0,314531.
Rilis aktif: `releases/grouped-v6-lgbm-d/`, 109 model dan 1.029 fitur.
Inferensi Python dan konfigurasi bawaan `/fingerprint` telah diselaraskan.
Fitur 6.686 molekul identik dengan dataset beku; 3.488 skor pada 32 contoh
identik dengan prediksi tersimpan. Sebanyak 23 tes komponen lulus.

Panduan aktif: [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md).
Bukti: [verification.json](reports/backend_alignment_20260921/verification.json).
ONNX lama 25 label dan dua array ambangnya dihapus atas instruksi pengguna.
ONNX baru, integrasi Android, dan deployment **ditunda**. Training v7 disiapkan sesuai revisi di atas.
Pseudo-labeling dan inferensi massal data yang ditolak tidak termasuk scope aktif.
Data, split, model terlatih, dan angka evaluasi v6 tidak diubah; hasil tetap eksploratori.

**Bagian bertanggal lebih lama di bawah adalah riwayat, bukan instruksi aktif.**


## Update grouped-v6 — 16 September 2026

V5 selesai: XGBoost D test AP 0,31453 dan LightGBM D 0,30925. Keduanya memilih
800 pohon (batas atas lama), sehingga pengguna menyetujui perluasan pencarian.
Paket v6 menilai semua A/B/C/D; A/B tetap baseline, empat studi C/D baru mendapat
8 jam/studi. Dataset/split tidak berubah. Checkpoint budget 15 detik, resume setelah
seleksi tidak mengubah artefak terkunci, dan model lengkap tersedia untuk delapan
kandidat. Ada tabel semua kandidat, ablasi fitur/tuning, perbandingan historis,
serta manifest model terpilih untuk tahap inferensi.

Status: paket siap untuk PC lab; 56 pengujian lulus, termasuk pengujian ulang
dari ZIP dan koleksi hasil sintetis. Hasil training v6 skala penelitian belum ada.
Panduan [Windows](WINDOWS_GUIDE_V6.md) / [Linux](LINUX_GUIDE_V6.md).
V6 tetap eksploratori; repeated CV tidak menggantikan test independen.
Catatan versi sebelumnya di bawah adalah riwayat, termasuk rencana berhenti v5
yang telah diganti atas permintaan pengguna.

## Update grouped-v5 — 15 September 2026

Grouped-v4 selesai tanpa kerusakan artefak dan tetap memilih incumbent D v3 untuk
kedua algoritma. Namun, `lgbm_D` menyelesaikan nol trial baru karena interupsi dan
mekanisme reservasi budget. Runner v5 membatasi eksperimen pada kondisi D, memberi
empat jam baru yang sama kepada XGBoost dan LightGBM, serta menyimpan budget setiap
15 detik. Warning nama fitur LightGBM yang tidak memengaruhi prediksi juga disaring.
V5 menjadi audit tuning terakhir; grouped-v3 tetap hasil utama.

## Update grouped-v4 — 12 September 2026

Paket lanjutan disiapkan dengan warm-start studi C dan D untuk XGBoost serta
LightGBM dari grouped-v3. Setiap studi mendapat 2 jam tambahan, sehingga kedua
algoritma menerima tambahan 4 jam yang sama. Runner menilai empat kandidat baru,
membandingkannya dengan dua incumbent v3, dan melatih final hanya kandidat yang
terpilih dari validasi. Grouped-v3 tetap menjadi hasil utama; v4 adalah penguatan
analisis Optuna dengan budget simetris.

## Update grouped-v3 — 11 September 2026

Hasil grouped-v2 telah dianalisis dan tidak dihapus. Runner grouped-v3 memperbaiki
perbandingan XGBoost–LightGBM dengan bobot kelas yang sama, menghapus batas 15 trial
yang sebelumnya menghentikan LightGBM terlalu cepat, menyimpan metrik per fold dan
prediksi out-of-fold, serta menambah lima pengulangan grouped CV untuk finalis.
Threshold hasil validasi menjadi analisis tambahan; enam metrik pada threshold 0,5
tetap menjadi laporan utama.

Status saat ini: kode dan protokol v3 disiapkan untuk validasi serta pembuatan ZIP
`perfume-campus-grouped-v3.zip`. Training v3 belum dijalankan. Catatan lama di bawah
dipertahankan sebagai riwayat.

## Revisi split kelompok — 10 September 2026

Versi aktif: **perfume-five-grouped-v2**. Seluruh 6.686 molekul dipertahankan.
Data latih 5.353, test 1.333, target 109 label, 5.363 kelompok; semua label memiliki 5 fold.
Tidak ada kelompok/fingerprint identik yang melintasi latih-test atau 545 fold validasi.
Daftar target mengikuti training baru: narcissus masuk, vetiver tidak memenuhi ambang training.
Gunakan [WINDOWS_GUIDE.md](WINDOWS_GUIDE.md) dan RUN_PERFUME_FIVE.cmd.
Paket lengkap: .local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v2.zip.
Profil PC kampus aktif memakai 6 thread CPU per model pada i7-8700K/32 GB; XGBoost dan LightGBM berjalan berurutan.
Dataset lama perfume-five-v1 dipertahankan sebagai riwayat dan ditolak runner aktif.
Bukti: reports/grouped_v2_20260910/validation.json. Training penelitian belum dijalankan.
Catatan versi sebelumnya di bawah adalah riwayat.



## Revisi aktif — 10 September 2026

Eksperimen aktif hanya memakai **GoodScents, IFRA 2019, Leffingwell, Arctander 1960, Sigma-Aldrich 2014**.
Gunakan [WINDOWS_GUIDE.md](WINDOWS_GUIDE.md), RUN_PERFUME_FIVE.cmd, serta
[notebook preprocessing](notebooks/02_perfume_five_preprocessing.ipynb).
Hasil: 6.686 molekul, 109 label, fitur 1.024 atau 1.029; metadata dan split tersimpan pada perfume-five-v1.
Preprocessing/notebook dan 21 tes komponen lulus. Training penelitian terbaru belum dijalankan.
Revisi ini menggantikan eksperimen 25 label dan rencana penggunaan seluruh dataset Suh.
Model/API historis dan arsip tetap disimpan; bagian prosedur lama di bawah bukan panduan run terbaru.


Pekerjaan sesi ini berlangsung langsung di checkout ML asli pada `C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier`. Frontend hanya referensi. Source LaTeX tetap pada folder laporan asli; isi bab belum diubah oleh Step 0–4.

| Tahap terbaru | Status dan bukti |
| --- | --- |
| Step 0 | GO: environment, rekonsiliasi 29 artefak, serta backup 53 berkas laporan terverifikasi. |
| Step 1 | GO sensor/baseline: 610 detik, suhu 80,125–85,875°C. Kesiapan termal training STOP. |
| Step 2 | GO: profil aman, log resource, lock per run, dan resume baseline tersedia; 90 tes ML sintetis lulus. |
| Step 3 | GO: audit dan [praregistrasi](PREREGISTRATION.md), 85 checksum, tujuh sel notebook dan sembilan pemeriksaan verifier lulus. |
| Step 4 | Attempt 03 STOP: baseline 610 detik, suhu akhir 84,875°C; belum ada fitting nyata. [Bukti terbaru](reports/step4_20260908/attempt-03/README.md). |
| Step 5 | Snapshot calon run 47 berkas terverifikasi dengan guard live; 1,802 detik, 75,375°C, peak RSS 28,02 MiB. Budget/run final masih menunggu smoke. [Bukti](reports/step5_20260908/attempt-20260908-213401/README.md). |
| Training protokol terbaru | Belum dimulai; belum ada hasil model atau test baru. |
| Berikutnya | Selesaikan kondisi termal, lalu ulangi Step 4 dan lanjut Step 5 sesuai izin yang sudah diberikan. Step 6 dan seterusnya belum diizinkan. |

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
