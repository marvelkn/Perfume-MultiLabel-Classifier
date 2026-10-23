# Bukti revisi ilmiah F01–F18 — 22 September 2026

## Ruang lingkup dan hasil pemeriksaan

Dokumen ini mendampingi SCIENTIFIC_AUDIT_FIX_PLAN_20260921.md. Penyuntingan hanya pada sumber laporan dan dokumentasi audit. Tidak ada training, fitting, prediksi baru, preprocessing, perubahan data/split/model, pseudo-labeling, ekspor ONNX, atau edit frontend.

Laporan utama: `C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael`.
Paket hasil yang dibaca: `.local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v6/perfume-campus-grouped-v6`.
Seluruh jalur DATA/RUN di bawah relatif terhadap paket tersebut:
- DATA: `data/builds/perfume-five-grouped-v2`.
- RUN: `runs/perfume-five-grouped-v6`.
- QA laporan: `qa/scientific_audit_20260922/verify_evidence.py` dan `verification.json`.

QA menghitung kembali metrik dari **skor dan keputusan tersimpan**, bukan menjalankan model. Pemeriksaan checksum sesuai: completion 1.198 berkas, selection 260, dataset 14, source 24. Ini adalah jumlah pemeriksaan per manifest, bukan jumlah berkas unik gabungan. Identitas dataset dan protokol cocok dengan identity.json. Model LightGBM D dan angka v6 tidak diubah.

## Peta klaim ke bukti aktual

| Gap | Sumber yang dibaca / pemeriksaan |
|---|---|
| F01–F02 | Artikel Suh, Saini, abstrak primer Tyagi; Bab 1/2, abstrak, kesimpulan; tidak ada perbandingan skor lintas dataset sebagai bukti superioritas |
| F03–F07 | alignment/perfume_data.py, src/build_dataset.py, normalize dari src/label_harmonization.py; DATA/audit.json, records.csv, label_mappings.csv, rejected_records.json, label_selection.csv, sources.json; snapshot tabel sumber |
| F08 | alignment/grouped_splits.py; DATA/groups.csv, splits.json, audit.json; 5.363 kelompok, terbesar 38, 545 fold awal diperiksa oleh audit tersimpan |
| F09–F12 | alignment/protocol_v6.json, alignment/experiments.py dan modul v6; RUN/identity.json, recipe.json, tuning_summary.json, baseline model metadata |
| F13–F14 | RUN/comparison_all.csv, ablation_deltas.csv, comparison_history.csv; delapan test_predictions.npz, ringkasan per-label, robustness setiap kandidat |
| F15 | Kode pemilihan ambang; RUN/robustness/*/thresholds.json; prediksi dan ringkasan ambang tambahan |
| F16 | alignment/experiments_v4.py: paired_seed_comparison dan paired_label_bootstrap; kedua JSON hasil perbandingan yang dipakai kembali oleh v6 |
| F17 | INFERENCE_GUIDE.md; reports/backend_alignment_20260921/verification.json; src/inference.py, src/featurize.py, src/release.py, api_light.py dan feature_spec.json |
| F18 | Bab 1–5, abstrak Indonesia/Inggris, settings.tex, SKRIPSI.tex, lampiran, pustaka.bib aktif; QA numerik, sitasi, dan kompilasi XDV |

Bukti 23 tes dan parity Python berasal dari pemeriksaan backend yang **sudah tersimpan**; tes/inferensi tidak dijalankan ulang pada revisi ini. Kesamaan 6.686 vektor fitur dan 3.488 skor pada 32 contoh menunjukkan kesetaraan implementasi pada cakupan itu, bukan akurasi baru, kalibrasi, atau keberhasilan aplikasi.

## F04 — Rekonsiliasi

13.337 baris sumber = 1.560 ditolak + 11.777 diterima.
1.560 = 482 gagal identitas + 313 multifragmen + 213 anotasi kosong + 552 tanpa label terpetakan.
11.777 - 5.091 konsolidasi rekaman = 6.686 molekul unik.
Tidak menyamakan baris sumber dengan molekul, atau menganggap semua baris yang berkurang sebagai data tanpa label yang siap dilatih ulang.
Tabel audit Bab 4 kini memisahkan dua alasan kegagalan label; semua nilai berasal dari DATA/audit.json.

## F05 — Aturan label dan antrean tinjauan ahli

Aturan yang dipakai alur lima sumber adalah **ALIASES pada alignment/perfume_data.py** dan kosakata teramati dari tiga kolom IFRA. Daftar SYNONYMS/taksonomi lain dalam src/label_harmonization.py bukan pengendali pemetaan alur ini; yang diimpor dari modul tersebut adalah fungsi normalize.

Urutan: pecah anotasi sesuai format sumber, huruf kecil, hapus karakter nonalfanumerik, rapikan spasi, cocokkan langsung; hanya jika gagal, cari alias eksplisit yang targetnya terdapat dalam kosakata. Tidak ada fuzzy matching. Union anotasi mempertahankan provenance tetapi dapat membawa noise.

**Penting:** kolom bernama raw_descriptor dalam label_mappings.csv sudah berisi token setelah normalisasi, bukan salinan teks asli byte-per-byte. Teks asli diperiksa pada snapshot:
`data/snapshots/perfume-five-v1/8054ea98ed675005ec10e67359902f500e4911b0`.

Contoh sumber asli yang dibaca:
- GoodScents Stimulus 100-09-4: `phenolic;animal;fecal;medicinal`. Token animal diarahkan ke animal like.
- GoodScents Stimulus 10031-71-7: `raisin;plum;floral;honey;fruity;powdery;green;geranium;tropical;papaya`. Kehadiran papaya/fruity memberi konteks pada contoh ini, tetapi tidak membuktikan bahwa semua tropical selalu bermakna tropical fruit.
- GoodScents Stimulus 10032-05-0: `fatty;privet;fruity;oily;orange;green;tropical;waxy`.
- Leffingwell Stimulus 8164: kolom asli `jasmine` bernilai 1; dipetakan ke jasmin.
- Nama kolom asli Sigma mencakup jasmine, tropical, animal, butter, balsam, dan herbaceous; hanya indikator biner yang dipakai, bukan kolom metadata descriptors.

### Alias yang benar-benar tercatat

Count adalah hitungan pemetaan dalam audit, **bukan** jumlah molekul unik final atau prevalensi setelah deduplikasi. Tabel ini mempertahankan seluruh 29 kombinasi sumber–alias yang digunakan, termasuk alias yang jarang; target tidak diubah.

| Sumber | Token sesudah normalisasi | Target | Count |
|---|---|---|---:|
| arctander | animal | animal like | 73 |
| goodscents | animal | animal like | 93 |
| goodscents | balsam | balsamic | 26 |
| goodscents | butter | buttery | 1 |
| goodscents | flower | floral | 7 |
| goodscents | flowers | floral | 1 |
| goodscents | fruit | fruity | 3 |
| goodscents | herb | herbal | 1 |
| goodscents | herbaceous | herbal | 1 |
| goodscents | mint | minty | 1 |
| goodscents | musk | musk like | 90 |
| goodscents | nut | nutty | 1 |
| goodscents | oil | oily | 3 |
| goodscents | savory | savoury | 57 |
| goodscents | spice | spicy | 7 |
| goodscents | tropical | tropical fruit | 235 |
| goodscents | wood | woody | 1 |
| leffingwell | animal | animal like | 48 |
| leffingwell | jasmine | jasmin | 42 |
| leffingwell | mint | minty | 123 |
| leffingwell | musk | musk like | 20 |
| leffingwell | savory | savoury | 82 |
| leffingwell | tropical | tropical fruit | 201 |
| sigma | animal | animal like | 9 |
| sigma | balsam | balsamic | 16 |
| sigma | butter | buttery | 32 |
| sigma | herbaceous | herbal | 66 |
| sigma | jasmine | jasmin | 14 |
| sigma | tropical | tropical fruit | 1 |

### Alasan pemetaan dan status semantik

| Keluarga aturan | Alasan operasional, bukan bukti sensori | Status |
|---|---|---|
| savory → savoury; jasmine → jasmin | Penyelarasan varian penulisan ke istilah kamus | Dapat ditelusuri; belum validasi ahli |
| flower/flowers → floral; fruit → fruity; wood → woody; spice → spicy; butter → buttery; nut → nutty; oil → oily; mint → minty | Bentuk nomina menjadi deskriptor target | Review makna diperlukan; tidak otomatis identik pada setiap konteks |
| herb/herbaceous → herbal; balsam → balsamic | Pengelompokan istilah ke label yang tersedia | Review konteks sumber dan pakar diperlukan |
| tropical → tropical fruit (437 pemetaan) | Target kamus lebih spesifik daripada token sumber | Prioritas review: belum dibuktikan semua pemetaan semantik benar |
| animal → animal like (223 pemetaan); musk → musk like (110 pemetaan) | Pemetaan ke istilah perbandingan dalam kamus | Prioritas review: perbedaan arti belum dinilai pakar |

Audit ini membuktikan keterlacakan, **bukan** menyetujui kesetaraan arti. Tidak ada kamus atau pakar yang diada-adakan. Koreksi pemetaan, jika diperlukan kemudian, akan mengubah target dan memerlukan versi data serta evaluasi baru; tidak dilakukan sekarang.

## F07–F08 — Contoh struktur tersimpan

Contoh berikut hanya dibaca dari groups.csv, tidak dibuat melalui kanonisasi atau ekstraksi ulang.

| source_row | SMILES tersimpan | group_id |
|---:|---|---:|
| 1278 | `CC(=O)OCC[C@@H](C)CCC=C(C)C` | 1018 |
| 1280 | `CC(=O)OCC[C@H](C)CCC=C(C)C` | 1018 |
| 1465 | `CC(C)=C1CC[C@@H](C)CC1=O` | 1200 |
| 1466 | `CC(C)=C1CC[C@H](C)CC1=O` | 1200 |

Masing-masing pasangan mempertahankan tanda stereokimia berbeda pada SMILES tetapi berada dalam kelompok yang sama. Ini mendukung uraian batas representasi/grouping; tidak membuktikan persamaan atau perbedaan aroma pasangan tersebut. Kanonisasi bukan normalisasi tautomer/protonasi; fragmen tidak otomatis dinetralkan atau diambil yang terbesar. Hash fingerprint identik bukan bukti identitas kimia. Aturan kelompok adalah komponen transitif struktur tanpa stereokimia **atau** Morgan identik, bukan scaffold split atau seluruh kemiripan parsial.

## F10–F12 — Reproduksibilitas desain

- Baseline A/B memakai n_estimators=100, random_state=42, n_jobs=6.
- XGBoost: tree_method=hist, eval_metric=logloss. Default efektif: learning_rate 0,3; max_depth 6; min_child_weight 1; reg_alpha 0; reg_lambda 1; gamma 0; subsample dan colsample_bytree 1.
- LightGBM: boosting_type=gbdt, verbosity=-1; learning_rate 0,1; max_depth -1; num_leaves 31; min_child_samples 20; min_child_weight 0,001; reg_alpha/reg_lambda 0; min_split_gain 0; subsample_freq 0; subsample/colsample_bytree 1.
- Angka itu dicocokkan dengan fungsi pembuat model, recipe baseline, serta satu model label acidic pada masing-masing A/B setelah verifikasi SHA256. **Pemeriksaan metadata model adalah sampel empat model**, bukan pemuatan seluruh model; tidak ada fit/predict. Hasil get_params dan booster config serta hash sampel disimpan dalam verification.json.
- Versi: NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.6.1, RDKit 2026.3.4, XGBoost 2.1.4, LightGBM 4.6.0, Optuna 4.5.0, joblib 1.6.0.
- Bobot dihitung sendiri sebagai N/(2 Nc) pada target biner satu label, hanya dari partisi pelatihan. Rujukan sklearn menjelaskan rumus; bukan klaim pemakaian fungsi multioutput yang mengalikan bobot antarlabel.
- TPE seed42, n_startup_trials10, n_ei_candidates24, multivariate=false. Satu trial memakai satu konfigurasi bagi 109 klasifier label; bukan 109 studi tuning terpisah.
- Empat studi × 28.800 detik anggaran aktif. Trial selesai/gagal: XGB C 20/1, XGB D 20/1, LGBM C 117/1, LGBM D 36/1. Tidak menyamakan budget dengan wall-clock seluruh eksperimen.
- Lima fold/lima seed dan paket lima deskriptor adalah pengaturan yang diuji. Penjelasan kompromi sumber daya dan sifat kimia adalah interpretasi desain saat ini, **bukan rekonstruksi niat historis yang tidak tercatat**.
- Tidak ada kontrol tanpa bobot, random/grid search, atau ablasi individual deskriptor. Bukti untuk keunggulan komponen-komponen tersebut belum tersedia.

## F15–F16 — Apa yang dihitung dan belum dapat disimpulkan

Ambang F1: grid 0,05–0,95 langkah0,05 dari rata-rata prediksi OOF lima pengulangan. Tie: terdekat0,5, lalu lebih besar. Test tidak dipakai memilih ambang; validasi tetap terkait seleksi model, bukan evaluasi independen. Skor tidak divalidasi sebagai probabilitas empiris, intensitas, atau komposisi.

Interval seed menggunakan lima selisih AP berpasangan, sample SD, t(0,975;4)=2,7764451051977987, SE=SD/sqrt(5). Interval label menggunakan 10.000 resampling 109 indeks label dengan pengembalian, seed42, persentil2,5–97,5. Formula dicocokkan dengan kode; **tidak menjalankan bootstrap baru**. Label berkorelasi dan pengulangan CV bertumpang tindih, sehingga interval nominal/deskriptif. Melewati nol bukan uji ekuivalensi. Ketidakpastian berbasis kelompok molekul, kalibrasi dan evaluasi independen tetap belum dilakukan.

## Sumber primer dan batas pemakaian

- Suh: https://d-nb.info/1382461763/34 — metode, 8.681 molekul/100 label dievaluasi, RF/XGB/LGBM; tidak dijadikan target skor setara.
- Saini: https://www.nature.com/articles/s41598-022-18086-y — 7.374/109, 80:20, lima fold, micro averaging, representasi dan metode.
- Tyagi: https://pubmed.ncbi.nlm.nih.gov/37723894/ — abstrak primer memverifikasi 1.278/7, 1.875 fitur, PCA/XGB dan precision/sensitivity. Proporsi split dan averaging belum terverifikasi dari abstrak; tidak ditebak.
- Bergstra et al. (2011): https://papers.nips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html — dasar TPE, bukan pembenaran keunggulan TPE pada data penelitian.
- Fourches et al. (2016): https://pmc.ncbi.nlm.nih.gov/articles/PMC5657146/ — landasan kurasi kimia; metadata DOI 10.1021/acs.jcim.6b00129, JCIM56(7)1243–1252.
- Lipton et al. (2014): https://arxiv.org/abs/1402.1892 — threshold terkait objektif F1. Rumus khusus probabilitas terkalibrasi tidak diterapkan.
- Bobot: https://scikit-learn.org/1.6/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html
- Ambang: https://scikit-learn.org/stable/modules/classification_threshold.html
- Kalibrasi: https://scikit-learn.org/stable/modules/calibration.html
- RDKit: https://www.rdkit.org/docs/GettingStartedInPython.html dan dokumentasi Descriptors/Crippen/Lipinski/MolSurf — definisi/perhitungan fitur, bukan pembuktian subset optimal. Dokumentasi daring dapat lebih baru daripada versi eksperimen; versi eksperimen tetap dicatat.
- Efron (1979), Annals of Statistics7(1)1–26: https://blogs.helsinki.fi/bk-club/files/2012/05/Efron_Bootstrap_AS1979.pdf — dasar resampling bootstrap; tidak membenarkan asumsi label independen. Pilihan interval persentil dilaporkan sebagai implementasi aktual.
- NIST/SEMATECH: https://itl.nist.gov/div898/handbook/eda/section3/eda352.htm — bentuk interval mean; tidak membuktikan asumsi independensi repeated CV.
- Rujukan Cawley, Akiba, Tsoumakas, splitter, AP dan ONNX yang sudah ada dipertahankan pada klaim relevan.

## Keterbatasan pemeriksaan akhir

Kompilasi memakai Tectonic cached-only, untrusted, format XDV (bukan PDF). QA statis memeriksa kunci sitasi dan label; compiler memeriksa rujukan dan struktur. Tidak dilakukan inspeksi visual halaman hasil akhir, sehingga tidak ada klaim validasi tata letak visual lengkap. Peringatan underfull, metadata JPEG dan Fontconfig nonfatal dicatat; periksa log kompilasi untuk status final.

Bab1, dua abstrak, judul/settings, diagram dan SKRIPSI.tex dipertahankan karena telah sesuai. Lampiran administratif tidak dijadikan bukti ilmiah baru. Arsip bibliografi duplikat yang tidak aktif tidak diedit. Backup sumber sebelum revisi berada pada .local-backups/scientific-audit-20260922 di laporan.

Status tiap gap dan pekerjaan empiris tersisa terdapat pada bagian8 rencana audit. Dokumentasi yang diperbaiki tidak menutup kekurangan bukti ahli atau eksperimen.

## Ringkasan cakupan akhir

**Status: dapat ditinjau dengan keterbatasan; cakupan bukti empiris dan visual masih parsial.**
Perubahan sumber telah diterapkan dan kompilasi lulus. Angka pada kolom Observed defects adalah temuan cacat yang masih terbukti / jumlah unit dalam lingkup, bukan persentase akurasi atau jumlah seluruh pemeriksaan yang lulus. Nol temuan tidak berarti bukti baru sudah tersedia. Unit sumber/kesimpulan adalah 18 topik F01–F18; unit angka adalah sembilan tabel yang dipetakan dalam QA. Kategori tumpang tindih dan tidak dijumlahkan.

### Kelengkapan dan penyajian laporan

| Category | Observed defects | Assessment |
|---|---|---|
| Analytical clarity | 0 / 5 | Lima bab ditinjau; bagian yang benar dipertahankan dan batas klaim diperjelas. |
| Visual and interaction consistency | 0 / 6 | Kompilasi XDV berhasil tanpa overfull, tetapi tidak ada inspeksi visual penuh halaman; tidak ada antarmuka interaktif. Partial review: 6 scoped items have incomplete or stale evidence (literature-Suh; literature-Saini-Tyagi; descriptors; and 3 more (see review notes)). |
| Usefulness and completeness | 0 / 18 | Seluruh 18 topik audit memiliki status dan tindak lanjut; bukti baru tidak dinyatakan selesai. |

### Ketepatan analitis

| Category | Observed defects | Assessment |
|---|---|---|
| Source authority and confidence | 0 / 18 | Bukti kode/artefak dan rujukan diperiksa. F02 terbatas abstrak Tyagi; bukti ahli dan evaluasi baru belum tersedia. Partial review: 6 scoped items have incomplete or stale evidence (F02; F05; F09; and 3 more (see review notes)). |
| SQL/value accuracy | 0 / 9 | Sembilan tabel numerik dicocokkan; metrik dihitung kembali dari prediksi tersimpan tanpa menjalankan model. |
| Within-chart agreement | 0 / 9 | Nilai dan label sembilan tabel dicocokkan dengan hasil tersimpan; bukan pemeriksaan visual halaman. |
| Complete source details | 0 / 18 | Peta F01–F18 ke berkas/kode/sumber disimpan dalam EVIDENCE_REVIEW.md; batas sumber dicatat. |
| Cross-artifact consistency | 0 / 1 | Bab 1–5, dua abstrak, judul dan status backend konsisten pada lingkup F01–F18. |
| Data-quality controls | 0 / 5 | Penolakan, harmonisasi, dukungan, struktur dan kelompok diperiksa dari kode/arsip; validasi ahli belum tersedia. |
| Conclusion support | 0 / 18 | Klaim dibatasi pada desain eksploratori dan hasil internal; kebutuhan empiris tetap terbuka. |

Catatan cakupan yang dapat diperiksa: `qa/scientific_audit_20260922/coverage.json` pada laporan. Ringkasan diperoleh dengan summarize-coverage.mjs dari panduan validasi; helper memeriksa konsistensi pencatatan, bukan menggantikan pembacaan sumber. Berkas verification.json mencatat hash sumber, rincian angka, dan sampel metadata model. Gap prioritas tetap validasi ahli F05, kalibrasi F15, dan mobile F17; kebutuhan independensi evaluasi F09 serta unit inferensi F16 tidak dihapus oleh koreksi narasi.


## Catatan revisi lanjutan — 23 September 2026

Dokumen ini menyimpan bukti snapshot 22 September. Project Context/Overview dan diagram delapan kandidat telah diperbarui pada 23 September, disertai perubahan caption/paragraf Bab 3. Untuk versi tersebut, gunakan bagian 10 SCIENTIFIC_AUDIT_FIX_PLAN_20260921.md dan qa/docs_diagram_v6_20260923 pada laporan. Halaman diagram dan Bab 3 yang terdampak sudah diperiksa visual; tidak ada evaluasi ML atau validasi empiris baru. Pemeriksaan kesamaan Bab 3/diagram dengan snapshot lama tidak lagi menjadi syarat revisi baru.
