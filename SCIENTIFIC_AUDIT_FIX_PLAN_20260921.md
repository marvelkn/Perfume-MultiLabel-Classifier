# Rencana Perbaikan Audit Ilmiah Skripsi — 21 September 2026

## 1. Tujuan dan status

Dokumen ini adalah daftar tindak lanjut audit kode, hasil grouped-v6, laporan LaTeX, dan landasan literatur. Tujuannya menutup gap argumentasi, sitasi, konsistensi, dan batas klaim. Dokumen ini BUKAN bukti bahwa semua perbaikan atau eksperimen sudah dilaksanakan.

**Status 22 September 2026:** revisi laporan dan dokumentasi berbasis bukti telah diterapkan. Status per gap terdapat pada Bagian 8; validasi ahli, kalibrasi, evaluasi independen, dan implementasi mobile tidak dianggap selesai. Bagian yang telah benar dipertahankan. Bukti rinci: [EVIDENCE_REVIEW.md](reports/scientific_audit_20260922/EVIDENCE_REVIEW.md).

**Prioritas:** perbaiki laporan dan dokumentasi berdasarkan bukti yang tersedia. Tidak ada perintah menjalankan training baru, preprocessing ulang, pseudo-labeling, inferensi massal data ditolak, konversi ONNX, integrasi Android, atau deployment. Pengembangan ONNX dan aplikasi tetap ditunda.

Metode memerlukan landasan ilmiah. Angka rancangan seperti seed, jumlah pengulangan, dan budget tidak harus meniru jurnal: jelaskan alasan, kompromi, serta keterbatasannya. Jangan mencari sitasi untuk membenarkan hasil secara retrospektif atau mengarang alasan seolah sudah ditetapkan sebelum eksperimen.

## 2. Lokasi kerja dan urutan pemeriksaan

- Backend utama: `C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier`
- Laporan utama: `C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael`
- Frontend: referensi baca saja. Dilarang mengedit frontend, termasuk dokumentasinya, dalam tindak lanjut ini.
- Jangan mengedit salinan laporan lama di `C:\Users\ACER\Documents\ChatGPT\Project Skripsi\Laporan-Update-20260904`.

Urutan kerja:
1. Baca `SAFE_EXECUTION_PLAN.md` lebih dahulu, lalu `README.md`, `IMPLEMENTATION_STATUS.md`, dan `INFERENCE_GUIDE.md` pada backend.
2. Baca `AGENTS.md` pada laporan. Ikuti pemeriksaan referensi frontend yang diminta sejauh read-only; apabila ada instruksi memperbarui changelog frontend, batas pengguna yang melarang edit frontend didahulukan. Catat perubahan pada dokumen ini sebagai gantinya.
3. Baca bab dan bagian awal/akhir yang relevan, lalu cocokkan dengan kode dan artefak hasil, bukan hanya ringkasan percakapan.
4. Verifikasi sumber primer sebelum memasukkan klaim atau entri bibliografi baru. Jangan membuat judul, DOI, angka, atau hasil penelitian.
5. Terapkan perbaikan laporan/dokumentasi yang didukung bukti. Untuk kebutuhan bukti baru, catat sebagai belum dikerjakan atau ditunda.
6. Periksa sitasi, struktur LaTeX, istilah, tabel, dan encoding; kompilasi jika compiler tersedia. Laporkan bila kompilasi tidak dapat dilakukan.

Berkas laporan yang perlu ditinjau:
- `src/2.Isi/Bab1.tex` sampai `Bab5.tex`.
- `src/1.Awal/07a-Abstrak.tex` dan `07b-Abstract.tex`.
- Sampul/judul, lampiran, dan bagian lain hanya jika ada klaim yang perlu diselaraskan.
- `SKRIPSI.tex` untuk konfigurasi kompilasi/bibliografi.
- Bibliografi aktif yang ditemukan saat audit adalah `pustaka.bib` di ROOT laporan, sesuai `\bibliography{pustaka}`. Jangan mengedit duplikat `src/3.Akhir/pustaka.bib` tanpa memeriksa apakah benar digunakan.

Bukti backend utama (relatif terhadap root backend):
- `alignment/perfume_data.py`, `src/build_dataset.py`, `src/label_harmonization.py`.
- `alignment/grouped_splits.py`, `alignment/experiments.py`, `alignment/protocol_v6.json`.
- Implementasi evaluasi tambahan dan bootstrap pada modul eksperimen terkait; cari definisi aktual, jangan mengasumsikan nama fungsi.
- `.local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v6/perfume-campus-grouped-v6/runs/perfume-five-grouped-v6/`.
- Dataset beku pada paket yang sama: `data/builds/perfume-five-grouped-v2/`.
- `releases/grouped-v6-lgbm-d/` dan `reports/backend_alignment_20260921/verification.json`.
- `src/inference.py`, `src/featurize.py`, `src/release.py`, `api_light.py`, `feature_spec.json`.

Jika lokasi berpindah, cari manifest/artefak aktual. Jangan membuat angka pengganti. Catat perbedaan dengan dokumen ini.

## 3. Fakta acuan yang harus dipertahankan

| Item | Fakta saat audit |
|---|---|
| Input penelitian | Satu molekul, direpresentasikan sebagai SMILES; model menerima fitur numerik |
| Sumber | GoodScents, IFRA 2019, Leffingwell, Arctander 1960, Sigma-Aldrich 2014 |
| Baris sumber | 13.337 |
| Ditolak preprocessing | 1.560 |
| Diterima sebelum konsolidasi | 11.777 |
| Berkurang akibat konsolidasi catatan | 5.091; bukan 5.091 molekul tak berlabel yang dibuang |
| Molekul unik | 6.686 |
| Train / test | 5.353 / 1.333 molekul |
| Kelompok | 5.363 menurut aturan struktur/fingerprint aktual |
| Label terpilih | 109 |
| Fitur gabungan | Morgan 1.024 bit + lima deskriptor = 1.029 fitur |
| Morgan | Radius 2, tanpa kiralitas, sesuai konfigurasi yang diaudit |
| Model terpilih | LightGBM D berdasarkan AP validasi |
| AP validasi | LightGBM D 0,302232; XGBoost D 0,298848 |
| AP test | LightGBM D 0,313921; XGBoost D 0,314531 |
| AUROC test | LightGBM D 0,843526; XGBoost D 0,838731 |
| Macro-F1 test, ambang 0,5 | LightGBM D 0,319645; XGBoost D 0,308414 |
| Macro precision / recall LightGBM D, ambang 0,5 | 0,249324 / 0,481812 |
| Semua label diprediksi negatif | Accuracy 0,960970; recall dan F1 nol |
| Bukti konsistensi fitur | 6.686 molekul, identik dengan fitur beku |
| Bukti konsistensi prediksi Python | 32 contoh, 3.488 skor, selisih maksimum nol |
| Cakupan verifikasi tersebut | Konsistensi implementasi; bukan evaluasi Android atau estimasi performa baru |
| Status eksperimen | Eksploratori: test telah diperiksa pada eksperimen sebelumnya |

Jangan mengubah angka di artefak agar sesuai narasi. Bila artefak berbeda, laporkan asal perbedaan dan koreksi narasinya berdasarkan sumber yang terverifikasi.

## 4. Daftar perbaikan lengkap

### F01 — Posisi Suh dan definisi kontribusi [P0, laporan]

**Gap:** narasi lama dapat memberi kesan Suh hanya menggunakan Random Forest, sedangkan penelitian kita pertama memakai XGBoost/LightGBM/RDKit.

**Tindakan:** periksa sumber Suh. Akui algoritma dan representasi yang sudah digunakan sebelumnya. Posisikan penelitian ini sebagai evaluasi konfigurasi fitur dan tuning pada dataset yang disusun, bukan penciptaan algoritma baru. Morgan sendiri juga dihitung dengan RDKit; kontras A–B bukan “tanpa RDKit vs dengan RDKit”.

**Selesai bila:** Bab 1/2, abstrak, dan kesimpulan konsisten; tidak ada klaim pertama atau peningkatan atas Suh yang tidak didukung eksperimen setara.

### F02 — Perbandingan antarpenelitian [P0, laporan]

**Gap:** metrik bernama sama belum membuat dataset, label, split, dan evaluasi setara.

**Tindakan:** tabel penelitian terdahulu mencantumkan input, target, data, jumlah label, split, dan jenis averaging metrik sejauh tersedia. Jangan menebak yang tidak dilaporkan. Bedakan AP dari perhitungan area PR lain, macro-F1 dari micro-F1, serta accuracy per-label dari subset accuracy. Hasil Saini/Tyagi atau penelitian lain menjadi konteks, bukan perlombaan angka langsung. Tidak perlu menambahkan eksperimen Random Forest tanpa perubahan scope dari pengguna.

**Selesai bila:** klaim utama bertumpu pada perbandingan internal A–D; angka penelitian lain disertai batas kesetaraannya.

### F03 — Ruang lingkup parfum dan pemilihan lima sumber [P0, laporan]

**Gap:** relevansi sumber dengan parfum tidak membuktikan semua molekul hanya digunakan pada parfum.

**Tindakan:** gunakan narasi “prediksi aroma molekul dalam konteks parfum”. Buat alasan pemilihan per sumber berdasarkan deskripsi sumber, ketersediaan identitas/struktur, dan anotasi aroma yang benar-benar tersedia. Jangan menyebut kelimanya paling relevan tanpa perbandingan sumber yang mendukung. Jangan menyatakan ada penyaringan molekul khusus parfum apabila tidak dilakukan.

**Selesai bila:** ada kriteria inklusi sumber yang transparan; tidak ada klaim semua molekul eksklusif parfum, keamanan bahan, atau rekomendasi formulasi.

### F04 — Alur jumlah data dan alasan penolakan [P0, laporan/dokumentasi]

**Gap:** 13.337 menjadi 6.686 berpotensi disalahartikan sebagai sekitar 7.000 molekul dibuang.

**Tindakan:** pisahkan baris mentah, baris ditolak, baris diterima, konsolidasi identitas, dan molekul unik. Ambil rincian penolakan per sumber dari audit dataset. Bedakan gagal pencocokan identitas, multikomponen, label kosong, dan label tak terpetakan. Jangan menyebut semuanya data tak berlabel yang siap dipakai ulang.

**Selesai bila:** seluruh jumlah cocok dan istilah baris/molekul/label konsisten. Tidak ada preprocessing ulang.

### F05 — Harmonisasi label dan penggabungan anotasi [P1, audit/dokumentasi]

**Gap:** pemetaan sinonim dan union label antar sumber belum divalidasi pakar sensori.

**Tindakan:** dokumentasikan istilah asli, istilah target, aturan, dan alasan. Tinjau pemetaan yang ambigu dengan sumber kamus/sumber asli. Bedakan normalisasi penulisan dari penyamaan arti. Nyatakan status validasi ahli apa adanya; jangan mengklaim validasi ahli sudah terjadi. Catat bahwa penggabungan dapat menambah cakupan sekaligus membawa noise anotasi.

**Selesai bila:** aturan dapat ditelusuri; pemetaan meragukan masuk daftar untuk review, tanpa mengganti dataset beku atau model.

### F06 — Makna label negatif dan dukungan label [P0, laporan]

**Gap:** label tidak tercatat diperlakukan sebagai 0, padahal belum tentu aroma benar-benar tidak ada.

**Tindakan:** jelaskan asumsi anotasi tersebut dan dampaknya pada precision/recall. Dokumentasikan syarat dukungan training aktual, termasuk minimum 30 positif dan syarat kelas negatif. Kaitkan preseden penyaringan dukungan pada literatur bila sesuai, tetapi jangan menyebut 30 optimal universal. Bahas bias ke label umum, keterbatasan test dengan sedikit positif, serta baris tanpa label positif setelah seleksi jika memang ada.

**Selesai bila:** tidak ada klaim 0 sebagai hasil uji sensoris negatif; seleksi label tidak menggunakan test.

### F07 — Standardisasi struktur dan stereokimia [P1, laporan/audit]

**Gap:** canonical SMILES bukan penyelesaian seluruh variasi tautomer/protonasi; fingerprint tanpa kiralitas menghilangkan informasi stereokimia.

**Tindakan:** tulis apa yang dilakukan dan tidak dilakukan: validasi RDKit, fragmen, identitas, canonicalisasi, tautomer/protonasi, dan kiralitas. Audit kasus relevan tanpa mengubah kimia secara otomatis. Jangan langsung menetralkan atau mengambil fragmen terbesar untuk menambah data. Tambahkan dasar kurasi kimia [R4].

**Selesai bila:** kebijakan dan batasnya jelas. Perubahan representasi, bila kelak diperlukan, menjadi eksperimen baru terpisah.

### F08 — Batas pembagian kelompok [P0, laporan/dokumentasi]

**Gap:** aturan saat ini menggabungkan struktur nonstereokimia identik ATAU fingerprint identik; belum mengelompokkan seluruh kemiripan struktur.

**Tindakan:** jelaskan aturan union/transitif yang benar-benar digunakan, ukuran kelompok, dan pemeriksaan tidak adanya overlap kelompok antarpartisi. Bedakan identitas kimia, kesamaan fitur, dan kemiripan parsial. Kesamaan hashed fingerprint tidak membuktikan dua molekul identik. Jangan menamainya scaffold split atau clustering semua molekul mirip. Jelaskan bahwa fold validasi dibentuk per label sesuai implementasi dan digunakan konsisten antar kandidat.

**Selesai bila:** kode dan narasi sama; dasar grouped splitting disitasi, aturan khusus diakui sebagai rancangan penelitian.

### F09 — Test yang sudah dilihat dan repeated CV [P0, laporan]

**Gap:** test pernah diperiksa pada putaran sebelumnya; repeated CV bukan nested CV dan nilai CV yang digunakan untuk tuning bukan estimasi independen.

**Tindakan:** pertahankan status eksploratori [R6]. Penguncian konfigurasi hanya menjaga urutan keputusan tahap terkait, tidak menghapus riwayat eksplorasi. Jangan menyebut seed/fold sebagai dataset independen. Jangan menganggap random split ulang akan menghasilkan test yang benar-benar belum pernah memengaruhi penelitian.

**Selesai bila:** Bab 3/4/5 dan abstrak tidak menyebut hasil sebagai konfirmasi independen. Evaluasi eksternal/nested menjadi opsi lanjutan, bukan hasil selesai atau kewajiban training otomatis.

### F10 — Baseline, binary relevance, dan bobot kelas [P0, laporan]

**Gap:** bobot N/(2Nc) belum memiliki rujukan langsung yang cukup; 100 pohon tidak berarti kapasitas kedua algoritma sama.

**Tindakan:** tambahkan dasar binary relevance yang sudah tersedia, rujukan bobot seimbang [R2], dan rumus dengan definisi N/Nc. Cocokkan bobot dihitung hanya dari training fold. Dokumentasikan parameter efektif baseline, seed, versi library, dan parameter tetap/default yang relevan dari bukti aktual. Jelaskan bahwa hiperparameter dituning bersama untuk seluruh 109 klasifier label, bukan optimalisasi tiap label terpisah.

**Selesai bila:** pembobotan dan baseline dapat direproduksi. Tidak ada klaim pembobotan lebih baik tanpa kontrol tanpa bobot.

### F11 — Alasan paket lima deskriptor [P1, laporan]

**Gap:** sitasi RDKit menjelaskan cara menghitung fitur, bukan mengapa subset tepat lima adalah yang terbaik.

**Tindakan:** ambil nama lima deskriptor dari kode, jelaskan sifat kimia yang diwakili dan alasan memilih paket ringkas tersebut. Bedakan alasan ilmiah saat ini dari alasan historis yang benar-benar tercatat. Kutip sumber relevan; jangan mengklaim formula/subset diambil dari Suh jika tidak persis demikian.

**Selesai bila:** paket lima disebut variabel yang diuji; tidak ada klaim setiap deskriptor pasti membantu. Ablasi satu-per-satu hanya opsi penelitian lanjutan.

### F12 — Optuna, TPE, ruang pencarian, dan budget [P0, laporan]

**Gap:** terbaik yang ditemukan dapat terbaca sebagai optimum global; tuning lebih lama tidak menjamin perbaikan.

**Tindakan:** pertahankan sitasi Akiba, tambahkan dasar TPE [R3]. Jelaskan objective AP aktual, rentang/distribusi parameter, parameter tetap, seed, trial awal historis yang dievaluasi ulang, jumlah trial dan durasi aktual bila tersedia, serta ada/tidaknya pruning dan early stopping. V6 menggunakan empat studi delapan jam; jangan mengubahnya menjadi delapan jam total per model. Kesamaan waktu adalah kebijakan sumber daya, bukan kesamaan jumlah trial atau kapasitas model. Jelaskan lima fold/lima pengulangan dan batas dukungan kelas sesuai kode, tanpa klaim angka paling optimal.

**Selesai bila:** tertulis “konfigurasi terbaik yang ditemukan dalam ruang pencarian dan anggaran yang ditetapkan”. Tidak ada klaim TPE mengalahkan random/grid search karena kontrol itu belum dilakukan.

### F13 — Interpretasi A/B/C/D dan riwayat versi [P0, laporan]

| Kontras | Interpretasi yang diperbolehkan |
|---|---|
| B–A | Pengaruh penambahan paket deskriptor pada baseline masing-masing algoritma |
| C–A dan D–B | Tuning dibanding konfigurasi baseline pada representasi terkait |
| D–C | Dua representasi yang masing-masing dituning; parameter optimalnya dapat berbeda |
| D–A | Perubahan gabungan fitur dan tuning |

**Tindakan:** cantumkan hasil yang turun juga. Bedakan selisih absolut skor dari persen kenaikan relatif. Riwayat v3–v6 mengubah lebih dari waktu pencarian; bukan eksperimen terkontrol tentang pengaruh lama training saja. Lima sumber tidak dibandingkan langsung dengan seluruh sumber Suh.

**Selesai bila:** kesimpulan hanya mengenai kontras yang benar-benar dijalankan; tidak ada klaim semua metrik membaik.

### F14 — Metrik, ketimpangan label, dan per-label [P0, laporan]

**Tindakan:** pertahankan macro AP sebagai objective sesuai protokol. Jelaskan averaging aktual per fold/label/pengulangan, penanganan kelas tak terdefinisi, serta perbedaan evaluasi threshold 0,5 dan threshold validasi. Sertakan baseline semua-negatif sebagai ilustrasi keterbatasan accuracy. Tampilkan dukungan positif bersama metrik per label; AP tinggi pada sedikit contoh tidak menjamin stabilitas. Jangan memilih label bagus saja sebagai bukti keberhasilan.

**Selesai bila:** angka utama dan definisi sama dengan kode; accuracy tidak dijelaskan sebagai persentase molekul yang seluruh labelnya tepat. Pelaporan AP/Saito yang sudah benar tidak perlu diganti.

### F15 — Threshold F1 dan kalibrasi skor [P0, laporan]

**Gap:** pemilihan threshold per label belum memiliki dasar langsung; skor belum diuji kalibrasinya.

**Tindakan:** tambahkan [R1] dan [R5]. Jelaskan grid aktual 0,05–0,95 dengan langkah 0,05, prediksi out-of-fold yang dirata-ratakan sesuai kode, serta tidak memakai test untuk memilih threshold. Karena model/parameter dipilih pada validasi yang terkait, jangan menganggap skor tersebut bebas bias seleksi. Pertahankan analisis threshold sebagai tambahan, bukan mengganti hasil utama setelah melihat test.

**Selesai bila:** skor bukan intensitas atau persentase komposisi; skor 80% tidak dijanjikan sebagai ketepatan empiris 80%. Kalibrasi belum diklaim selesai. Jangan menerapkan rumus threshold dari literatur yang memerlukan kalibrasi tanpa memenuhi syaratnya.

### F16 — Bootstrap dan interval pengulangan [P0, laporan; analisis baru opsional]

**Gap:** bootstrap 109 label memakai unit yang dapat berkorelasi; repeated CV saling bertumpang tindih.

**Tindakan:** lengkapi referensi statistik yang terverifikasi untuk bootstrap/interval yang dijelaskan. Pertahankan nama “interval nominal/deskriptif”, unit label, jumlah pengambilan ulang, dan keterbatasannya. Interval melewati nol bukan bukti dua model setara. Interval tidak melewati nol pada seed yang bertumpang tindih bukan bukti independen superioritas populasi.

**Opsi lanjutan, jangan otomatis dijalankan:** paired bootstrap pada kelompok molekul dari prediksi tersimpan, dengan definisi target inferensi, penanganan label minim positif, dan dependensi yang jelas. Ini bisa tidak memerlukan retraining, tetapi merupakan analisis baru yang harus dibedakan dari hasil lama. Evaluasi pada test lama tetap eksploratori.

**Selesai bila:** klaim signifikansi/equivalence tidak melampaui rancangan; dasar metode disitasi tanpa menyamarkan masalah unit analisis.

### F17 — Status backend, ONNX, dan mobile [P0, laporan; implementasi ditunda]

**Gap:** parity Python tidak sama dengan model sudah diuji pada Android.

**Tindakan sekarang:** jelaskan apa yang sudah dibuktikan oleh verification.json dan keterbatasan sampelnya. Sebut aplikasi sebagai demonstrasi pendukung dengan status aktual. Jangan menghidupkan kembali klaim ONNX lama 25 label. Angka ukuran artefak Python sekitar 365 MB saat audit bukan ukuran ONNX yang sudah diukur.

**Checklist nanti, bukan dikerjakan sekarang:**
- Putuskan arsitektur: inference server, fitur dari server + inference lokal, atau seluruh alur lokal.
- ONNX tidak otomatis mengubah SMILES menjadi fitur RDKit; jika fitur melalui API, alur belum sepenuhnya offline.
- Verifikasi operator/runtime, ukuran hasil konversi, RAM, waktu muat, latensi cold/warm dan p50/p95, serta overhead jaringan bila ada [R7].
- Uji parity Python–ONNX pada skor dan keputusan, urutan fitur/label, dtype, threshold, dan input invalid.
- Pastikan model/skema/label/threshold berversi konsisten dan memiliki penanganan kegagalan.
- Jangan mengklaim manfaat pengguna, evaluasi sensori, keamanan bahan, atau kelayakan formulasi tanpa pengujian tersendiri.

**Selesai untuk scope saat ini bila:** laporan membedakan backend selesai dari pekerjaan mobile yang ditunda. Pengujian perangkat tetap berstatus belum dilakukan.

### F18 — Konsistensi lintas laporan dan reproducibility [P0, laporan/dokumentasi]

**Tindakan:** telusuri abstrak Indonesia/Inggris, rumusan masalah, tujuan, metode, tabel hasil, kesimpulan, dan lampiran. Samakan 109 label, 1.029 fitur, sumber data, model terpilih, metrik/ambang, status eksploratori, dan status mobile. Simpan tautan bukti serta versi/checksum yang sudah ada. Jangan mengganti berkas hasil atau menandai rencana sebagai implementasi selesai.

**Selesai bila:** tidak ada kontradiksi lintas bab; seluruh perubahan tercatat dengan file dan alasan; sitasi baru tersambung ke bibliografi aktif.

## 5. Sumber pendukung dan cara memakainya

Sumber di bawah telah dirujuk dalam audit. Tetap buka sumber primer dan periksa metadata lengkap sebelum membuat BibTeX; artikel tidak otomatis mendukung semua keputusan angka penelitian.

- **R1 — Lipton, Elkan, Narayanaswamy (2014), Thresholding Classifiers to Maximize F1 Score.** Dasar pemilihan ambang terkait F1 pada klasifikasi biner/multi-label. https://arxiv.org/abs/1402.1892
- **R2 — scikit-learn 1.6.1, compute_sample_weight.** Formula pembobotan balanced. https://scikit-learn.org/1.6/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html
- **R3 — Bergstra et al. (2011), Algorithms for Hyper-Parameter Optimization.** Dasar algoritma TPE; melengkapi Akiba/Optuna. https://papers.nips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html
- **R4 — Fourches et al. (2016), Trust, but Verify II: A Practical Guide to Chemogenomics Data Curation.** Dasar kurasi data kimia. https://pmc.ncbi.nlm.nih.gov/articles/PMC5657146/
- **R5 — scikit-learn, Tuning the decision threshold for class prediction.** Pemisahan pemilihan threshold dari data evaluasi. Dokumentasi bukan bukti kode kita menggunakan estimator bawaan tersebut. https://scikit-learn.org/stable/modules/classification_threshold.html
- **R6 — Cawley dan Talbot (2010), On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.** Bias pemilihan model/evaluasi. Sudah ada di bibliografi saat audit; gunakan ulang jika metadata benar. https://jmlr.org/papers/v11/cawley10a.html
- **R7 — ONNX Runtime Mobile.** Kompatibilitas dan evaluasi implementasi perangkat. Sudah ada rujukan ONNX pada laporan; cek kecukupan dan hindari duplikasi. https://onnxruntime.ai/docs/tutorials/mobile/
- **R8 — Suh et al., PDF artikel acuan.** Verifikasi model, data, representasi, label, dan metrik pada artikel asli. https://d-nb.info/1382461763/34
- **R9 — scikit-learn, Probability calibration.** Dasar membedakan skor model dari probabilitas yang sudah diuji kalibrasinya. https://scikit-learn.org/stable/modules/calibration.html

Rujukan yang ditambahkan setelah verifikasi adalah Efron (1979), *Bootstrap Methods: Another Look at the Jackknife*, The Annals of Statistics 7(1):1–26, sebagai dasar bootstrap, serta NIST/SEMATECH *Confidence Limits for the Mean* untuk bentuk interval mean. Prosedur persentil aktual dicocokkan dengan kode tersimpan. Rujukan tersebut tidak membenarkan asumsi bahwa label berkorelasi atau pengulangan CV merupakan sampel independen.

Rujukan yang sudah ada, seperti Saini, Hamel, Morgan/Rogers, Tsoumakas, Chen, Ke, Akiba, Saito, MoleculeNet/Wu, RDKit, dan dokumentasi splitter, harus diperiksa lalu digunakan ulang. Tidak semua gap merupakan ketiadaan sumber di bibliografi; sebagian adalah sitasi yang belum ditempatkan pada klaim terkait, atau klaim yang belum didukung bukti.

## 6. Batas klaim: contoh kalimat

**Disarankan:**
- “Pada dataset dan pembagian data yang digunakan, tambahan lima deskriptor meningkatkan AP dibanding representasi Morgan pada konfigurasi baseline kedua algoritma.”
- “LightGBM D dipilih berdasarkan AP validasi tertinggi; perbandingan test menunjukkan pertukaran antarmetrik.”
- “Penelitian menggunakan lima sumber yang relevan dengan konteks parfum, tanpa menyatakan bahwa seluruh molekul digunakan secara eksklusif dalam parfum.”
- “Hasil bersifat eksploratori karena data test telah diperiksa dalam eksperimen pendahuluan.”
- “Konfigurasi yang dilaporkan merupakan hasil terbaik yang ditemukan dalam ruang pencarian dan anggaran eksperimen.”

**Hindari:**
- “Model terbukti mengalahkan Suh” hanya karena skor berbeda pada dataset berbeda.
- “RDKit/Optuna/XGBoost adalah metode baru yang pertama diperkenalkan penelitian ini.”
- “Semua data adalah molekul khusus parfum.”
- “Accuracy 95% berarti 95% molekul diprediksi seluruh aromanya dengan tepat.”
- “Optuna pasti meningkatkan semua metrik” atau “delapan jam menjamin optimal”.
- “Tidak signifikan berarti kedua model sama.”
- “Semua molekul mirip sudah dipisahkan” pada grouping identitas/fitur saat ini.
- “Android/offline/ONNX sudah berjalan” tanpa artefak dan pengujian aktual.

## 7. Urutan eksekusi dan kriteria selesai

### Tahap A — Perbaikan tanpa training
- [x] Verifikasi fakta dan berkas terbaru; catat ketidaksesuaian terhadap snapshot dokumen ini.
- [x] Buat peta gap F01–F18: sudah benar / perlu edit / butuh bukti tambahan / ditunda.
- [x] Perbaiki narasi dan sitasi di seluruh bagian laporan yang benar-benar terdampak.
- [x] Lengkapi alasan desain dan dokumentasi yang didukung kode/artefak.
- [x] Tinjau contoh pemetaan label/struktur yang meragukan; jangan mengubah dataset beku.
- [x] Periksa perhitungan/tabel dari hasil tersimpan tanpa menjalankan model ulang.
- [x] Periksa dan kompilasi LaTeX bila tersedia; laporkan hasilnya secara jujur.
- [x] Laporkan file yang berubah dan gap yang masih terbuka.

### Tahap B — Kebutuhan bukti tambahan, belum diotorisasi
- [ ] Review pakar atas harmonisasi label bila dapat dilakukan.
- [ ] Analisis ketidakpastian dengan unit yang lebih sesuai, bila diperlukan.
- [ ] Evaluasi independen, kalibrasi, atau ablasi tambahan hanya dengan rancangan dan instruksi baru.

### Tahap C — Mobile, tetap ditunda
- [ ] Jalankan checklist F17 setelah pengguna memulai fase implementasi mobile.

Dokumentasi tidak bisa membuat validasi ahli, test independen, atau pengujian Android menjadi selesai. Jika bukti belum tersedia, hasil perbaikannya adalah batasan yang jelas dan rencana terpisah, bukan klaim masalah empiris sudah hilang.

## 8. Format laporan penyelesaian untuk chat pelaksana

Isi tabel berikut setelah bekerja; jangan menandai semua selesai hanya karena narasi sudah ditambahkan.

| ID gap | Status | Berkas/bagian yang diubah | Bukti/sitasi | Sisa pekerjaan |
|---|---|---|---|---|
| F01 | sudah benar | Bab 1/2, abstrak, Bab 5 dipertahankan pada posisi kontribusi | Suh; perbandingan internal A–D | Tidak ada klaim algoritma/RDKit baru atau kemenangan atas Suh |
| F02 | diperbaiki | Bab 2: tabel posisi dan konteks studi | Saini/Suh primer; abstrak Tyagi | Split/averaging Tyagi belum terverifikasi penuh; ditandai, tidak ditebak |
| F03 | diperbaiki | Bab 3: kriteria inklusi dan tabel per sumber | Snapshot sumber dan jalur pencocokan aktual | Tidak ada validasi seluruh molekul eksklusif parfum |
| F04 | diperbaiki | Bab 4: tabel alasan penolakan dipisah | audit.json; QA rekonsiliasi | Tidak ada preprocessing ulang |
| F05 | butuh bukti tambahan | Bab 3; EVIDENCE_REVIEW.md: 29 alias, contoh asli, antrean semantik | ALIASES aktual, label_mappings.csv, snapshot sumber | Validasi pakar belum dilakukan; tropical/animal/musk dan alias lainnya belum dinyatakan ekuivalen |
| F06 | diperbaiki | Bab 3: makna nol, bias dukungan dan interpretasi precision/recall | Seleksi training 30 positif/2 negatif; 11 baris training semua nol | Tidak ada verifikasi negatif sensori atau optimalitas ambang dukungan |
| F07 | diperbaiki | Bab 3; audit dua pasangan struktur tersimpan | Fourches; groups.csv; kode kanonisasi/Morgan | Normalisasi kimia/representasi kiral baru tidak dilakukan |
| F08 | diperbaiki | Bab 3: batas hashing, komponen transitif, ukuran kelompok | 5.363 kelompok, maksimum38; audit split/fold; dokumentasi splitter | Tidak menguji seluruh kemiripan parsial/scaffold |
| F09 | sudah benar | Narasi eksploratori Bab 3/4/5 dan abstrak dipertahankan | Protokol v6; Cawley | Evaluasi independen/nested masih belum dilakukan; status benar hanya untuk pelaporan batasan |
| F10 | diperbaiki | Bab 3: tabel baseline, rumus/sitasi bobot dan batas kontrol | Kode, recipe; sampel metadata empat model ber-SHA sesuai; sklearn | Keunggulan bobot versus tanpa bobot belum diuji |
| F11 | diperbaiki | Bab 2: makna lima deskriptor; Bab 4/5: batas ablasi | RDKit dan urutan fitur aktual | Ablasi individual belum dilakukan; subset optimal tidak diklaim |
| F12 | diperbaiki | Bab 2/3: TPE, parameter tetap, anggaran dan batas perbandingan | Bergstra/Akiba; protocol_v6.json; tuning_summary | Tidak ada kontrol random/grid atau bukti optimum global |
| F13 | sudah benar | Kontras, penurunan metrik dan riwayat Bab 3/4/5 dipertahankan | ablation_deltas.csv dan comparison_history.csv; QA tabel | Tidak dianggap eksperimen terkontrol pengaruh waktu saja |
| F14 | sudah benar | Definisi metrik, baseline negatif dan contoh per-label dipertahankan | Prediksi tersimpan seluruh delapan kandidat; QA hitung ulang | Dukungan kecil tetap membatasi stabilitas label |
| F15 | butuh bukti tambahan | Bab 2/3: rujukan, tie-break, bias seleksi, batas interpretasi skor | Lipton; sklearn threshold/calibration; kode dan thresholds.json | Kalibrasi empiris dan validasi independen ambang belum dilakukan |
| F16 | diperbaiki | Bab 4: rumus t, unit/resampling dan bukan ekuivalensi; Bab 5 | Efron/NIST; kode statistik dan hasil v6 tersimpan | Bootstrap kelompok molekul dan rancangan inferensi yang lebih kuat belum dilakukan |
| F17 | ditunda | Status Python/mobile yang sudah benar dipertahankan | INFERENCE_GUIDE; verification.json backend | ONNX, arsitektur akhir, deployment, parity/performa Android belum dilakukan |
| F18 | diperbaiki | Bibliografi aktif, QA, changelog dan dokumen ini | Pemeriksaan sitasi, checksum, tabel dan kompilasi | Tidak ada inspeksi visual penuh halaman; batas empiris di atas tetap berlaku |

Status yang digunakan: **sudah benar**, **diperbaiki**, **butuh bukti tambahan**, **ditunda**, atau **terhambat**.

Ringkasan akhir wajib menyebut:
1. Seluruh file yang berubah, dengan lokasi utama yang benar.
2. Klaim yang dikoreksi dan sitasi yang ditambahkan.
3. Hasil pemeriksaan sitasi/LaTeX dan status kompilasi.
4. Gap empiris yang belum selesai dan alasannya.
5. Konfirmasi bahwa dataset, model, split, angka v6, frontend, dan tahap ONNX/Android tidak diubah.

## 9. Hasil verifikasi dan berkas revisi

- Sumber laporan yang diubah: `src/2.Isi/Bab2.tex`, `Bab3.tex`, `Bab4.tex`, `Bab5.tex`, `pustaka.bib` aktif, dan `CHANGELOG.md`.
- Pendukung baru pada laporan: `qa/scientific_audit_20260922/verify_evidence.py`, hasil `verification.json`, catatan cakupan `coverage.json`, serta berkas pemeriksaan kompilasi XDV/log/AUX/BibTeX. Tidak membuat PDF atau ZIP.
- Dokumentasi backend: rencana ini dan `reports/scientific_audit_20260922/EVIDENCE_REVIEW.md`. Kode backend operasional, README/status yang masih benar, serta frontend tidak diedit.
- Bab 1, kedua abstrak, judul, konfigurasi utama, diagram yang disederhanakan, dan angka hasil dipertahankan.
- QA menghitung ulang metrik dari prediksi tersimpan tanpa fit/predict; memeriksa checksum completion1.198, selection260, dataset14, source24 (hitungan per manifest, dapat beririsan).
- Sitasi baru: Bergstra, Fourches, Lipton, Efron, NIST, dokumentasi bobot/ambang/kalibrasi sklearn, dan deskriptor RDKit. Tidak ada sumber atau bukti sensori yang dikarang.
- QA akhir lulus: 74 baris pada sembilan tabel numerik cocok, 33 kunci sitasi tersedia, tanpa duplikasi kunci bibliografi atau label LaTeX. Kompilasi Tectonic ke XDV berhasil, 64 halaman, tanpa sitasi/referensi tidak terdefinisi maupun overfull box. Peringatan nonfatal underfull, Fontconfig, dan metadata JPEG tetap dicatat. Pemeriksaan visual penuh halaman tidak dilakukan; kompilasi bukan bukti seluruh tata letak sudah diperiksa.
- Detail metode verifikasi, keterbatasan sampling metadata baseline, dan sumber primer ada pada EVIDENCE_REVIEW.md.
- Backup sumber: `.local-backups/scientific-audit-20260922/` di laporan utama.

**Belum selesai secara empiris:** validasi pakar harmonisasi (F05), kalibrasi serta validasi ambang independen (F15), evaluasi generalisasi independen (F09), ketidakpastian dengan unit kelompok molekul (F16), kontrol/ablasi tambahan (F10–F12), dan seluruh pekerjaan mobile (F17). Perbaikan pelaporan F09/F10–F12/F16 tidak berarti eksperimen tambahan itu telah dilakukan. Detail split/averaging Tyagi tetap merupakan batas verifikasi literatur (F02).


## 10. Pembaruan dokumentasi dan diagram — 23 September 2026

Pembaruan ini memenuhi permintaan penyelarasan Project Context/Overview, tabel analisis, dan diagram delapan kandidat. Tidak ada bukti empiris baru atau perubahan hasil training.

| Bagian | Status terbaru | Bukti/cakupan |
|---|---|---|
| PROJECT_CONTEXT.md | Diperbaiki | V6 selesai; threshold OOF; ONNX baru/Android/deployment ditunda |
| PROJECT_OVERVIEW.md dan tabel | Diperbaiki | Delapan kandidat v6; 26 baris metrik/delta dicocokkan dengan hasil beku |
| Diagram lima sumber | Diperbaiki | Delapan cabang mandiri, empat baseline dan empat Optuna; muat satu halaman |
| Bab 3 | Diselaraskan | Caption dan narasi menyebut delapan kandidat serta eksekusi berurutan |
| Pemeriksaan visual F18 | Diperluas secara terbatas | PDF final 64 halaman berhasil; halaman PDF 30–32 diperiksa, termasuk diagram. Bukan inspeksi seluruh halaman |
| F17 ONNX/Android | Tetap ditunda | Tidak ada implementasi atau pengujian baru |
| Batas empiris F05/F09/F15/F16 dan kontrol tambahan | Tetap berlaku | Dokumentasi tidak menggantikan validasi ahli, kalibrasi, atau evaluasi independen |

Bukti revisi terbaru: `qa/docs_diagram_v6_20260923/content_verification.json` dan `render_verification.json` pada laporan utama. PDF: `output/pdf/SKRIPSI_V6_DELAPAN_CABANG_20260923.pdf`. Ringkasan perubahan lengkap ada di `CHANGELOG.md` laporan.

**Penting untuk audit berikutnya:** bukti QA 22 September adalah snapshot sebelum perubahan ini. Guard pada skrip lama yang mengharuskan diagram/Bab 3 identik dengan snapshot lama bukan pemeriksaan kompatibilitas untuk revisi baru. Jangan menimpa bukti lama atau menganggap perbedaan yang disengaja sebagai kerusakan; gunakan bukti bertanggal 23 September untuk berkas yang berubah. Dataset, model, split, dan angka v6 tetap sama.
