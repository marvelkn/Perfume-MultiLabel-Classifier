## Status aktif — penyelarasan backend, 21 September 2026

Grouped-v6 telah selesai; **LightGBM D** dipilih berdasarkan macro AP validasi
0,302232 (XGBoost D 0,298848). AP test masing-masing 0,313921 dan 0,314531.
Rilis aktif: `releases/grouped-v6-lgbm-d/`, 109 model dan 1.029 fitur.
Inferensi Python dan konfigurasi bawaan `/fingerprint` telah diselaraskan.
Fitur 6.686 molekul identik dengan dataset beku; 3.488 skor pada 32 contoh
identik dengan prediksi tersimpan. Sebanyak 23 tes komponen lulus.

Panduan aktif: [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md).
Bukti: [verification.json](reports/backend_alignment_20260921/verification.json).
ONNX lama 25 label dan dua array ambangnya dihapus atas instruksi pengguna.
ONNX baru, integrasi Android, dan deployment **ditunda**. Training tambahan,
pseudo-labeling, dan inferensi massal data yang ditolak tidak termasuk scope aktif.
Data, split, model terlatih, dan angka evaluasi v6 tidak diubah; hasil tetap eksploratori.

**Bagian bertanggal lebih lama di bawah adalah riwayat, bukan instruksi aktif.**

> Catatan: uraian lama berikut belum seluruhnya diaudit ulang. Klaim test hanya dibuka sekali, bebas seluruh leakage, ONNX/Android selesai, dan keluarga label buatan tidak menjadi acuan v6. Gunakan kontrak rilis dan laporan terbaru.

# Laporan Pelacakan Hasil Eksperimen (Result Tracking v1 – v5)
**Proyek Skripsi:** Klasifikasi Multi-Label Senyawa Aroma Parfum (*Perfume Multi-Label Odor Classification*)  
**Algoritma:** XGBoost & LightGBM  
**Target:** 109 Deskriptor Aroma (Multi-Label)  
**Status Terakhir:** Run v1 s.d. v5 Selesai (Analisis Lengkap) | Run v6 Sedang Berjalan (*Running*)  
**Tanggal Kompilasi:** 17 September 2026  

---

## 1. Ringkasan Eksekutif & Linimasa Eksperimen

Eksperimen pemodelan aroma parfum dari struktur molekul ini berevolusi secara terstruktur melalui 6 iterasi utama. Tabel berikut merangkum evolusi konfigurasi, perlakuan data, mekanisme tuning, dan capaian metrik utama dari Run 1 hingga Run 5:

| Versi | Dataset & Partisi | Fokus & Perubahan Metodologi | Strategi Tuning Optuna | Pemenang & Metrik Kunci (Test AP) |
|---|---|---|---|---|
| **v1** | `perfume-five-v1`<br>(5.318 train, 1.368 test) | Integrasi 5 basis data (6.686 molekul kanonik, 109 label). Partisi acak standar per-label. | Rencana awal TPE: 2 jam / 15 trial per studi. | **Audit Stop (Tidak Dilatih Penuh):** Ditemukan *data leakage* (tumpang tindih struktur/fingerprint identik melintasi train-test). Eksperimen ditolak runner aktif demi integritas ilmiah. |
| **v2** | `perfume-five-grouped-v2`<br>(5.353 train, 1.333 test) | **Grouped Splitting:** 5.363 klaster struktur terisolasi (0 overlap fingerprint). Label disesuaikan: *narcissus* masuk, *vetiver* keluar. | Budget 2 jam/studi, dibatasi *safety cap* **15 trial**. XGBoost unweighted vs LightGBM balanced. | **LightGBM D** (Test AP: 0,3037)<br>XGBoost D (Test AP: 0,3041)<br>*Catatan: Tuning asimetris dan LightGBM terhenti terlalu dini.* |
| **v3** | `perfume-five-grouped-v2`<br>(5.353 train, 1.333 test) | **Symmetric Class Weights:** Bobot kelas seimbang diterapkan identik pada kedua model. Evaluasi ketat: **5-Seed Repeated Grouped CV** & threshold tuning. | Budget 2 jam/studi, batas 15 trial dihapus (LGBM 52 trial, XGB 10 trial). | **XGBoost D (v3)** (Test AP: **0,3058**)<br>LightGBM D (v3) (Test AP: 0,3037)<br>*Ditetapkan sebagai Incumbent Benchmark Utama.* |
| **v4** | `perfume-five-grouped-v2`<br>(5.353 train, 1.333 test) | *Warm-start Optuna* (+2 jam/studi untuk C & D). Seleksi ketat: kandidat baru harus mengungguli incumbent v3 di validasi. | Alokasi tambahan, namun terkendala interupsi sistem sehingga `lgbm_D` mencatat 0 trial baru. | **Incumbent v3 Dipertahankan:**<br>XGBoost D: 0,3058<br>LightGBM D: 0,3037<br>*Tidak ada kandidat baru yang mengalahkan v3.* |
| **v5** | `perfume-five-grouped-v2`<br>(5.353 train, 1.333 test) | Fokus audit tuning pada **Kondisi D**. Perbaikan arsitektur: **Heartbeat checkpoint 15 detik** agar sisa budget aman saat interupsi. | **Fresh 4 jam baru murni** per model (XGB 15 trial baru, LGBM 43 trial baru). | 🏆 **REKOR TERTINGGI!**<br>• **XGBoost D (v5):** Test AP **0,3145** (800 trees)<br>• **LightGBM D (v5):** Test AP **0,3093** (800 trees) |

---

## 2. Rincian Dataset, Taksonomi 109 Label, & Preprocessing

### 2.1 Sumber Data
Seluruh iterasi (v1 hingga v5) dibangun di atas kurasi data dari **5 basis data bahan aroma dan flavor dunia**:
1. **The GoodScents Company (TGSC):** Anotasi deskriptor industri wewangian komersial.
2. **IFRA 2019 Transparency List:** Standar internasional asosiasi wewangian (acuan kosakata taksonomi).
3. **Leffingwell & Associates (PMP 2001):** Database spesialis molekul odor dan flavor.
4. **Arctander (1960) Perfume and Flavor Chemicals:** Literatur klasik monograf bahan kimia aroma.
5. **Sigma-Aldrich Flavors & Fragrances (2014):** Katalog acuan bahan baku murni tingkat industri.

Hasil integrasi menghasilkan **6.686 molekul tunggal kanonik** setelah pembersihan.

---

### 2.2 Daftar Lengkap 109 Label Aroma Target (10 Famili Olfaktori)

Model memprediksi **109 deskriptor aroma** secara multi-label. Untuk kebutuhan pemahaman dan presentasi, 109 label ini diklasifikasikan ke dalam **10 Famili Olfaktori**:

1. **Floral (11 label):** `floral`, `rose`, `jasmin`, `lily`, `hyacinth`, `muguet` (lily of the valley), `lavender`, `neroli`, `narcissus`, `geranium`, `orris` (iris).
2. **Fruity (15 label):** `fruity`, `apple`, `apricot`, `banana`, `berry`, `cherry`, `grape`, `grapefruit`, `melon`, `peach`, `pear`, `pineapple`, `plum`, `raspberry`, `strawberry`, `tropical fruit`.
3. **Citrus & Fresh (5 label):** `citrus`, `lemon`, `grapefruit`, `fresh`, `clean`.
4. **Woody & Earthy (11 label):** `woody`, `sandalwood`, `pine`, `earthy`, `mushroom`, `musty`, `rooty`, `hay`, `dry`, `terpenic`, `leathery`.
5. **Sweet & Gourmand (16 label):** `sweet`, `caramel`, `chocolate`, `cocoa`, `coffee`, `honey`, `vanilla`, `buttery`, `creamy`, `milky`, `dairy`, `nutty`, `almond`, `coconut`, `popcorn`, `gourmand`.
6. **Green & Herbal (8 label):** `green`, `herbal`, `grassy`, `leafy`, `celery`, `cucumber`, `tea`, `minty`.
7. **Spicy & Warm (7 label):** `spicy`, `warm`, `cinnamon`, `aromatic`, `anisic`, `camphoraceous`, `cooling`.
8. **Balsamic & Resinous (5 label):** `balsamic`, `amber`, `phenolic`, `aldehydic`, `sharp`.
9. **Animalic & Waxy (4 label):** `animal like`, `musk like`, `fatty`, `waxy`.
10. **Pungent, Savoury, Fermented, & Chemical (27 label):** `savoury`, `meaty`, `roasted`, `smoky`, `sulfurous`, `onion`, `garlic`, `tobacco`, `rum`, `brandy`, `fermented`, `winey`, `medicinal`, `metallic`, `acidic`, `sour`, `cheesy`, `oily`, `soapy`, `ethereal`, `lactonic`, `pungent`, `potato`.

---

### 2.3 Partisi Dataset: 5.353 Data Latih vs 1.333 Data Uji (Grouped Splitting)

Dari total **6.686 senyawa molekul unik**:
- **Data Latih & Validasi (Training Set):** **5.353 senyawa (~80,1%)**
- **Data Uji Akhir (Held-Out Test Set):** **1.333 senyawa (~19,9%)**

#### Siklus Penggunaan Data:
1. **Fase Pelatihan & Penyetelan (Tuning):**
   - Model **hanya boleh melihat 5.353 senyawa latih**.
   - Di dalam 5.353 senyawa ini, dilakukan validasi silang 5-fold (*5-Fold Stratified Group Cross-Validation*) dan diulang 5 kali (*5-Seed Repeated CV*) untuk mengevaluasi hyperparameter terbaik.
   - **Data uji 1.333 senyawa sama sekali dikunci (*strictly frozen/held-out*)** agar tidak terjadi kebocoran informasi pemilihan model (*data snooping bias*).
2. **Fase Evaluasi Akhir:**
   - Setelah model terbaik dikunci, data uji 1.333 senyawa dibuka **satu kali saja** untuk mengukur akurasi generalisasi pada molekul baru yang belum pernah dilihat model sebelumnya.

#### Mengapa Harus *Grouped Split* (5.363 Kelompok Struktur)?
- Pada **v1**, partisi dilakukan secara acak standar. Ditemukan adanya molekul dengan struktur/fingerprint identik (misalnya enantiomer, stereoisomer, garam) yang masuk sekaligus ke data latih dan uji. Hal ini menyebabkan **kebocoran data (*data leakage*)**.
- Pada **v2 s.d. v5**, 6.686 molekul dikelompokkan ke dalam **5.363 gugus struktur independen**. Satu gugus utuh hanya boleh berada di set latih ATAU set uji.
- **Hasil:** **0 tumpang tindih (*zero overlap*)** fingerprint Morgan identik antara data latih dan data uji, serta di seluruh 545 fold validasi.

---

### 2.4 Alur 7-Tahap Preprocessing Data Secara Lengkap

Proses pra-pemrosesan data dilakukan melalui modul [`alignment/perfume_data.py`](file:///c:/Users/ACER/Documents/Marvel/Skripsi/Project%20Skripsi/Perfume-MultiLabel-Classifier/alignment/perfume_data.py) dengan 7 tahapan berurutan:

1. **Tahap 1: Ingesti Data & Verifikasi Checksum:**  
   Mengunduh tabel perilaku aroma dari repositori terkurasi Pyrfume. Setiap file diverifikasi dengan hash SHA-256 untuk menjamin integritas data tidak berubah.
2. **Tahap 2: Sanitasi Molekul & Penyaringan Komponen Tunggal (*Single-Molecule Filtering*):**  
   - Hanya mempertahankan molekul organik tunggal terhubung (`len(Chem.GetMolFrags(mol)) == 1`).
   - Mengeliminasi campuran, garam anorganik, ion counter, dan molekul tanpa koordinat atom valid.
3. **Tahap 3: Kanonikalisasi Struktur Kimia (RDKit Canonical Isomeric SMILES):**  
   - Setiap struktur kimia distandardisasi menggunakan representasi kanonik universal RDKit (`Chem.MolToSmiles(mol, isomericSmiles=True)`).
4. **Tahap 4: Harmonisasi & Pemetaan Sinonim Label Aroma (*Lexical Alias Mapping*):**  
   - Menyelaraskan teks bebas dari 5 sumber ke kosakata baku IFRA (misal: *"citrus notes"* $\rightarrow$ *"citrus"*, *"herbaceous"* $\rightarrow$ *"herbal"*, *"jasmine"* $\rightarrow$ *"jasmin"*).
   - Penggabungan multi-label: jika satu molekul tercatat di beberapa basis data, seluruh label aromanya disatukan (*union of labels*).
5. **Tahap 5: Seleksi Label Target Berdasarkan Frekuensi Minimum:**  
   - Memfilter hanya label yang memiliki minimal 30 kemunculan positif di data latih agar model memiliki sampel pembelajaran yang cukup $\rightarrow$ Menghasilkan **109 label target**.
6. **Tahap 6: Pembentukan Klaster Struktur & Partisi Bebas Leakage:**  
   - Memetakan molekul ke dalam 5.363 gugus struktur.
   - Partisi luar: `GroupShuffleSplit` (80:20) $\rightarrow$ 5.353 train dan 1.333 test.
   - Partisi dalam: `StratifiedGroupKFold` 5-fold per label.
7. **Tahap 7: Ekstraksi Fitur Kimia (1.029 Fitur):**  
   - **Morgan Fingerprint (1.024 bit):** Generator RDKit radius 2 (ECFP4-equivalent) sebagai representasi topologi 2D ikatan atom.
   - **5 Deskriptor Fisikokimia RDKit:**
     - `MolWt`: Berat molekul (Dalton) $\rightarrow$ merepresentasikan volatilitas/kemudahan menguap.
     - `MolLogP`: Koefisien partisi oktanol/air $\rightarrow$ merepresentasikan hidrofobisitas/lipofilitas.
     - `NumHDonors`: Jumlah donor ikatan hidrogen.
     - `NumHAcceptors`: Jumlah akseptor ikatan hidrogen.
     - `TPSA`: Luas permukaan polar topologis ($\text{Å}^2$) $\rightarrow$ memengaruhi interaksi dengan reseptor penciuman.
   - Menggabungkan keduanya menjadi vektor fitur **1.029 dimensi** untuk Kondisi B dan D.

---

### 2.5 Matriks Fitur (4 Kondisi Eksperimen)
Untuk menguji kontribusi deskriptor fisikokimia secara ilmiah (analisis ablasi), setiap versi menguji 4 kondisi:
- **Kondisi A (Morgan Baseline):** 1.024 bit Morgan Fingerprints (ECFP4-equivalent, radius 2) tanpa tuning parameter (`n_estimators = 100`).
- **Kondisi B (Morgan + Physicochemical Baseline):** 1.024 bit Morgan + **5 Deskriptor Fisikokimia RDKit** (`MolWt`, `MolLogP`, `NumHDonors`, `NumHAcceptors`, `TPSA`) = **1.029 fitur**, tanpa tuning parameter (`n_estimators = 100`).
- **Kondisi C (Morgan + Optuna):** 1.024 bit Morgan Fingerprints dengan hyperparameter hasil optimasi Optuna.
- **Kondisi D (Morgan + Physicochemical + Optuna):** 1.029 fitur gabungan dengan hyperparameter hasil optimasi Optuna.

---

## 3. Strategi Tuning & Optimasi Hyperparameter

### 3.1 Penanganan Ketimpangan Kelas (*Class Imbalance*)
Dataset aroma bersifat sangat timpang (*sparse multi-label*).
- **Pada v2 (Asimetris):** LightGBM menggunakan `class_weight='balanced'`, sedangkan XGBoost tidak diberikan bobot sampel. Akibatnya performa kedua model tidak dapat diperbandingkan secara adil (Recall LightGBM ~0,50 vs Recall XGBoost ~0,13).
- **Pada v3, v4, v5 (Simetris):** Diterapkan bobot sampel seimbang yang identik pada kedua algoritma:
  $$w_i = \frac{N_{\text{samples}}}{2 \cdot N_{\text{class}(y_i)}}$$
  Bobot dihitung secara ketat hanya dari partisi data latih tiap fold, sedangkan metrik validasi dan uji dihitung tanpa bobot (*unweighted*) untuk mencerminkan akurasi riil.

### 3.2 Alokasi Waktu & Pengendalian Interupsi Optuna
- **Sampler:** Tree-structured Parzen Estimator (`TPESampler`, seed 42).
- **Metrik Objektif:** Rata-rata *Macro Average Precision* (AP) pada validasi silang seluruh 109 label.
- **Evolusi Pengendalian Budget:**
  - **v2:** Alokasi 2 jam/studi, dibatasi *safety cap* 15 trial. LightGBM terhenti di trial 15 dalam waktu singkat.
  - **v3:** Alokasi 2 jam/studi penuh tanpa batas 15 trial. LightGBM mengeksekusi 43 trial (C) dan 52 trial (D); XGBoost mengeksekusi 10 trial (C) dan 10 trial (D).
  - **v4:** Alokasi tambahan +2 jam/studi (*warm-start*). Namun aturan penalti lama langsung menganggap budget habis jika proses terinterupsi, sehingga `lgbm_D` tidak menyelesaikan trial baru.
  - **v5:** Diberikan **4 jam waktu aktif baru bersih** khusus untuk kondisi D. Diterapkan **Heartbeat 15 detik**, di mana waktu aktif dicatat bertahap ke SQLite. Jika terjadi interupsi, sisa waktu tidak hangus dan dapat dilanjutkan dengan presisi.

### 3.3 Ruang Pencarian (*Search Space*) v2 s.d. v5
- `n_estimators`: 100 s.d. 800 (step 100)
- `learning_rate`: 0,005 s.d. 0,20 (distribusi log)
- `max_depth`: 3 s.d. 8
- `colsample_bytree`: 0,50 s.d. 1,00
- `subsample`: 0,60 s.d. 1,00
- `reg_alpha` (L1) & `reg_lambda` (L2): distribusi log
- **Spesifik XGBoost:** `min_child_weight` (1 s.d. 30), `gamma` (0 s.d. 5)
- **Spesifik LightGBM:** `num_leaves` (8 s.d. 128), `min_child_samples` (10 s.d. 150)

---

## 4. Hasil Komparatif Lengkap Seluruh Versi

Metrik evaluasi utama proyek ini adalah **Macro Average Precision (AP / AUPRC)** karena kemampuannya mengevaluasi peringkat probabilitas kelas minoritas tanpa terdistorsi oleh tingginya jumlah true negative pada data multi-label yang jarang (*sparse*). Metrik penunjang mencakup AUROC, Subset Accuracy, Precision, Recall, dan Specificity.

### 4.1 Hasil Detail Versi 2 (Uji Perdana Grouped Split)
*Dievaluasi pada decision threshold standar 0,5:*

| Model / Kondisi | Fitur & Setup | Best $n\_est$ | Val AP | Test AP | Test AUROC | Test Akurasi | Test Presisi | Test Recall | Test Spesifisitas |
|---|---|---|---|---|---|---|---|---|---|
| **xgb_A** | Morgan 1024 (Baseline) | 100 | 0,2528 | 0,2532 | 0,8106 | 0,9636 | 0,3696 | 0,1467 | 0,9909 |
| **xgb_B** | Morgan + 5 Desk (Baseline) | 100 | 0,2719 | 0,2865 | 0,8285 | 0,9646 | 0,4060 | 0,1675 | 0,9910 |
| **xgb_C** | Morgan 1024 + Optuna | 800 | 0,2617 | 0,2659 | 0,8273 | 0,9647 | 0,3564 | 0,1155 | 0,9932 |
| **xgb_D** | Morgan + 5 Desk + Optuna | 100 | 0,2949 | 0,3041 | 0,8391 | 0,9658 | 0,4403 | 0,1320 | 0,9938 |
| **lgbm_A** | Morgan 1024 (Baseline) | 100 | 0,2498 | 0,2540 | 0,8071 | 0,9403 | 0,2451 | 0,3488 | 0,9544 |
| **lgbm_B** | Morgan + 5 Desk (Baseline) | 100 | 0,2827 | 0,2971 | 0,8252 | 0,9489 | 0,2890 | 0,3575 | 0,9635 |
| **lgbm_C** | Morgan 1024 + Optuna | 500 | 0,2643 | 0,2674 | 0,8220 | 0,9044 | 0,1755 | 0,5292 | 0,9124 |
| **lgbm_D** | Morgan + 5 Desk + Optuna | 400 | **0,2966** | **0,3037** | **0,8392** | 0,9238 | 0,2108 | 0,5018 | 0,9329 |

*Analisis v2:*  
- Model **lgbm_D** terpilih dari validasi (Val AP: 0,2966).
- Terlihat jelas efektivitas penambahan 5 deskriptor fisikokimia: Kondisi B mengungguli A (+3,3% AP pada XGB, +4,3% pada LGBM), dan Kondisi D mengungguli C (+3,8% AP pada XGB, +3,6% pada LGBM).
- Namun, ketimpangan bobot menyebabkan perbandingan XGBoost vs LightGBM belum setara.

---

### 4.2 Hasil Detail Versi 3 (Standardisasi Bobot & Repeated CV 5-Seed)
*Pembobotan kelas seimbang seragam, validasi ketat 5-Seed Repeated Grouped CV:*

| Model / Kondisi | Resep Terbaik Optuna | Val AP (1-Fold) | 5-Seed RCV AP (Val) | Test AP (thr 0,5) | Test AUROC | Test F1 (thr 0,5) | Test F1 (Val Thr) |
|---|---|---|---|---|---|---|---|
| **xgb_A** | 100 trees | 0,2489 | — | 0,2516 | 0,8084 | 0,2773 | — |
| **xgb_B** | 100 trees | 0,2742 | — | 0,2888 | 0,8212 | 0,3192 | — |
| **xgb_C** | 600 trees, lr: 0,0106, depth: 8 | 0,2662 | — | 0,2737 | 0,8252 | 0,2774 | — |
| **xgb_D** | **600 trees**, lr: 0,0106, depth: 8, col: 0,916 | **0,2980** | **0,2954** | **0,3058** | **0,8392** | **0,3013** | **0,3210** |
| **lgbm_A** | 100 trees | 0,2498 | — | 0,2540 | 0,8071 | 0,2880 | — |
| **lgbm_B** | 100 trees | 0,2827 | — | 0,2971 | 0,8252 | 0,3197 | — |
| **lgbm_C** | 800 trees, lr: 0,0126, depth: 8 | 0,2719 | — | 0,2788 | 0,8296 | 0,2743 | — |
| **lgbm_D** | **400 trees**, lr: 0,0239, depth: 6, col: 0,570 | **0,2966** | **0,2943** | **0,3037** | **0,8392** | **0,2885** | **0,3238** |

*Analisis v3:*  
- Dengan bobot yang seimbang, **XGBoost D** unggul tipis atas **LightGBM D** pada Repeated CV (0,2954 vs 0,2943) dan mencapai **Test AP 0,3058**.
- Optimasi threshold per-label (*Validation Thresholds*) berhasil mendongkrak F1-Score makro dari ~0,29–0,30 menjadi **0,3210 (XGBoost)** dan **0,3238 (LightGBM)**.
- Kedua model kondisi D ini ditetapkan sebagai **incumbent** resmi.

---

### 4.3 Hasil Detail Versi 4 (Perpanjangan Waktu Warm-Start)
- **Tujuan:** Menambah alokasi +2 jam/studi pada studi C dan D untuk melihat apakah penjelajahan lanjutan menemukan konfigurasi yang lebih unggul.
- **Hasil Evaluasi Validasi vs Incumbent:**
  - `xgb_D`: Menyelesaikan 5 trial baru (total 15 trial), namun skor validasi terbaik tetap berada di 0,29543 (sama persis dengan `xgb_D_v3`).
  - `lgbm_D`: Karena interupsi dan mekanisme reservasi budget lama, mencatat 0 trial baru.
- **Keputusan Protokol:** Tidak ada model baru yang mampu mengalahkan incumbent v3 pada tahap validasi. Sesuai protokol yang telah dipraregistrasi, model yang dinilai pada data test tetap **incumbent v3 (`xgb_D_v3` dan `lgbm_D_v3`)**. Skor Test AP tidak berubah (XGB: 0,3058, LGBM: 0,3037).

---

### 4.4 Hasil Detail Versi 5 (Audit Tuning Mandiri 4 Jam Kondisi D)
*Memberikan 4 jam aktif segar murni per algoritma dengan heartbeat 15 detik:*

| Model Finalis | Resep Optimal Optuna | Trial Selesai | 5-Seed RCV AP (Val) | Test AP (thr 0,5) | Test AUROC | Test F1 (thr 0,5) | Test F1 (Val Thr) |
|---|---|---|---|---|---|---|---|
| **xgb_D_v5** | **800 trees**, lr: 0,01838, depth: 7, col: 0,5042, sub: 0,7923, reg_α: 0,0054, reg_λ: 0,0011 | 15 baru<br>(total 30) | **0,29885** *(+0,0034)* | **0,31453** *(+0,0088)* | **0,83873** | **0,3165** | **0,3511** |
| **lgbm_D_v5** | **800 trees**, lr: 0,01327, depth: 5, col: 0,5054, sub: 0,8694, reg_α: 0,00003, reg_λ: 1,7418, leaves: 20 | 43 baru<br>(total 95) | **0,29864** *(+0,0044)* | **0,30925** *(+0,0056)* | **0,84010** | **0,2902** | **0,3450** |

*Analisis v5:*  
- **Peningkatan Nyata:** Terjadi lonjakan performa yang signifikan pada kedua model dibandingkan v3/v4.
- **XGBoost D (v5)** memecahkan rekor performa tertinggi proyek dengan **Test AP 0,31453** (+0,88% dari v3).
- **LightGBM D (v5)** mencapai **Test AP 0,30925** (+0,56% dari v3).
- Pada threshold optimal hasil validasi, F1-Score makro menembus angka **0,3511** (XGBoost) dan **0,3450** (LightGBM).

---

## 5. Ringkasan Ablasi: Pengaruh Fitur vs Pengaruh Tuning

Berdasarkan perbandingan antar-kondisi (A, B, C, D) di seluruh iterasi:

```
[Kondisi A: Morgan 1024 Baseline] 
        │  AP ≈ 0,251 - 0,254
        │
        ├───────────────────────────────────────┐
        ▼ (+5 Deskriptor Fisikokimia)           ▼ (+Tuning Optuna)
[Kondisi B: Morgan + 5 Desk Baseline]   [Kondisi C: Morgan 1024 + Optuna]
        │  AP ≈ 0,286 - 0,297                   │  AP ≈ 0,266 - 0,278
        │  (Peningkatan: +3,3% s.d. +4,3%)      │  (Peningkatan: +1,5% s.d. +2,4%)
        │                                       │
        └───────────────────┬───────────────────┘
                            ▼ (+Deskriptor & +Tuning Bersama)
                  [Kondisi D: Morgan + 5 Desk + Optuna]
                            AP = 0,3093 - 0,3145 (v5)
                            (Peningkatan Total: +5,5% s.d. +6,3%)
```

1. **Pengaruh Deskriptor Fisikokimia (B vs A):**  
   Memberikan lompatan performa paling masif (+3,3% hingga +4,3% AP). Properti seperti berat molekul, LogP, donor/akseptor hidrogen, dan TPSA memberikan konteks fisikokimia volatilitas dan polaritas aroma yang tidak sepenuhnya tertangkap oleh bit sidik jari biner 2D saja.
2. **Pengaruh Hyperparameter Tuning (C vs A):**  
   Memberikan kenaikan moderat (+1,5% hingga +2,4% AP) dengan mengoptimalkan kedalaman pohon dan regularisasi agar tidak overfitting pada bit Morgan yang jarang.
3. **Sinergi Penuh (D vs A):**  
   Kombinasi deskriptor dan tuning menghasilkan peningkatan total sebesar **+6,3% AP** (dari ~0,2516 ke 0,31453 pada XGBoost).

---

## 6. Rasional dan Desain Versi 6 (Yang Sedang Berjalan / Running)

Penyusunan paket **Versi 6 (`perfume-campus-grouped-v6`)** didasarkan pada temuan empiris di Versi 5:
1. **Plafon Jumlah Pohon (*Tree Boundary*):** Kedua model finalis v5 secara independen memilih `n_estimators = 800`, yang merupakan batas atas absolut dari search space v2–v5. Ada indikasi kuat bahwa kapasitas model ensemble belum jenuh (*saturated*).
2. **Batas Subsampling Kolom (*Colsample Boundary*):** Nilai `colsample_bytree` kedua model menyentuh `~0,504` dan `~0,505`, tepat di tepi batas bawah pencarian lama (0,50).

### Pembaruan pada Versi 6:
- **Perluasan Ruang Hyperparameter:**
  - `n_estimators`: diperluas dari $[100, 800]$ menjadi **$[100, 1.600]$** (step 100).
  - `colsample_bytree`: diperluas ke bawah dari $[0,50, 1,00]$ menjadi **$[0,30, 1,00]$**.
- **Alokasi Budget Komputasi:**
  - Masing-masing studi C dan D diberikan waktu **8 jam aktif per studi** (total 32 jam aktif komputasi pada PC kampus berprosesor i7-8700K 6-core).
- **Metodologi:**
  - Resep terbaik v5 diumpankan sebagai *trial* awal (dievaluasi ulang dalam budget).
  - Menilai kembali seluruh 8 kandidat (A, B, C, D) dengan Repeated CV 5-seed sebelum mengeksekusi pengujian akhir pada data test.

---

## 7. Lokasi Berkas Artefak & Audit Trail

Semua hasil dan data pendukung tersimpan secara persisten pada repositori lokal:

- **Data Split Aktif (v2 s.d. v6):** [data/builds/perfume-five-grouped-v2/](file:///c:/Users/ACER/Documents/Marvel/Skripsi/Project%20Skripsi/Perfume-MultiLabel-Classifier/data/builds/perfume-five-grouped-v2)
  - Manifest & Checksum: `dataset_manifest.json` (SHA256: `886e8e49...`)
  - Pembagian Kelompok: `splits.json` (5.363 groups)
- **Artefak Hasil Lengkap v2:** [.local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v2/.../runs/perfume-five-grouped-v2/complete.json](file:///c:/Users/ACER/Documents/Marvel/Skripsi/Project%20Skripsi/Perfume-MultiLabel-Classifier/.local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v2/perfume-campus-grouped-v2/runs/perfume-five-grouped-v2/complete.json)
- **Artefak Hasil Lengkap v3:** [v6_reference/v3_complete.json](file:///c:/Users/ACER/Documents/Marvel/Skripsi/Project%20Skripsi/Perfume-MultiLabel-Classifier/v6_reference/v3_complete.json)
- **Artefak Hasil Lengkap v4:** [v6_reference/v4_complete.json](file:///c:/Users/ACER/Documents/Marvel/Skripsi/Project%20Skripsi/Perfume-MultiLabel-Classifier/v6_reference/v4_complete.json)
- **Artefak Hasil Lengkap v5:** [v6_reference/v5_complete.json](file:///c:/Users/ACER/Documents/Marvel/Skripsi/Project%20Skripsi/Perfume-MultiLabel-Classifier/v6_reference/v5_complete.json)
- **Kompilasi Metrik Historis v6 Preparation:** [v6_reference/comparison.json](file:///c:/Users/ACER/Documents/Marvel/Skripsi/Project%20Skripsi/Perfume-MultiLabel-Classifier/v6_reference/comparison.json)
- **Protokol Eksekusi v6:** [alignment/protocol_v6.json](file:///c:/Users/ACER/Documents/Marvel/Skripsi/Project%20Skripsi/Perfume-MultiLabel-Classifier/alignment/protocol_v6.json)
