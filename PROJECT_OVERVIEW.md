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

# RINGKASAN PROYEK SKRIPSI (PROJECT OVERVIEW & PRESENTATION REFERENCE)

**Judul Proyek:** Klasifikasi Multi-Label Karakter Aroma Senyawa Molekul Berbasis Struktur Kimia untuk Aplikasi Rekomendasi Parfum (*Perfume Multi-Label Odor Classification from Molecular Structure*)  
**Peneliti / Penulis:** Marvel Kevin Nathanael  
**Domain:** Kemoinformatika (*Chemoinformatics*), *Machine Learning*, Aplikasi Mobile (*Edge AI*)  
**Aplikasi Pendukung:** Essenza (Aplikasi Android Penjelajah dan Rekomendasi Parfum)  
**Tujuan Dokumen:** Dokumen referensi komprehensif untuk penyusunan slide presentasi (PPT / Pitch Deck / Seminar Proposal / Sidang Skripsi) menggunakan AI generator presentasi (seperti Gamma, ChatGPT, Claude, Tome, atau Canva).

---

## 1. Ringkasan Eksekutif (Executive Summary / Elevator Pitch)

- **Masalah:** Industri wewangian selama berabad-abad mengandalkan panel uji sensori manusia (*perfumer*) untuk mengevaluasi aroma suatu senyawa kimia. Proses ini sangat mahal, lambat, memiliki bias subjektivitas indra penciuman manusia, dan dibatasi oleh risiko toksisitas senyawa baru.
- **Tantangan Teknis:** Hubungan antara struktur kimia molekul dan persepsi aroma (*Structure-Odor Relationship* / SOR) sangat kompleks dan non-linear. Selain itu, satu molekul umumnya memiliki profil multi-aroma (misalnya suatu senyawa bisa beraroma *floral*, *citrus*, dan *sweet* sekaligus), menjadikannya persoalan **klasifikasi multi-label yang sangat timpang (*sparse multi-label classification*)**.
- **Solusi yang Dikembangkan:**
  1. Mengkurasi **6.686 molekul kanonik** dari **5 basis data aroma global** dengan **109 deskriptor aroma target**.
  2. Menerapkan partisi **Grouped Splitting (5.363 gugus struktur)** untuk **mengeliminasi 100% *data leakage*** (kebocoran kemiripan struktur molekul identik antara data latih dan uji).
  3. Membagi dataset secara objektif: **5.353 senyawa untuk pelatihan dan validasi (train & CV)** dan **1.333 senyawa terisolasi ketat untuk pengujian akhir (held-out test set)**.
  4. Membangun dan membandingkan dua algoritma ensemble mutakhir: **XGBoost** dan **LightGBM** dengan pendekatan *Binary Relevance* dan bobot sampel terstandarisasi.
  5. Menganalisis pengaruh representasi molekul: **Sidik jari Morgan 1.024-bit (ECFP4)** vs kombinasi **Morgan + 5 Deskriptor Fisikokimia RDKit (1.029 fitur)**.
  6. Mengoptimasi hyperparameter dengan **Optuna (TPE Sampler)** dan memvalidasi ketat dengan **5-Seed Repeated Grouped Cross-Validation**.
  7. Mengekspor model terbaik ke format **ONNX Runtime Mobile** dan mengintegrasikannya ke aplikasi Android **Essenza** untuk inferensi aroma lokal secara offline dan rekomendasi parfum.
- **Capaian Terbaik (v5):**
  - **XGBoost D:** Macro Average Precision (Test AP) **0,3145** (AUROC: **0,8387**, F1 adaptif: **0,3511**).
  - **LightGBM D:** Macro Average Precision (Test AP) **0,3093** (AUROC: **0,8401**, F1 adaptif: **0,3450**).
  - Penambahan 5 deskriptor fisikokimia terbukti secara konsisten meningkatkan akurasi rata-rata AP sebesar **+3,3% s.d. +4,3%**.

---

## 2. Latar Belakang & Rumusan Masalah

1. **Subjektivitas & Skalabilitas Pengujian Aroma:**
   Persepsi penciuman manusia bervariasi antar-individu karena faktor genetik reseptor olfaktori (*olfactory receptors*), kelelahan penciuman (*olfactory fatigue*), dan usia. Skrining aroma senyawa baru di laboratorium membutuhkan waktu berbulan-bulan.
2. **Kompleksitas Hubungan Struktur Kimia ke Aroma (SOR):**
   Dua molekul dengan kerangka struktur serupa terkadang memiliki aroma yang bertolak belakang, atau sebaliknya dua molekul dengan struktur berbeda dapat memicu aroma yang serupa. Diperlukan algoritma *machine learning* non-linear yang mampu menangkap interaksi fitur tingkat tinggi.
3. **Karakteristik Data Multi-Label & Ekstrem Timpang (*Class Imbalance*):**
   Dari ribuan molekul, sebuah label aroma tertentu (seperti *narcissus*, *camphoraceous*, atau *caramel*) mungkin hanya dimiliki oleh beberapa puluh molekul. Metode pemodelan konvensional cenderung gagal mendeteksi aroma minoritas.
4. **Kebutuhan Penerapan Praktis di Perangkat Pengguna (*Edge Deployment*):**
   Penggemar parfum dan formulator membutuhkan estimasi profil aroma molekul dan pencarian parfum langsung di smartphone tanpa selalu bergantung pada koneksi cloud yang lambat.

---

## 3. Daftar Lengkap 109 Label Aroma Target (Dikelompokkan Berdasarkan Famili Aroma)

Model memprediksi **109 deskriptor aroma** secara simultan. Untuk kebutuhan presentasi, 109 label ini diklasifikasikan ke dalam **10 Famili Olfaktori (*Odor Families*)**:

```
                                 ┌─ 1. Floral (11 label)
                                 ├─ 2. Fruity (15 label)
                                 ├─ 3. Citrus & Fresh (5 label)
                                 ├─ 4. Woody & Earthy (11 label)
                                 ├─ 5. Sweet & Gourmand (16 label)
  109 Deskriptor Aroma Target ───┼─ 6. Green & Herbal (8 label)
                                 ├─ 7. Spicy & Warm (7 label)
                                 ├─ 8. Balsamic & Resinous (5 label)
                                 ├─ 9. Animalic & Waxy (4 label)
                                 └─ 10. Pungent, Savoury, & Chemical (27 label)
```

### Rincian Anggota Tiap Famili:
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

## 4. Partisi Dataset: Mengapa 5.353 Train vs 1.333 Test?

Dari total **6.686 senyawa molekul unik**:
- **Data Latih & Validasi (Training Set):** **5.353 senyawa (~80,1%)**
- **Data Uji Akhir (Held-Out Test Set):** **1.333 senyawa (~19,9%)**

### Alur Siklus Penggunaan Data:
1. **Selama Proses Pengembangan & Tuning (Optuna):**
   - Model **hanya boleh melihat 5.353 senyawa latih**.
   - Di dalam 5.353 senyawa ini, dilakukan validasi silang 5-fold (*5-Fold Stratified Group Cross-Validation*) dan diulang 5 kali (*5-Seed Repeated CV*) untuk mengevaluasi hyperparameter terbaik.
   - **Data uji 1.333 senyawa sama sekali dikunci (*strictly frozen/held-out*)** agar tidak terjadi kebocoran informasi pemilihan model (*data snooping bias*).
2. **Setelah Model Selesai & Dikunci:**
   - Kandidat terbaik dilatih penuh pada seluruh 5.353 data latih.
   - Barulah data uji 1.333 senyawa dibuka **satu kali saja** untuk mengukur akurasi generalisasi di dunia nyata pada molekul baru.

### Mengapa Disebut *Grouped Splitting* (5.363 Kelompok)?
- Dalam kimia, satu zat bisa memiliki varian garam, stereoisomer, atau kembaran molekul yang memiliki sidik jari identik.
- Jika dibagi acak biasa (seperti pada v1), molekul kembar ini bisa masuk ke train dan test sekaligus, sehingga model tampak "pintar" padahal hanya menghafal molekul kembarannya (*data leakage*).
- Pada *Grouped Split*, 6.686 molekul dikelompokkan ke dalam **5.363 gugus struktur independen**. Satu gugus utuh dimasukkan ke train ATAU test.
- **Hasilnya: 0 molekul dengan fingerprint identik yang melintasi data train dan test.**

---

## 5. Rincian Alur Preprocessing Data (Langkah demi Langkah)

Proses pra-pemrosesan data dilakukan secara otomatis, terverifikasi checksum, dan terdokumentasi penuh melalui skrip [`alignment/perfume_data.py`](file:///c:/Users/ACER/Documents/Marvel/Skripsi/Project%20Skripsi/Perfume-MultiLabel-Classifier/alignment/perfume_data.py):

```
[Tahap 1: Ingesti Data 5 Sumber]
  GoodScents, IFRA 2019, Leffingwell, Arctander, Sigma-Aldrich (Verifikasi Hash SHA-256)
                       │
                       ▼
[Tahap 2: Sanitasi & Penyaringan Molekul Tunggal (Single-Molecule Filtering)]
  Buang senyawa tanpa ID, SMILES korup, dan multi-fragmen (garam, pelarut, campuran)
                       │
                       ▼
[Tahap 3: Kanonikalisasi Struktur Kimia]
  Standarisasi representasi atomik ke RDKit Canonical Isomeric SMILES (6.686 molekul unik)
                       │
                       ▼
[Tahap 4: Harmonisasi & Normalisasi Label Aroma]
  Pemetaan teks aroma bebas dari 5 sumber ke Taksonomi IFRA via kamus sinonim terdefinisi
                       │
                       ▼
[Tahap 5: Seleksi Label Target (Thresholding)]
  Filter label aroma dengan frekuensi latih ≥ 30 kemunculan ➔ Terpilih 109 Label
                       │
                       ▼
[Tahap 6: Grouped Splitting (Pencegahan Data Leakage)]
  5.363 klaster struktur ➔ 5.353 Latih (Train) & 1.333 Uji (Test)
                       │
                       ▼
[Tahap 7: Ekstraksi Fitur Molekuler]
  Morgan Fingerprint (1.024-bit) + 5 Deskriptor Fisikokimia RDKit ➔ Matriks 1.029 Fitur
```

### Penjelasan Detail 7 Tahap Preprocessing:

1. **Tahap 1: Pengambilan & Verifikasi Integritas Sumber Data:**
   Mengunduh tabel perilaku aroma dan identitas kimia dari repositori terkurasi Pyrfume. Setiap file diverifikasi dengan hash SHA-256 untuk menjamin data tidak berubah diam-diam (*reproducibility*).
2. **Tahap 2: Sanitasi Molekul & Eliminasi Campuran (*Single-Molecule Filtering*):**
   - Struktur kimia yang dimasukkan wajib berupa **molekul organik tunggal terhubung (*single connected component*)**.
   - Campuran zat, garam anorganik, dan pelarut dieliminasi menggunakan fungsi `len(Chem.GetMolFrags(mol)) == 1`. Senyawa tanpa SMILES valid langsung ditolak.
3. **Tahap 3: Kanonikalisasi SMILES (RDKit):**
   - Struktur kimia ditulis dalam format string SMILES (*Simplified Molecular Input Line Entry System*).
   - Karena satu molekul dapat ditulis dengan banyak variasi string SMILES, RDKit menstandarkannya menjadi satu representasi kanonik universal (`Chem.MolToSmiles(mol, isomericSmiles=True)`).
4. **Tahap 4: Harmonisasi Label & Pemetaan Sinonim (*Lexical Alias Mapping*):**
   - Setiap sumber data menggunakan istilah bahasa Inggris yang sedikit berbeda (contoh: GoodScents menulis *"citrus notes"*, Arctander menulis *"citrusy"*, IFRA menulis *"citrus"*).
   - Dibuat kamus pemetaan formal (`ALIASES`) yang memetakan variasi penulisan ke kosakata baku taksonomi IFRA tanpa menggunakan tebakan semantik kabur (*fuzzy matching*).
   - Penggabungan multi-label: satu molekul menggabungkan label aroma dari seluruh sumber yang memuat molekul tersebut.
5. **Tahap 5: Seleksi Label Target Berdasarkan Frekuensi:**
   - Dari 184 istilah taksonomi IFRA, dipilih label yang memiliki **minimal 30 kemunculan positif** di partisi data latih agar model memiliki sampel pembelajaran yang cukup.
   - Tahap ini menyaring tepat **109 label aroma**.
6. **Tahap 6: Pembentukan Gugus Struktur & Partisi Bebas Leakage:**
   - Menghitung kemiripan kerangka molekul menggunakan graf kesetaraan struktur dan sidik jari Morgan.
   - Menghasilkan 5.363 gugus molekul.
   - Menggunakan `GroupShuffleSplit` (outer 80:20) dan `StratifiedGroupKFold` (inner 5-fold).
7. **Tahap 7: Ekstraksi Fitur Kimia (1.029 Fitur):**
   - **Morgan Fingerprint (1.024 bit):** Menggunakan generator RDKit radius 2 (merepresentasikan lingkungan ikatan hingga 4 ikatan atom di sekitarnya, setara ECFP4) yang diproyeksikan ke dalam array biner 1.024 dimensi.
   - **5 Deskriptor Fisikokimia RDKit:**
     - `MolWt`: Berat molekul (satuan Dalton).
     - `MolLogP`: Koefisien partisi oktanol-air (tingkat kelarutan lemak vs air).
     - `NumHDonors`: Jumlah atom donor ikatan hidrogen (-OH, -NH).
     - `NumHAcceptors`: Jumlah atom akseptor ikatan hidrogen (-O-, -N=).
     - `TPSA`: Luas permukaan polar topologis ($\text{Å}^2$).

---

## 6. Desain 4 Kondisi Eksperimen (Studi Ablasi)

Untuk membuktikan apa yang sebenarnya meningkatkan akurasi model, dilakukan pemisahan menjadi 4 kondisi:

| Kondisi | Fitur yang Digunakan | Optimasi Hyperparameter | Tujuan Eksperimen |
|---|---|---|---|
| **Kondisi A** | Morgan 1.024-bit | Tanpa Tuning (Default 100 Trees) | Baseline murni representasi struktur 2D. |
| **Kondisi B** | Morgan 1.024 + 5 Deskriptor (1.029 fitur) | Tanpa Tuning (Default 100 Trees) | Mengukur dampak langsung 5 deskriptor fisikokimia tanpa campur tangan tuning. |
| **Kondisi C** | Morgan 1.024-bit | Dioptimasi Optuna (TPE Sampler) | Mengukur dampak tuning hyperparameter pada fitur Morgan saja. |
| **Kondisi D** | Morgan 1.024 + 5 Deskriptor (1.029 fitur) | Dioptimasi Optuna (TPE Sampler) | Mengukur sinergi maksimal antara fitur fisikokimia dan tuning hyperparameter. |

---

## 7. Hasil Eksperimen Utama & Analisis Temuan (v1 s.d. v5)

### 7.1 Hasil Uji Akhir pada Data Test (Held-Out Test Set 1.333 Molekul)

| Model & Versi | Kondisi | Resep Hyperparameter Kunci | Test AP (AUPRC) | Test AUROC | Test F1 (Threshold 0,5) | Test F1 (Adaptive Val Thr) |
|---|---|---|---|---|---|---|
| **XGBoost D (v5)** | D (Morgan + Desk + Optuna) | **800 trees**, lr: 0,0184, depth: 7, col: 0,504 | **0,3145** 🏆 | **0,8387** | **0,3165** | **0,3511** 🏆 |
| **LightGBM D (v5)** | D (Morgan + Desk + Optuna) | **800 trees**, lr: 0,0133, depth: 5, col: 0,505 | **0,3093** | **0,8401** | **0,2902** | **0,3450** |
| **XGBoost D (v3)** | D (Morgan + Desk + Optuna) | 600 trees, lr: 0,0106, depth: 8, col: 0,916 | 0,3058 | 0,8392 | 0,3013 | 0,3210 |
| **LightGBM D (v3)** | D (Morgan + Desk + Optuna) | 400 trees, lr: 0,0239, depth: 6, col: 0,570 | 0,3037 | 0,8392 | 0,2885 | 0,3238 |
| **XGBoost B (v3)** | B (Morgan + Desk Default) | 100 trees, parameter default | 0,2888 | 0,8212 | 0,3192 | — |
| **LightGBM B (v3)** | B (Morgan + Desk Default) | 100 trees, parameter default | 0,2971 | 0,8252 | 0,3197 | — |
| **XGBoost C (v3)** | C (Morgan Optuna) | 600 trees, lr: 0,0106, depth: 8 | 0,2737 | 0,8252 | 0,2774 | — |
| **LightGBM C (v3)** | C (Morgan Optuna) | 800 trees, lr: 0,0126, depth: 8 | 0,2788 | 0,8296 | 0,2743 | — |
| **XGBoost A (v3)** | A (Morgan Baseline Default) | 100 trees, parameter default | 0,2516 | 0,8084 | 0,2773 | — |
| **LightGBM A (v3)** | A (Morgan Baseline Default) | 100 trees, parameter default | 0,2540 | 0,8071 | 0,2880 | — |

### 7.2 Tiga Kesimpulan Utama Ablasi:
1. **Deskriptor Fisikokimia adalah Faktor Paling Berdampak:**  
   Menambahkan 5 deskriptor fisikokimia meningkatkan AP sebesar **+3,3% s.d. +4,3%** (B vs A). Karakteristik aroma terbukti sangat dipengaruhi oleh volatilitas dan interaksi membran mukosa hidung.
2. **Optuna Tuning Menyempurnakan Hasil:**  
   Tuning hyperparameter memberikan kenaikan tambahan sebesar **+1,5% s.d. +2,4% AP** (C vs A) dan mengantarkan performa puncak pada Kondisi D.
3. **Thresholding Adaptif Mendongkrak F1-Score:**  
   Penyesuaian threshold per label pada data validasi meningkatkan F1-Score makro dari **0,3165 menjadi 0,3511 (+10,9% relatif)**.

---

## 8. Arsitektur Sistem & Integrasi Mobile (Aplikasi Essenza)

Aplikasi **Essenza** mengimplementasikan hasil riset ke dalam produk nyata:
1. **Input Pengguna:** Pengguna memasukkan nama molekul (contoh: *"Vanillin"*, *"Linalool"*, *"Iso E Super"*) atau kode SMILES.
2. **Ekstraksi Fitur Instan:** RDKit mengekstraksi vektor 1.029 dimensi dalam hitungan milidetik.
3. **Inference On-Device (ONNX Runtime Mobile):** Bobot model berjalan offline langsung di prosesor Android smartphone tanpa memerlukan kuota internet.
4. **Visualisasi Radar Aroma & Rekomendasi Parfum:** Profil 109 aroma yang diprediksi dicocokkan dengan katalog wewangian komersial untuk menampilkan parfum di pasaran yang memiliki kesamaan profil aroma molekul tersebut.

---

## 9. Kerangka Slide Presentasi Siap Pakai (12 Slides Outline)

- **Slide 1: Judul & Identitas**
  - Judul: *Klasifikasi Multi-Label Karakter Aroma Senyawa Molekul Berbasis Struktur Kimia untuk Rekomendasi Parfum*
  - Subjudul: *Studi Perbandingan XGBoost dan LightGBM dengan Ekstraksi Deskriptor RDKit pada Aplikasi Mobile Essenza*
  - Pemrasaran: Marvel Kevin Nathanael
- **Slide 2: Latar Belakang & Masalah**
  - Keterbatasan uji sensori manual: mahal, lambat, subjektif, risiko toksisitas.
  - Visi: Digitalisasi penciuman (*digital olfaction*) berbantuan AI.
- **Slide 3: Tantangan Teknis Utama**
  - Hubungan struktur kimia ke aroma (*SOR*) sangat non-linear.
  - Sifat *multi-label* (satu molekul memiliki banyak aroma).
  - Ketimpangan kelas ekstrem (*class imbalance*).
- **Slide 4: Alur Metodologi Penelitian**
  - Diagram tahapan: 5 Sumber Data $\rightarrow$ Preprocessing $\rightarrow$ Grouped Split $\rightarrow$ Ekstraksi Fitur $\rightarrow$ Tuning Optuna $\rightarrow$ Validasi 5-Seed $\rightarrow$ ONNX Mobile $\rightarrow$ Essenza.
- **Slide 5: Dataset & Pencegahan Kebocoran Data (Grouped Splitting)**
  - 5 Sumber data global (TGSC, IFRA, Leffingwell, Arctander, Sigma-Aldrich).
  - 6.686 Molekul Kanonik, 109 Label Aroma Target.
  - Partisi: 5.353 Train vs 1.333 Test (5.363 klaster struktur, 0 overlap fingerprint).
- **Slide 6: 109 Deskriptor Aroma dalam 10 Famili Olfaktori**
  - Visualisasi famili aroma: Floral (11), Fruity (15), Citrus (5), Woody (11), Sweet (16), Green (8), Spicy (7), Balsamic (5), Animalic (4), Savoury & Chemical (27).
- **Slide 7: Rekayasa Fitur & Desain 4 Kondisi Eksperimen**
  - Morgan Fingerprint 1.024-bit + 5 Deskriptor RDKit (MolWt, LogP, HDonors, HAcceptors, TPSA).
  - Penjelasan 4 kondisi eksperimen (A, B, C, D) untuk pembuktian ablasi ilmiah.
- **Slide 8: Strategi Pemodelan & Tuning Optuna**
  - XGBoost vs LightGBM (Binary Relevance).
  - Pembobotan kelas seimbang seragam (*Symmetric Sample Weights*).
  - Tuning Bayesian Optuna TPE & 5-Seed Repeated Grouped CV.
- **Slide 9: Hasil Eksperimen Utama (Puncak di Run v5)**
  - Rekor v5: XGBoost D Test AP **0,3145** dan LightGBM D Test AP **0,3093**.
  - Threshold tuning adaptif mendongkrak F1-Score hingga **0,3511**.
- **Slide 10: Analisis Ablasi: Apa yang Paling Berpengaruh?**
  - Deskriptor fisikokimia memberikan lompatan akurasi terbesar (+3,3% s.d. +4,3% AP).
  - Tuning Optuna menyempurnakan performa (+1,5% s.d. +2,4% AP).
  - Sinergi fitur + tuning menghasilkan lonjakan total +6,3% AP.
- **Slide 11: Implementasi Aplikasi Mobile (Essenza)**
  - Arsitektur Edge AI: Model ONNX berjalan offline di smartphone.
  - Prediksi profil aroma molekul baru dan pencarian parfum komersial berbasis kemiripan aroma.
- **Slide 12: Kesimpulan & Sesi Tanya Jawab (Q&A)**
  - Keberhasilan memodelkan 109 aroma molekul tanpa data leakage.
  - Bukti krusialnya sifat fisikokimia molekul dalam persepsi aroma.
  - Integrasi sukses dari riset kemoinformatika ke aplikasi nyata.
  - Terima Kasih & Tanya Jawab.
