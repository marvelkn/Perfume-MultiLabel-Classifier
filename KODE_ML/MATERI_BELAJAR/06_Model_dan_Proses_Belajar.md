# 6 Model dan proses belajar

## Yang perlu kamu pahami

Targetnya: paham apa yang dipelajari model dan bedanya classifier dengan pohon.

**Model tidak membaca tulisan SMILES langsung.** RDKit lebih dulu mengubah molekul menjadi angka. Untuk skenario B dan D, satu molekul menjadi **2.056 angka**: Morgan 2.048 bit ditambah delapan deskriptor. Angka inilah yang masuk ke model.

## Satu label punya satu model sendiri

Bayangkan tiga aroma: sweet, woody, floral. Target `[1, 0, 1]` berarti sweet dan floral tercatat, woody tidak. Setiap model belajar dari kolom aromanya sendiri.

Dalam project ini ada **109 label**. Jadi satu kandidat mempunyai **109 classifier biner**. Biner berarti jawabannya dua kelas: 0 atau 1. Setiap classifier berisi banyak pohon. Jadi **109 model bukan 109 pohon**.

| Yang dihitung | Jumlah dan artinya |
|---|---|
| Label per kandidat | 109 aroma yang diprediksi sendiri-sendiri |
| Kandidat eksperimen | 2 algoritma × 4 skenario = 8 kandidat |
| Model final tersimpan | 8 × 109 = **872 classifier** |
| Model yang dipakai prediksi utama | 109 classifier milik kandidat terpilih |

Angka 872 hanya model final. Model sementara saat tuning dan validasi jauh lebih banyak. Prediksi normal tidak menjalankan seluruh delapan kandidat sekaligus. Hasil utama memilih **lgbm_D**.

## Mulai baca kode dari sini

Sumber aktif: `KODE_ML/research/training.py`. Buka juga notebook `05_Model_dan_Bobot.ipynb` sebagai teman membaca. Library **XGBoost** menyediakan `XGBClassifier`; **LightGBM** menyediakan `LGBMClassifier`. Kita mengatur dan memanggilnya, sedangkan pencarian percabangan pohon dilakukan di dalam library.

Potongan asli `make_model`, baris 9–18:

```python
def make_model(algorithm, parameters):
    params = dict(parameters)
    profile = compute_profile()
    threads = profile["threads_per_model"]
    if algorithm == "xgb":
        return XGBClassifier(n_jobs=threads, random_state=42, eval_metric="logloss",
                             tree_method=profile["xgboost_tree_method"], **params)
    if algorithm == "lgbm":
        return LGBMClassifier(n_jobs=threads, random_state=42, verbosity=-1, **params)
    raise ValueError("Unknown algorithm.")
```

Urutannya: salin pengaturan, baca profil komputasi, ambil jumlah thread, lalu buat jenis classifier yang diminta. **Thread** adalah pekerja komputasi dalam satu model. Profil saat ini memakai CPU dan enam thread per model; model dijalankan bergantian. `random_state=42` menjaga pengacakan model tetap terkendali. `**params` memasukkan pengaturan seperti jumlah pohon dan learning rate. Fungsi ini baru membuat objek; model **belum belajar**.

Kemudian baca `fit`, baris 33–53. Pemanggilan `model.fit(...)` di baris 45 atau 51 adalah saat model belajar memakai fitur, target, dan bobot data.

Terakhir lihat `fit_final_models`, baris 55–66. Bagian pentingnya, baris 62–66:

```python
        model = fit(algorithm, params, x, y[:, j], Limits())
        filename = f"label_{j:03d}.joblib"
        joblib.dump(model, folder / filename)
        saved[name] = {"file": filename, "sha256": digest(folder / filename)}
        write(folder / "models.json", saved)
```

`y[:, j]` mengambil kolom aroma ke-`j`. **joblib** menyimpan classifier. `sha256` memeriksa perubahan isi file. `models.json` menghubungkan aroma dengan file modelnya.

**Input final fit:** `x` berukuran `(5353, 2048)` untuk A/C atau `(5353, 2056)` untuk B/D; `y` berukuran `(5353, 109)`. Satu pemanggilan `fit` hanya menerima satu target `(5353,)`. **Output:** satu classifier per label yang disimpan sebagai `label_000.joblib`, dan seterusnya.

## Pohonnya belajar apa

Pohon mempelajari aturan seperti “apakah bit ini aktif?” atau “apakah MolLogP melebihi batas tertentu?”. Batasnya dicari saat training, bukan aturan aroma yang kita tulis manual.

**Boosting** menambah pohon secara bertahap. Pohon berikutnya memperbaiki kekurangan gabungan pohon sebelumnya. Skor pohon dijumlahkan, lalu diubah menjadi skor kelas positif. Konsep penjumlahan pohon ini dijelaskan dalam [panduan resmi XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).

Untuk konfigurasi kita, XGBoost memakai `hist` dan kebijakan pertumbuhan default `depthwise`: percabangan lebih dekat akar didahulukan. LightGBM tumbuh `leaf-wise`: daun dengan potensi perbaikan terbesar didahulukan. Keduanya memakai histogram pada pengaturan ini. Jadi jangan bilang hanya LightGBM yang memakai histogram. Sumber: [parameter XGBoost](https://xgboost.readthedocs.io/en/release_2.1.0/parameter.html) dan [fitur LightGBM 4.6.0](https://lightgbm.readthedocs.io/en/v4.6.0/Features.html).

## Hitungan kecil untuk membayangkan boosting

**Ini ilustrasi buatan, bukan hasil eksperimen atau parameter project.** Anggap untuk label sweet skor awal `z = 0`. Skor `z` belum berbentuk peluang. Ubah dengan:

`p = 1 / (1 + exp(-z))`

`p` adalah skor positif, sedangkan `exp` adalah fungsi eksponensial. Saat `z = 0`, `p = 0,5`. Misalkan pohon baru memberi koreksi `+2` dan learning rate `0,5`. Learning rate mengatur seberapa besar koreksi dipakai.

`z_baru = z_lama + learning_rate × koreksi = 0 + 0,5 × 2 = 1`

`p_baru = 1 / (1 + exp(-1)) ≈ 0,731`

Skor sweet naik menjadi 0,731. Koreksi `+2` diberikan untuk contoh; aslinya dipelajari library. Skor pohon tidak dirata-ratakan sebagai persentase.

## Kenapa ada bobot kelas

Lihat `balanced_sample_weights`, baris 21–30. Rumus yang **benar-benar dipakai kode** adalah `bobot_kelas = N / (2 × jumlah_kelas)`. `N` berarti jumlah baris pada bagian latih saat itu.

Contoh buatan: sepuluh baris, delapan negatif dan dua positif. Bobot negatif `10/(2×8) = 0,625`; positif `10/(2×2) = 2,5`. Total bobot kedua kelas sama-sama lima. Ini membantu kelas positif yang jarang tercatat ikut diperhatikan. Baris tidak digandakan dan label tidak diubah. Saat CV, bobot dihitung dari **train fold**, bukan validation atau test.

Batasnya: bobot tidak menambah pengetahuan baru. Label 0 juga berarti tidak tercatat positif dalam dataset; ini belum tentu pembuktian bahwa manusia tidak akan mencium aroma tersebut.

## Bekal sidang

**“Apakah satu pohon untuk satu aroma?”** “Satu aroma punya satu classifier. Setiap classifier berisi banyak pohon boosting. Seluruh 109 classifier memberi skor aromanya masing-masing.”

**“Kenapa memakai bobot kelas?”** “Sebagian aroma jarang muncul. Bobot dihitung hanya dari data latih agar kelas positif tidak kalah hanya karena jumlah negatif jauh lebih banyak.”

## Cek pemahaman

- [ ] Aku bisa membedakan satu kandidat, satu classifier, dan satu pohon.
- [ ] Aku bisa menunjukkan `fit` sebagai proses belajar.
- [ ] Aku bisa menghitung contoh bobot kelas.
- [ ] Aku tidak menyebut skor prediksi sebagai akurasi atau intensitas aroma.
