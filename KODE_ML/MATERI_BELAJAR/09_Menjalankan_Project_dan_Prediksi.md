# 9 Menjalankan project dan prediksi

## Yang perlu kamu pahami

Targetnya: membuka panduan dan menjelaskan prediksi. **Fit** berarti belajar; **predict** memakai pola yang sudah dipelajari. Prediksi tidak melatih 109 model dari awal.

## Buka dari tempat yang tepat

Buka folder utama repository:

`C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier`

Gunakan interpreter `.venv` project. Baca `KODE_ML/README.md`, lalu delapan notebook `01_...ipynb` sampai `08_Prediksi_Molekul.ipynb`. Setiap notebook dirancang mandiri dari **kernel baru**, yaitu sesi Python kosong. Audit 27 Oktober menemukan alat kernel notebook belum terpasang di environment ML lokal. Output tersimpan tetap bisa dibaca. Untuk menjalankan cell, ikuti pemasangan `KODE_ML/requirements_belajar.txt` pada README; panduan ini tidak memasang dependency.

Notebook memakai helper `belajar.tampilkan_sumber` untuk menampilkan fungsi sumber aktif, lokasi, nomor baris, dan sidik jari kodenya. Jadi ikuti sumber `KODE_ML/research`, bukan mengira seluruh cell merupakan salinan training mandiri. Notebook prediksi menjalankan inferensi ringan dari model yang sudah ada.

Kalau mau menjalankan sel, langkahnya dari PowerShell di root repository:

```powershell
.\.venv\Scripts\python.exe -m pip install -r KODE_ML\requirements_belajar.txt
```

Perintah ini petunjuk pemasangan untuk kamu nanti, bukan dijalankan saat panduan dibuat. Di VS Code buka notebook01, pilih Select Kernel -> Python Environments -> interpreter `.venv\Scripts\python.exe`. Pilih Restart Kernel, lalu Run All. Tahap berhasil jika sel selesai tanpa traceback; notebook01 menampilkan tabel sumber, notebook03 ukuran fitur/target yang sesuai, dan notebook07 menampilkan delapan kandidat. Jika tidak ada `.venv` atau artefak model/data hilang, berhenti dan periksa kebutuhan project; jangan menganggap cukup memasang library notebook untuk membangun ulang seluruh environment penelitian.

Sumber aktif dirapikan setelah eksperimen kampus; bukti kesetaraan fungsi dan artefak ada di `HASIL_VERIFIKASI.md`. Ini bukan file literal yang tetap sama sejak training kampus. Hasil historis dikunci terpisah dari run baru.

## Periksa input tanpa training

Command berikut ditulis untuk PowerShell dari root repository. Ini petunjuk; tidak dijalankan ketika menyusun panduan ini.

```powershell
.\.venv\Scripts\python.exe -B -m research.experiment --check
```

Lihat `KODE_ML/research/experiment.py`, fungsi `main`, baris 302–313. Tanpa `--train`, CLI juga masuk jalur pemeriksaan. Output sukses mencantumkan `MAIN_M2048_D8_INPUTS_VERIFIED` dan `training_started: False`. Pemeriksaan ini memastikan input sesuai protokol; ini bukan evaluasi akurasi baru.

Di notebook `07_Evaluasi_dan_Pemilihan.ipynb`, gate `JALANKAN_EKSPERIMEN_BARU = False` harus tetap False untuk belajar. Training baru memerlukan tugas dan izin terpisah, folder run baru, serta lolos `runtime.check_host` baris 49–55. Guard memblokir laptop asal yang dicatat protokol. Profil yang didukung adalah CPU. Jangan mengubah guard demi menjalankan cell belajar.

## Ikuti satu SMILES sampai jadi fitur

Sumber: `KODE_ML/research/inference.py`, fungsi `predict`. Potongan asli baris 31–39:

```python
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Input must be one valid connected molecule.")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=spec["radius"], fpSize=spec["n_bits"], includeChirality=spec["include_chirality"])
    x = np.concatenate([generator.GetFingerprintAsNumPy(mol),
                        [getattr(Descriptors,name)(mol) for name in names]]).astype(np.float32)[None,:]
    if not np.isfinite(x).all() or x.shape[1] != spec["dimensions"]["input"]:
        raise ValueError("Feature schema mismatch.")
```

RDKit mengubah teks menjadi objek molekul. Input harus satu molekul terhubung yang valid. Pengaturan Morgan dan urutan deskriptor dibaca dari bundle model. NumPy menggabungkan nilainya. `[None, :]` membuat bentuk satu baris: **`(1, 2056)`** untuk kandidat utama lgbm_D. Ini urutan fitur yang sama dengan saat training. SMILES valid dan terhubung belum membuktikan molekul sesuai domain bahan pewangi. Audit 27 Oktober mencatat batas input mode nama dan SMILES belum konsisten; struktur valid saja bukan jaminan prediksi dapat dipercaya.

Sebelumnya, baris 18–30 memeriksa bundle, versi library, dan spesifikasi fitur. Bila gagal, cocokkan environment dan artefaknya; jangan langsung mematikan pemeriksaan.

## Bagaimana 109 skor keluar

Potongan asli baris 53–58:

```python
            score = float(model.predict_proba(x)[0,1])
        output.append({"label":label, "score":score, "predicted":score>=bundle["primary_threshold"]})
    return {"smiles":Chem.MolToSmiles(mol), "candidate":bundle["selected_candidate"],
            "descriptors":len(names), "threshold":bundle["primary_threshold"],
            "score_meaning":"uncalibrated model score, not aroma intensity",
            "labels":sorted(output,key=lambda row:row["score"],reverse=True)}
```

Bagian ini berada dalam loop semua label. Sebelumnya `joblib.load` membuka model label dan memeriksa sidik jari serta jumlah fitur. `[0, 1]` mengambil skor kelas positif untuk baris molekul pertama. `predicted` bernilai True jika skor minimal 0,5. Daftar akhirnya diurutkan dari skor terbesar.

Contoh **ilustrasi**: sweet 0,82, woody 0,17, floral 0,61. Keputusannya sweet dan floral positif. Boleh menampilkan 0,82 sebagai “skor 82%”, tetapi **bukan akurasi 82%**. Skor belum dikalibrasi: belum dibuktikan bahwa seratus prediksi berskor 82% akan tepat 82 kali. Skor juga bukan intensitas aroma. Jumlah skor seluruh label tidak harus 100% karena beberapa aroma bisa positif bersamaan.

## Command prediksi dari model tersimpan

```powershell
.\.venv\Scripts\python.exe -B -m research.inference --smiles "COc1cc(C=O)ccc1O"
```

Ini contoh SMILES vanillin. CLI `main`, baris 61–67, memakai `results.main_run()` bila `--run` tidak diberikan. Fungsi tersebut menunjuk hasil utama yang dikunci. Bila hasil tidak ditemukan, program berhenti; tidak otomatis training. Perintah ini menghasilkan JSON berisi SMILES, kandidat, threshold, dan 109 skor.

Vanillin tercatat dalam dataset, jadi contoh ini bukan bukti kemampuan pada molekul baru. Evaluasi resmi memakai test terpisah.

## Hubungannya dengan aplikasi

Audit `reports/final_audit_20261027/README.md` membuktikan prototipe memakai API RDKit untuk fitur dan 109 ONNX di aplikasi. API aktif ialah `feature_api` di repo frontend, berbeda dari API lama repo ML. Uji desktop 109 model × 32 vektor menghasilkan galat maksimum `4,001013e-7`, dengan keputusan 0,5 sama. Demo Android fisik pernah dicatat, tetapi audit ini tidak menguji ulang HP. Jadi integrasi tersedia dan diuji terbatas; hosting produksi dan pengujian luas belum selesai.

## Bekal sidang

**“Apakah input baru membuat model belajar lagi?”** “Tidak. SMILES diubah menjadi fitur dengan skema training, kemudian 109 classifier tersimpan menghitung skor. Tidak ada pemanggilan fit.”

**“Kenapa versi RDKit diperiksa?”** “Agar pembentukan fitur konsisten dengan training. Model mengharapkan jumlah dan urutan fitur tertentu, jadi perubahan pembentukan fitur bisa mengubah hasil.”

## Cek pemahaman

- [ ] Aku tahu perbedaan check, fit, dan predict.
- [ ] Aku bisa menunjukkan perubahan teks SMILES menjadi `(1, 2056)`.
- [ ] Aku tahu lokasi pemilihan model utama dan threshold.
- [ ] Aku bisa membaca skor tanpa menyebutnya akurasi atau intensitas.
