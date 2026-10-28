# 12 Library dan cara membaca kode

## Bedakan tiga hal dulu

**Library** adalah kumpulan alat siap pakai. **Import** membuka alat itu supaya bisa dipanggil kode. **Dependency** adalah paket yang disediakan environment; keberadaannya belum berarti seluruh jalur program memakai paket tersebut.

Daftar versi di bawah berasal dari `requirements_research.txt` dan `KODE_ML/requirements_belajar.txt`. Status terpasang diperiksa dari metadata `.venv` pada 27 Oktober 2026, tanpa mengimpor model atau menjalankan training. **Versi yang diminta file requirements tidak selalu sama dengan kondisi environment.**

## Library penelitian

Seluruh paket pada tabel ini terpasang dengan versi yang tercantum. Kolom terakhir membedakan yang diimpor langsung oleh kode aktif dari paket lain di environment.

| Paket dan versi | Nama saat import | Peran yang ditemukan dalam kode |
|---|---|---|
| NumPy 1.26.4 | `numpy as np` | Mengolah matriks fitur, target, indeks, skor, serta file NPZ |
| pandas 2.2.3 | `pandas as pd` | Membaca CSV, membersihkan tabel, dan menulis rangkuman hasil |
| scikit-learn 1.6.1 | `sklearn` | Metrik dan `StratifiedGroupKFold`; helper asal kohor juga memakai `GroupShuffleSplit` |
| RDKit 2026.3.4 | `rdkit` | Membaca struktur, kanonisasi, Morgan, dan delapan deskriptor |
| XGBoost 2.1.4 | `xgboost`, `XGBClassifier` | Classifier XGBoost dan callback pemeriksaan batas komputasi |
| LightGBM 4.6.0 | `LGBMClassifier` | Classifier LightGBM |
| Optuna 4.5.0 | `optuna` | Study, trial, pencarian hyperparameter, dan pembacaan hasil study |
| joblib 1.6.0 | `joblib` | Menyimpan/membaca model dan keadaan sampler |
| psutil 7.0.0 | `psutil` | Memeriksa proses, RAM, dan informasi CPU |
| SciPy 1.15.3 | Tidak diimpor langsung | Tersedia sebagai dukungan numerik/dependency; jangan mengaku memanggil fungsi SciPy pada sumber aktif |
| iterative-stratification 0.1.9 | Tidak diimpor langsung | Terdaftar di environment; split aktif memakai scikit-learn, bukan paket ini |
| PyYAML 6.0.2 | Tidak diimpor langsung | Terdaftar; konfigurasi jalur aktif dibaca sebagai JSON |
| requests 2.32.5 | Tidak diimpor langsung | Terdaftar; pembacaan snapshot lokal aktif bukan pengunduhan HTTP lewat requests |

“Tidak diimpor langsung” berlaku untuk sumber penelitian aktif dan helper delapan notebook yang diperiksa. Ini bukan klaim bahwa paket tersebut tidak pernah dipakai skrip arsip atau dependency lain.

## Alat notebook dan grafik

| Alat | Versi yang diminta | Kondisi `.venv` saat diperiksa dan kegunaannya |
|---|---|---|
| ipykernel | 7.4.0 | Belum terpasang; menghubungkan sesi Python dengan notebook |
| nbformat | 5.11.1 | Terpasang 5.10.4; membaca/menulis struktur notebook |
| nbclient | 0.11.0 | Belum terpasang; menjalankan sel notebook secara terprogram |
| nbconvert | 7.17.2 | Belum terpasang; mengubah notebook menjadi format lain |
| IPython | Tidak dipatok langsung dalam dua requirements tersebut | Belum terpasang; `IPython.display` menampilkan tabel dan Markdown |
| Matplotlib | Tidak dipatok dalam dua requirements tersebut | Terpasang 3.10.6; `matplotlib.pyplot` dipakai cell grafik |

Output notebook yang sudah tersimpan tetap bisa dibaca. Untuk menjalankan sel, ikuti petunjuk environment di README. Alat notebook tidak mengganti algoritma ML. `pytest` 8.4.2 juga terpasang dan diimpor oleh file tes; itu alat pemeriksaan, bukan pembuat prediksi aroma.

## Yang tidak perlu di-install lewat pip

Python sudah membawa **standard library**. Pada sumber aktif, kelompoknya antara lain:

- `pathlib`, `os`, `shutil`: alamat file dan operasi berkas.
- `json`, `re`, `collections`: membaca JSON, aturan teks, serta penghitung/pengelompokan.
- `argparse`, `platform`, `datetime`, `time`, `contextlib`: argumen command, identitas komputer, waktu, dan pengelolaan konteks.
- `hashlib`, `importlib.metadata`, `warnings`: sidik jari file, informasi versi paket, dan pesan peringatan.
- `ast`, `inspect`, `sys`: dipakai helper/notebook untuk membaca struktur atau lokasi kode dan mengatur pencarian modul.

`research`, `belajar`, `audit_labels`, dan `asal_kohor` adalah **modul project sendiri**, bukan paket yang perlu dicari di pip. Titik pada `from .runtime import ROOT` berarti mengambil modul tetangga dalam paket yang sama.

## Contoh membaca fungsi per baris

Buka `KODE_ML/03_Kode/features.py`, fungsi `molecule`, baris 12–16. Ini potongan asli:

```python
def molecule(smiles, index):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"Invalid or multicomponent structure at row {index}; no rows may be dropped.")
    return mol
```

1. **Baris 12:** membuat fungsi milik project. Inputnya teks `smiles` dan nomor baris `index`.
2. **Baris 13:** memanggil RDKit melalui `Chem`. Hasilnya disimpan sebagai objek molekul `mol`, bukan fingerprint.
3. **Baris 14:** mengecek apakah pembacaan gagal atau jumlah komponen bukan satu. `or` berarti salah satu kondisi cukup untuk ditolak.
4. **Baris 15:** menghentikan proses dengan pesan berisi nomor baris. Data bermasalah tidak diam-diam dibuang.
5. **Baris 16:** jika valid, mengembalikan objek molekul untuk dihitung fitur berikutnya.

Jawaban sidang: “`molecule` fungsi project saya. Fungsi ini memakai RDKit untuk membaca SMILES dan memastikan satu struktur terhubung. Outputnya objek molekul, lalu fungsi fitur menghitung Morgan dan deskriptor.”

**Cek paham:** tunjukkan mana nama library, alias, fungsi project, input, pemanggilan library, kondisi penolakan, dan outputnya.
