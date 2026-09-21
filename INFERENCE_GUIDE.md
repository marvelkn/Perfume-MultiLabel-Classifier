# Inferensi Python model final

Status 21 September 2026: model final **LightGBM D grouped-v6**, 109 label,
1.029 fitur. Training tambahan dan pseudo-labeling tidak dijalankan.
ONNX, Android, dan deployment ditunda. API `/fingerprint` tetap layanan fitur;
perintah Python di bawah menjalankan prediksi model secara lokal tanpa jaringan.

## Jalankan dari folder backend

PowerShell Windows:

```powershell
Set-Location 'C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier'
.\.venv\Scripts\python.exe -m src.release --verify
.\.venv\Scripts\python.exe -m src.inference --smiles 'COc1cc(C=O)ccc1O'
```

Linux, setelah menyalin backend dan paket rilis ke komputer tujuan:

```bash
cd /lokasi/Perfume-MultiLabel-Classifier
.venv/bin/python -m src.release --verify
.venv/bin/python -m src.inference --smiles 'COc1cc(C=O)ccc1O'
```

Lingkungan lokal saat penyelarasan sudah cocok. Untuk lingkungan baru gunakan
Python 3.12, buat `.venv`, lalu instal `requirements_inference.txt` dengan pip
pada lingkungan tersebut. Dependensi layanan API terpisah di
`requirements_api.txt`. Inferensi memeriksa versi pustaka terhadap paket training.

## Lokasi paket

`releases/grouped-v6-lgbm-d/` berisi:

- `release.json`: pengikatan model-label, checksum, versi dan identitas rilis;
- `feature_spec.json`: radius 2, 1.024 bit Morgan tanpa kiralitas, lima deskriptor;
- `models/`: 109 model Python hasil training yang sudah ada;
- `evidence/`: salinan hasil, penguncian seleksi, ambang tambahan dan provenance.

Paket lokal ini diabaikan Git karena memuat model biner. Clone repository saja
belum membawa model. Salin seluruh folder rilis atau bangun kembali dari run
kampus yang selesai. Jangan menggunakan aset ONNX 25 label; 50 ONNX lama dan
dua array ambangnya telah dihapus. Model hasil penelitian kampus tidak dihapus.

Untuk mengemas kembali (hanya bila folder tujuan belum ada):

```powershell
.\.venv\Scripts\python.exe -m src.release --run '.local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v6/perfume-campus-grouped-v6/runs/perfume-five-grouped-v6'
```

Perintah ini memverifikasi dan menyalin model, bukan melatih. Rilis yang sudah
ada tidak ditimpa. `--output` dapat dipakai untuk tujuan baru; `--release` pada
perintah inferensi memilih paket tersebut. Jangan memuat joblib dari pihak yang
tidak dipercaya; checksum mendeteksi perubahan, bukan membuktikan asal penerbit.

## Kontrak fitur dan keluaran

Urutan fitur: 1.024 bit Morgan, kemudian `MolWt`, `MolLogP`, `NumHDonors`,
`NumHAcceptors`, `TPSA`, seluruhnya `float32`. Satu molekul terhubung diperlukan.
Penyeragaman SMILES dilakukan sebelum ekstraksi agar sama dengan training.
Schema ID aktif:
`630793962c4fb25dd0e1fbf39c02d1b4ba24502cf4512f6110ee841e957870eb`.

JSON hasil memuat `model_id`, `feature_schema_id`, `model_manifest_sha256`,
`predictions` (109 baris dalam urutan label tetap), dan `predicted_labels`.
Setiap prediksi memuat `label`, `score`, `threshold`, dan `predicted`.
Ambang utama tetap 0,5. Ambang per label hasil validasi hanya disimpan sebagai
analisis tambahan dan tidak diaktifkan diam-diam. Semua label dapat berada di
bawah ambang; daftar positif kosong tidak dipaksa menjadi top-k.

Skor belum terkalibrasi; bukan intensitas aroma, komposisi, atau prediksi campuran.
Nilai antarlabel tidak harus berjumlah satu. Prediksi ini tidak menggantikan uji
sensori atau identifikasi bahan parfum.

Untuk penggunaan berulang dalam proses Python, muat `Predictor` sekali:

```python
from src.inference import Predictor
model = Predictor()
result = model.predict('COc1cc(C=O)ccc1O')
```

## Bukti verifikasi

`reports/backend_alignment_20260921/verification.json` mencatat:

- 124 berkas paket dan 14 berkas dataset lolos checksum;
- fitur seluruh 6.686 molekul sama persis dengan artefak training/test;
- 32 posisi test dipilih merata sebelum membandingkan skor: 3.488 skor dari
  109 label memiliki selisih maksimum 0 dan keputusan ambang 0,5 identik;
- input salah/multifragmen dan schema ID salah ditolak;
- SMILES setara, prediksi berulang, fitur API, dan keluaran tanpa positif diperiksa;
- 23 tes fitur, API dan kontrak inferensi lulus (`tests.xml`).

Pemeriksaan tersebut menguji kesetaraan implementasi pada sampel prediksi,
bukan estimasi akurasi baru atau bukti kemampuan pada molekul independen.
Tidak ada ONNX, Android, deployment, atau training yang dijalankan.
