# Menjalankan V8 di Windows

Paket: **perfume-campus-v8-preml-m2048-d8.zip**.

V8 tahap ini hanya memeriksa data Suh dan membuat fitur **Morgan 2048 + 8
deskriptor = 2056 fitur**. Tidak ada training, Optuna, ONNX, atau perubahan v7.
Data sudah di dalam ZIP; tidak perlu clone GitHub, login OSF, atau download dataset.

## Langkah singkat

1. Salin ZIP ke komputer lab, lalu **Extract All**. Masuk ke folder
   **perfume-campus-v8-preml-m2048-d8** yang berisi file RUN_PREML_V8.cmd.
2. Pastikan **Python 3.12 64-bit** sudah terpasang. Jika belum, pasang dari
   [python.org](https://www.python.org/downloads/windows/) dan sertakan Python Launcher.
3. Klik dua kali **RUN_PREML_V8.cmd**. Internet diperlukan saat instalasi
   dependency pertama. Script akan membuat lingkungan .venv-v8 sendiri.
4. Tunggu tulisan **AUDIT SELESAI. STOP SEBELUM ML**.
5. Baca **data/builds/suh-ten-v8-m2048-d8-preml/HASIL_AUDIT_V8.md**.

Kalau lebih suka terminal: buka PowerShell di folder hasil ekstrak, lalu:

~~~powershell
.\RUN_PREML_V8.cmd
~~~

**STOP ini disengaja**, bukan training gagal. Ada masalah pada indeks split kode
acuan dan keputusan kebijakan data yang harus direview. Exit code 2 adalah status
audit selesai tetapi belum disetujui lanjut ML. Error teknis memakai exit code 1.

## Hasil yang dikembalikan

Simpan atau ZIP folder **data/builds/suh-ten-v8-m2048-d8-preml**. Isinya:

- HASIL_AUDIT_V8.md: ringkasan temuan.
- audit.json dan comparison_with_suh.csv: bukti/tabel pembanding.
- manifest.json: checksum data, kode, fitur, dan versi dependency.
- records.csv, label_support.csv, source_inventory.csv.
- PREML_AUDIT_ONLY.npz: 2056 fitur, target, dan fitur acuan untuk audit.
- REFERENCE_SPLIT_DIAGNOSTIC_ONLY.csv: bukti masalah split, bukan split training.

Folder **v8_reference/preml** berisi hasil audit yang sudah dibuat di komputer
penyiapan, untuk pembanding. Runner membangun output lab sendiri, tidak menyalin
hasil audit referensi sebagai hasil komputasi lab.

## Menjalankan ulang

Jalankan file CMD yang sama. Jika hasil sudah lengkap, script memverifikasi
checksum tanpa menimpa hasil. Untuk menghitung ulang dari data asli, gunakan:

~~~powershell
.\.venv-v8\Scripts\python.exe -B -m v8.prepare run --output data/builds/suh-ten-v8-lab-recheck
~~~

Folder tujuan harus baru. Jangan menghapus output lama untuk melewati error.

## Notebook opsional

Notebook ada di **notebooks/04_suh_v8_preml.ipynb**. Untuk membuka:

~~~powershell
.\.venv-v8\Scripts\python.exe -m pip install -r requirements_notebook_v8.txt
.\.venv-v8\Scripts\python.exe -m jupyterlab notebooks/04_suh_v8_preml.ipynb
~~~

Notebook membaca hasil audit dan menampilkan tahapan, distribusi label, serta
perbedaannya. Penggunaan notebook tidak diperlukan untuk menjalankan CMD.

**Jangan jalankan notebook author_code.ipynb milik Suh**, karena memuat training.
Jangan jalankan runner v7 untuk dataset V8. Target V8 bukan 109 label lama dan
keputusan split V8 belum dibekukan.
