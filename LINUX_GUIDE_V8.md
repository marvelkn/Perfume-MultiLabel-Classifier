# Menjalankan V8 di Linux

Data Suh sudah ada di ZIP. V8 hanya menyiapkan dan mengaudit fitur
**Morgan 2048 + 8 deskriptor (2056 fitur)**, tanpa training atau Optuna.

## Langkah singkat

1. Ekstrak **perfume-campus-v8-preml-m2048-d8.zip** melalui file manager.
2. Buka terminal di folder **perfume-campus-v8-preml-m2048-d8**.
3. Periksa Python lalu jalankan:

~~~bash
python3.12 --version
bash RUN_PREML_V8.sh
~~~

Jika Python 3.12 belum ada, minta pengelola lab memasangnya. Pada **Ubuntu 24.04**
yang menyediakan paket tersebut, perintahnya:

~~~bash
sudo apt update
sudo apt install python3.12 python3.12-venv
~~~

Perintah instalasi ini khusus distro yang menyediakan paket Python 3.12 tersebut;
jangan mengganti Python sistem atau menganggap semua distro memakai apt.

Script membuat .venv-v8 dan memasang dependency. Internet dibutuhkan pertama
kali untuk dependency, bukan untuk dataset.

## Selesai ketika muncul STOP

Tulisan **AUDIT SELESAI. STOP sebelum ML** berarti proses audit selesai dan
dihentikan sesuai rancangan. Exit code 2 disengaja karena temuan belum disetujui
untuk training. Exit code 1 adalah error teknis.

Buka **data/builds/suh-ten-v8-m2048-d8-preml/HASIL_AUDIT_V8.md**.
Simpan atau ZIP seluruh folder hasil tersebut untuk dikembalikan.

## Mengulang

~~~bash
bash RUN_PREML_V8.sh
~~~

Hasil lengkap diverifikasi, tidak ditimpa. Untuk menghitung ulang dari data asli
pada direktori baru:

~~~bash
.venv-v8/bin/python -B -m v8.prepare run --output data/builds/suh-ten-v8-lab-recheck
~~~

## Notebook opsional

~~~bash
.venv-v8/bin/python -m pip install -r requirements_notebook_v8.txt
.venv-v8/bin/python -m jupyterlab notebooks/04_suh_v8_preml.ipynb
~~~

**Jangan menjalankan notebook asli Suh atau runner v7.** Paket ini hanya tahap
pre-ML, bukan eksperimen 8 jam. Lihat **V8_PREML_PLAN.md** untuk keputusan yang
masih harus diselesaikan.
