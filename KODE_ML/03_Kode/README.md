# Kode asli project

Di sinilah implementasi yang dipakai notebook dan program berada. Untuk urutan belajar, buka ../MULAI_BELAJAR.md.

- **config/**: pengaturan dataset, eksperimen dan komputasi.
- **tests/**: pemeriksaan kontrak data dan jalur program; bukan training.
- **preprocessing.py**: membersihkan dan menggabungkan data.
- **features.py**: menghitung Morgan dan deskriptor.
- **training.py**: membuat dan melatih classifier.
- **tuning.py, validation.py**: Optuna, fold dan OOF.
- **metrics.py, experiment.py**: evaluasi dan urutan eksperimen.
- **inference.py**: memakai model tersimpan.

Nama foldernya 03_Kode. Nama import Python tetap research, diarahkan oleh research/__init__.py di root project. Jangan mengganti import menjadi 03_Kode. Tidak ada perubahan rumus, parameter, data atau model dalam perapian nama folder.
