# Hasil audit dan perapian KODE_ML — 26 Oktober 2026

**Selesai: delapan notebook bertahap dan source utama berada di dalam KODE_ML.**
Hasil utama penelitian tetap dikunci. Tidak ada training/tuning baru, perubahan dataset/model,
penambahan fuzzy/alias, atau perubahan frontend/LaTeX.

## Apa yang sebelumnya belum sesuai?

KODE_ML sebelumnya berisi lima wrapper/contoh singkat dan satu notebook panjang,
sementara implementasi utama berada di luar folder. Ini tidak memenuhi tujuan belajar semua proses dari satu folder.
Sekarang seluruh package aktif, empat konfigurasi dan tes berada di `KODE_ML/research/`.
Notebook memanggil dan menampilkan fungsi yang sama dari sana; tidak membuat implementasi kedua.
`research/__init__.py` di root hanya mempertahankan kompatibilitas import/CLI.

## Kesetaraan yang diperiksa

| Bagian | Bukti dan status |
|---|---|
| Relokasi source | 68 fungsi/kelas diperiksa: 66 AST identik; dua runner hanya mengubah ekspresi lokasi source untuk identitas run. Setelah normalisasi kedua ekspresi, isinya identik. |
| Konfigurasi | Empat JSON byte-identik sebelum/sesudah relokasi: sumber, dataset, eksperimen, profil CPU. |
| Cleaning/pemetaan | Replay snapshot di memori; records, mapped/unmapped counters/CSV, urutan label dan target sama dengan artefak asli. |
| Asal 109 kelas | Seleksi 184 istilah berdasarkan train >=30 positif dan >=2 negatif menghasilkan daftar/urutan yang sama. |
| Fitur | Morgan2048 dan delapan deskriptor seluruh 6.686 molekul dihitung ulang; nilainya persis sama dengan artefak. |
| Asal grup/split | Tiga fungsi historis AST-identik; fingerprint historis tersimpan menghasilkan grup, indeks train/test, 545 fold awal dan target identik. |
| Parameter/model | Delapan recipe dimuat untuk membuat objek estimator tanpa fit; bobot kelas diperiksa dengan rumus asli. |
| Optuna | Empat studi/CSV hasil utama direkonsiliasi; tidak membuat trial baru. |
| OOF | Seluruh 109 ambang tambahan kandidat terpilih dihitung ulang dari pooled OOF dan dictionary-nya identik. |
| Evaluasi | Enam metrik serta macro-F1 delapan kandidat dihitung ulang dari prediksi tersimpan; toleransi absolut 1e-12. Pilihan kandidat dari CV sama. |
| Inferensi | Tujuh tes baca-saja lulus, termasuk perbandingan seluruh skor/keputusan tiga SMILES dengan keluaran sebelum perapian. Notebook 08 menjalankan inferensi nyata dan penolakan multifragmen. |
| Keutuhan artefak | SHA-256 19 file kontrak data/fitur dan 1.199 file hasil utama cocok. |
| Source yang tampil | 84 blok source/konstanta pada output notebook cocok dengan file aktual dan hash-nya. |

**Batas klaim:** tidak dilakukan training ulang puluhan jam. Kesetaraan source/metode, data,
fitur, split dan keluaran yang diperiksa tidak membuktikan training ulang pada semua hardware
akan menghasilkan model bit-identik. Prediksi/metrik tersimpan tidak disebut sebagai hasil training notebook.

## Perbedaan yang disengaja, bukan perubahan metode ML

1. Lokasi package/config berpindah ke KODE_ML. Identitas run baru mencatat lokasi source baru;
   hasil historis tidak boleh di-resume dengan identitas baru.
2. Notebook menampilkan source melalui introspeksi, lalu memanggil fungsi aktual. Tidak ada salinan
   definisi training alternatif di notebook. Helper `belajar.py` membuka decorator untuk menampilkan fungsi asli.
3. `audit_labels.py` adalah alat pemeriksa tambahan. Ia memakai mapper produksi, lalu membuat tabel jejak token.
   Tidak menambahkan label atau memperbaiki pemetaan secara diam-diam.
4. `asal_kohor.py` hanya audit pembentukan kohor historis. Tiga fungsi algoritmanya sama, tetapi
   fallback pembuat fitur lama diganti error agar argumen Morgan historis tersimpan wajib diberikan.
   Kode aktif training tetap membaca kohor beku dan fitur 2048+8.
5. Loader sumber lokal dan loader kohor beku yang ada sebelum tugas ini dipertahankan. Notebook tidak
   mengunduh ulang sumber atau membangun ulang artefak disk. Urutan notebook adalah urutan belajar;
   setiap notebook membaca inputnya sendiri dan dapat dijalankan dengan kernel baru.
6. Training penuh ditampilkan melalui runner asli di notebook 07, dengan gerbang default False.
   Run All bawaan hanya audit/pembacaan hasil. Mengaktifkan gerbang berarti eksperimen baru dengan folder baru.

## Status notebook

| Notebook | Sel kode dijalankan | Status |
|---|---:|---|
| 01_Sumber_Data.ipynb | 16 | Lulus, kernel baru |
| 02_Preprocessing_dan_Audit_Label.ipynb | 16 | Lulus, kernel baru |
| 03_Dataset_dan_Fitur.ipynb | 14 | Lulus, kernel baru |
| 04_Grup_Split_dan_Fold.ipynb | 9 | Lulus, kernel baru |
| 05_Model_dan_Bobot.ipynb | 10 | Lulus, kernel baru |
| 06_Optuna_CV_dan_OOF.ipynb | 19 | Lulus, kernel baru |
| 07_Evaluasi_dan_Pemilihan.ipynb | 31 | Lulus, kernel baru |
| 08_Prediksi_Molekul.ipynb | 10 | Lulus, kernel baru |

Total **125 sel kode**, masing-masing notebook memakai kernel baru. Saat verifikasi, pemanggilan
fit XGBoost/LightGBM, pembuatan trial Optuna, serta penulisan ke proyek diblokir.
Output disimpan setelah eksekusi sukses. Semua empat grafik dan tampilan HTML notebook diperiksa.
Penjelasan Markdown kemudian diperkaya tanpa mengubah sel kode yang sudah diuji.
Tes targeted: `python -B -m pytest KODE_ML/research/tests/test_research_readonly.py -q -p no:cacheprovider`
→ **7 lulus**. Tidak menjalankan keseluruhan tes aplikasi lama karena sebagian melakukan fitting.

## Temuan label yang tidak boleh ditutupi

- **31 alias manual**, 17 terpakai; bukan 109 kelas model.
- **184 istilah IFRA** dibaca dinamis → **109 target** lolos dukungan train; 75 lainnya punya 1–28 positif train.
- **6.494 dari 45.689 kemunculan token gagal dipetakan (498 istilah berbeda)**.
- **3.898 baris diterima** tetap mempunyai token yang tidak terpetakan; pencatatannya ada, tetapi token tidak masuk target.
- **1.230 token** yang berhasil cocok menuju kelas yang kemudian dikecualikan karena dukungan train.
- Ada **11 baris train dan 4 test** dengan seluruh 109 target nol setelah seleksi. Nol bukan bukti tidak berbau.
- Contoh istilah perlu tinjauan: caramellic, camphoreous, musky. Istilah lain tidak punya tujuan IFRA,
  misalnya meaty/winey. Kemiripan tulisan saja belum cukup untuk memutuskan alias.

**Audit membuktikan pelaksanaan aturan, bukan akurasi semantik 100%.** Seluruh daftar dan alasan
tersimpan pada `hasil_audit_label/`, dan dijelaskan di notebook 02. Perbaikan taxonomy/alias yang
benar-benar mengubah target harus menjadi eksperimen baru; angka hasil lama tidak boleh dilabeli ulang.

## File dibuat, diubah dan dipindahkan

| Kelompok | Perubahan |
|---|---|
| 01–08 .ipynb | Delapan notebook baru, dengan penjelasan, source aktual, output, dan pertanyaan sidang. |
| KODE_ML/research/ | Package source/config/tes direlokasi dari root; menjadi acuan utama. |
| belajar.py | Penampil source lokal yang diimpor; tanpa logika ML alternatif. |
| audit_labels.py | Audit jejak pemetaan dan seleksi label secara read-only. |
| asal_kohor.py | Kode asal grup/split/seleksi target untuk audit di memori. |
| hasil_audit_label/ | Ringkasan, sembilan CSV lengkap, dan penjelasan temuan. |
| README.md, PANDUAN_SIDANG_KODE.md | Urutan belajar, peta source/data/hasil, batas klaim dan tanya jawab. |
| requirements_belajar.txt | Kebutuhan alat notebook; environment ML asli tidak diubah saat tugas. |
| HASIL_VERIFIKASI.md/.json, DAFTAR_PERUBAHAN.json | Bukti verifikasi dan inventaris perubahan. |
| Root README, MULAI_BACA_KODE, PROJECT_CONTEXT, PROJECT_OVERVIEW, IMPLEMENTATION_STATUS, SAFE_EXECUTION_PLAN | Tautan/status diselaraskan dengan KODE_ML; riwayat SAFE_EXECUTION_PLAN dipertahankan. |
| research/__init__.py di root | Penghubung ke source dalam KODE_ML, tanpa logika ML duplikat. |
| Wrapper dan notebook monolitik lama | Diarsipkan bersama package root lama, bukan dihapus permanen. |

**34 file sebelum perubahan** masuk backup ZIP yang diperiksa CRC dan SHA.
**27 file lama** dipindahkan ke arsip dengan struktur path semula.
Daftar path lengkap: [DAFTAR_PERUBAHAN.json](DAFTAR_PERUBAHAN.json).

Backup: `../.local-backups/kode-ml-notebooks-20261026/source-before.zip`.
Arsip: `../archive/kode-ml-before-notebooks-20261026/`.
Untuk memulihkan, ekstrak backup ke folder berbeda dan bandingkan; jangan menimpa source baru tanpa meninjau perubahan.

## Mulai belajar

Buka [README.md](README.md), lalu [01_Sumber_Data.ipynb](01_Sumber_Data.ipynb).
Jika ingin langsung menjawab pertanyaan ALIASES/109, buka [02_Preprocessing_dan_Audit_Label.ipynb](02_Preprocessing_dan_Audit_Label.ipynb).
