# Step 3 — Audit final dataset dan praregistrasi

**GO untuk integritas dataset dan keputusan protokol sebelum training.** Pengguna mengizinkan Step 3 dan menetapkan **4 jam komputasi aktif per model** untuk Optuna. Aturan disimpan dalam [PREREGISTRATION.md](../../PREREGISTRATION.md) dan [protocol.json](../../protocols/essenza_v1_20260908/protocol.json). Jumlah trial final menunggu pengukuran resource smoke di Step 4 dan pembekuan Step 5.

## Dataset dan pemeriksaan

Unit analisis adalah satu molekul canonical single-fragment dengan 25 label aroma. Development memuat **5.383 molekul × 2.053 fitur**; test **1.320 molekul** tetap tertutup untuk analisis pada tahap ini.

Notebook memeriksa 29 checksum dataset/snapshot/run referensi, dataset ID, feature schema, urutan label, dimensi/dtype, nilai nonfinite, bit biner, range deskriptor, identitas unik development, dukungan label, serta keterpisahan dan cakupan partisi. Hanya matriks development yang dideserialisasi. Seluruh pemeriksaan struktural/integritas lulus.

Partisi utama seed 42 cocok persis dengan indeks arsip: 4.575 fitting/CV, 808 threshold, dan tiap fold 2.592 train / 458 stop / 1.525 score. Semua peran utama memiliki kedua kelas untuk semua label.

## Temuan dan implikasi

| Temuan | Bukti | Risiko / confidence | Keputusan |
| --- | --- | --- | --- |
| Scaffold mendominasi beberapa fold | 606 kelompok development; kelompok kosong 2.303/5.383 (42,78%) | Tinggi / tinggi: fold molekul tidak seimbang | Kelompok kosong tetap utuh; laporkan ukuran tiap fold dan pooled AP selain mean fold AP. |
| Early-stop scaffold awal tidak layak untuk semua label | Fold 1 kekurangan kelas pada apple dan winey | Tinggi / tinggi: kebijakan early stopping tidak sebanding | Semua sensitivitas menggunakan rounds utama tetap, train+stop digabung, tanpa early stopping ulang. Seluruh train/score final lolos class-support check. |
| Missing annotation bukan negatif terverifikasi | 381/5.383 development (7,08%) tanpa target positif; kebijakan sumber eksplisit | Tinggi / tinggi untuk keberadaan asumsi; besarnya dampak belum diukur | Pertahankan baris; ungkapkan keterbatasan anotasi; jangan menyebut angka sebagai probabilitas sensori. |
| Ketidakseimbangan target | Positif development 233 (musty) sampai 1.942 (fruity) | Sedang / tinggi: agregat berbobot dapat didominasi label umum | Primary macro AP; imbalance none/class_weight/random_oversample hanya pada training. |
| Source coverage tidak seragam | Acceptance record: GoodScents 92,07%; Leffingwell 100%; Arctander 78,43%; Flavornet 67,46%; Sigma 74,81% | Sedang / tinggi untuk counts; dampak bias belum diukur | Provenance dipertahankan; acceptance bukan kontribusi molekul unik atau validasi sensori. |
| Rentang domain luas | MolWt 4,003–1.297,128; contoh [He] tidak mempunyai positif pada target terpilih | Sedang / tinggi: cakupan lebih luas dari bahan parfum biasa | Tidak mengubah dataset berdasarkan skor; batasi klaim penggunaan dan catat contoh development. |
| Batas generalisasi historis | Audit lama: 1.231/1.320 test (93,26%) memiliki scaffold yang terlihat; sebagian molekul historis | Tinggi / tinggi untuk bukti audit; performa belum tersedia | Sensitivitas bersyarat pada resep, bukan nested CV; tidak mengklaim test eksternal independen. |

Tidak ada analisis tren waktu: dataset merupakan snapshot source revision tetap, bukan seri waktu. Tidak ada refetch sumber atau perubahan label. Identitas development unik diperiksa ulang; overlap identitas lintas development/test dan scaffold test memakai audit historis yang datasetnya tetap sama hash, bukan pembacaan ulang test pada Step 3.

## Keputusan yang dibekukan

| Aspek | Keputusan |
| --- | --- |
| Metrik utama | Mean tiga fold dari macro AP seluruh 25 label |
| Seed utama / tambahan | **42 / 123, 2026** |
| Sensitivitas | Random fixed-round 42/123/2026 dan scaffold 42, development tetap |
| Resep sensitivitas | Parameter/imbalance/median rounds utama dibekukan; tidak retuning; threshold/test tidak dipakai |
| Optuna | Studi terpisah, TPE seed 42, startup 10, MedianPruner setelah fold, satu trial bersamaan |
| Budget | **14.400 detik aktif/model**; target N sama, diturunkan dari waktu smoke; batas per sesi tetap 20 menit |
| Trial final | Belum ada angka hasil pengukuran; aturan floor/cap/gate dibekukan dalam protokol |
| MLSMOTE | Nonaktif |
| Pemilihan final | Best COMPLETE trial tiap learner, lalu CV kedua kandidat; test tidak menentukan pemilihan |
| Klaim | Tidak menjamin global optimum, manfaat tuning, atau signifikansi dari tiga fold/seed |

Dasar pencarian sekuensial dan pengendalian eksperimen dicatat di protokol dengan sumber primer [Bergstra et al.](https://papers.nips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html), [Akiba et al.](https://arxiv.org/abs/1907.10902), dan [Cawley–Talbot](https://jmlr.org/papers/v11/cawley10a.html). Nilai 4 jam berasal dari pengguna; faktor estimasi/cadangan serta batas jumlah trial adalah keputusan rekayasa yang dijelaskan, bukan angka optimal dari paper.

## Validasi dan artefak

Notebook akhir menjalankan **7 sel kode berurutan**, seluruhnya lulus, melalui interpreter Python proyek. Durasi eksekusi notebook **5,672 detik**; peak working set proses audit **171,40 MiB**. RAM sistem bebas setelah audit sekitar **13,35 GiB**. Pengukuran RAM ini tidak mencakup launcher/aplikasi lain. Suhu CPU aktual tidak tersedia; tidak ada sensor live terverifikasi pada audit ringan ini.

- [audit_notebook.ipynb](audit_notebook.ipynb) dan [execute_audit.py](execute_audit.py): kode pemeriksaan serta output aktual; tidak memerlukan state kernel interaktif.
- [dataset_audit.json](dataset_audit.json): integritas, profil development, serta temuan rancangan partisi awal.
- [development_partitions.json](development_partitions.json): partisi utama.
- [sensitivity_partitions.json](sensitivity_partitions.json): rancangan awal dengan holdout stop, dipertahankan sebagai bukti diagnosis.
- [sensitivity_execution_partitions.json](sensitivity_execution_partitions.json): **partisi sensitivitas final** yang digunakan tahap berikutnya.
- [sensitivity_resolution.json](sensitivity_resolution.json): alasan fixed rounds dan bukti class support final.
- [development_scaffold_groups.json](development_scaffold_groups.json), [development_domain_examples.json](development_domain_examples.json): konteks development.
- [alignment.md](alignment.md): gap Bab III, lokasi frontend, kontrak fitur, dan kebutuhan runner tahap berikutnya.
- [audit-execution.json](audit-execution.json), [preflight.json](preflight.json), [completion.json](completion.json): jejak eksekusi dan verifikasi.
- [freeze_manifest.json](../../protocols/essenza_v1_20260908/freeze_manifest.json): checksum dokumen, indeks, source/config, dan artefak terkait.

Tidak ada training, prediksi/evaluasi test, run operasional baru, perubahan source model/CLI, perubahan frontend/LaTeX, commit, atau push. Unit test ML Step 2 tidak diulang karena source produksi dan tesnya tetap sama; Step 3 memvalidasi notebook audit dan verifier praregistrasi.

**Berikutnya Step 4**, hanya setelah perintah terpisah dan pemeriksaan ulang sensor/baseline termal. GO Step 3 belum berarti laptop siap training.

