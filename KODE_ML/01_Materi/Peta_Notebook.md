# Peta notebook: inti dan rujukan

Mulai dari **dasar ilmiah, gap, rancangan A–D, dan alasan XGBoost/LightGBM** pada PDF, lalu ikuti urutan di bawah. Ini peta untuk panduan aktif, bukan jalur materi baru. Revisi prioritas: **6 November 2026**. Rujukan notebook di bawah mengikuti **materi inti halaman 1-22**; judul bagian tetap bisa dicari dengan Ctrl+F. PDF kini berjumlah **34 halaman**. Tambahan halaman **23-34** adalah peta 33 referensi skripsi, tanpa notebook tambahan; lihat [versi teks](../05_Referensi/PETA_REFERENSI_SKRIPSI.md).

**Inti** berarti dapat menjelaskan keputusan, alasan, output, dan batasnya. **Rujukan fungsi** cukup dipahami tujuan serta lokasinya; **Pendalaman** dibuka untuk rincian. Baca output tersimpan tanpa menjalankan sel bila hanya belajar. Persiapan diperlukan saat akan menjalankan notebook mandiri.

| PDF | Notebook | Inti yang dibaca | Rujukan fungsi / pendalaman |
|---|---|---|---|
| 2–6 | Belum perlu notebook | Masalah, gap, kontribusi, A–D, alasan dua boosting dan batas tanpa RF | Angka lengkap studi terdahulu dan uraian RF lanjutan |
| 7 | [01_Sumber_Data](../01_Sumber_Data.ipynb) | **Konfigurasi sumber**, **Baca snapshot dan ukur tabel**, aturan identitas dan CAS, **Ikuti contoh join GoodScents** | `load_config`, `load_sources`, `_columns`, `_key`, exception dan sintaks join |
| 8 | [02_Preprocessing_dan_Audit_Label](../02_Preprocessing_dan_Audit_Label.ipynb) | **normalize**, **taxonomy_from_ifra**, **tokens_for**, **map_labels**, **merge_sources**, **Buktikan asal angka 109**; satu contoh alias, kegagalan/parsial dan target nol | Implementasi audit menyeluruh, seluruh alias/istilah gagal dan rekonsiliasi setiap angka. Temuan dan batas maknanya tetap inti |
| 9 | [03_Dataset_dan_Fitur](../03_Dataset_dan_Fitur.ipynb) | **calculate_morgan**, **calculate**, **features**, **encode_frozen_records**, **Lihat satu molekul yang memang ada dalam data** | Wrapper `verify`, format NPZ, checksum dan perhitungan ulang seluruh baris. Fungsi menjaga pasangan fitur/target tetap perlu diketahui |
| 10 | [04_Grup_Split_dan_Fold](../04_Grup_Split_dan_Fold.ipynb) | **Bedakan pembentukan historis dengan pemakaian saat ini**, aturan **structure_feature_groups**, **outer_split**, **encode_and_split**, **inner_folds**, **Contoh pembagian satu label**, batas split | AST, union-find, fallback dan rekonstruksi seluruh indeks. Morgan historis 1.024 bit versus fitur utama 2.048 bit tetap inti |
| 11–12 | [05_Model_dan_Bobot](../05_Model_dan_Bobot.ipynb) | **training.make_model**, **training.balanced_sample_weights**, **Buktikan rumus bobot pada label aktual**, alur **training.fit** dan final fit; jumlah classifier | Seluruh nilai recipe, callback, nama file, checksum dan checkpoint. Bobot hanya dari bagian fitting tetap inti |
| 13–15 | [06_Optuna_CV_dan_OOF](../06_Optuna_CV_dan_OOF.ipynb) | **Baca kebijakan tuning**, alur **tuning.suggest/initialize/tune**, **Baca hasil empat studi sebenarnya**, **validation.cross_validate**, **validation.repeated_grouped_cv** | `atomic_dump`, `event`, database, heartbeat dan rincian budget keeper; waktu aktif dan resume tanpa anggaran baru tetap inti |
| 16–18 | [07_Evaluasi_dan_Pemilihan](../07_Evaluasi_dan_Pemilihan.ipynb) | Arti **metrics.binary_metrics**, **metrics.summarize**, **Hitung ulang metrik delapan kandidat**, beda AP/accuracy dan macro averaging | Penurunan akar SD, enumerasi banyak pasangan AUROC dan detail penyimpanan hasil. Makna SD/AUROC tetap inti |
| 19 | [06_Optuna_CV_dan_OOF](../06_Optuna_CV_dan_OOF.ipynb) | **validation.f1_from_metrics**, alur **validation.select_validation_thresholds**, **Hitung ulang ambang dari OOF tersimpan**: OOF train → pilih → kunci → test; utama tetap 0,5 dan batas non-nested | Seluruh grid threshold dan aturan seri |
| 20 | [07_Evaluasi_dan_Pemilihan](../07_Evaluasi_dan_Pemilihan.ipynb) | **Baca delapan kandidat dan hasil utama**, **Reproduksi pilihan dari CV tersimpan**, **Baca delta dan cek kesamaan cakupan**, urutan **experiment.run_all**, seluruh metrik yang menurun dan keterbatasan | CLI, seluruh hash, format bundle dan bagian **Kode pendukung eksekusi** (`runtime.py`). Tidak perlu mengaktifkan gerbang eksperimen |
| 21–22 | [08_Prediksi_Molekul](../08_Prediksi_Molekul.ipynb) | **Temukan hasil utama dan kontrak model**, alur **inference.predict**, output **Jalankan prediksi nyata**, **Lihat keputusan positif**, **Periksa input di luar kontrak** dan batas klaim | Pencarian folder hasil, serializer dan CLI. Notebook ini inferensi Python; pengujian Android/USB memakai bukti aplikasi tersendiri |

## Latihan hitung

- **Inti:** satu contoh bobot kelas, keputusan threshold, confusion counts, precision/recall/F1, AP, macro AP, serta mean fold/seed. Hitung F1 per label sebelum macro averaging.
- **Pendalaman:** akar SD dan banyak pasangan AUROC. Tetap jelaskan bahwa SD menggambarkan variasi pembagian train yang sama, bukan interval kepercayaan; AUROC membaca ranking skor positif terhadap negatif.

## Batas yang menyertai alur inti

Target nol berarti anotasi tidak tercatat; alias manual belum seluruhnya divalidasi pakar. Group split tidak memisahkan seluruh kemiripan/scaffold. Test pernah diakses dan CV bukan nested; lima seed bukan lima dataset independen. Model terpilih dari AP validasi tidak wajib terbaik untuk semua metrik. Skor belum terkalibrasi sebagai persepsi manusia, bukan intensitas/komposisi, dan belum memprediksi aroma akhir campuran.

Pada aplikasi per 7 November 2026, **RDKit dan 109 model LightGBM D asli (joblib) berjalan di komputer lokal**; Android terhubung melalui USB untuk mengirim struktur dan menerima skor. Prediksi baru tetap memerlukan komputer, sedangkan Riwayat/Koleksi tersimpan pada HP. Notebook 08 menunjukkan inferensi Python model asli. Kesetaraan implementasi dan uji fungsi bukan validasi sensori, manfaat pengguna, atau kesiapan produksi.

Urutan ini untuk membaca, bukan ketergantungan eksekusi antar-notebook. **JALANKAN_EKSPERIMEN_BARU = False** tetap dipertahankan. Sel kode, output, data, hasil, dan model tidak dipangkas untuk mempersingkat bacaan.
