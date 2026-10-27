# Agenda setelah hasil v7 Morgan 2048 tersedia

Ditetapkan atas instruksi pengguna 24 September 2026; dicatat 25 September 2026.
**Jalankan setelah training v7 selesai dan hasil diaudit.**
Catatan ini bukan pernyataan bahwa hasil v7 sudah tersedia.

## Metode terbaru

Run perfume-five-grouped-v7-m2048-d8, protocol revision 2.
5 sumber Pyrfume, 6686 molekul, 109 label, split 5353/1333 dan kelompok lama tetap.
Morgan radius 2, 2048 bit, tanpa kiralitas.
Tambahan 8: MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA,
NumRotatableBonds, HeavyAtomCount, RingCount.
A/C = 2048 fitur; B/D = 2056. XGBoost/LightGBM = 8 kandidat.
A/B baseline 100; C/D empat studi baru masing-masing 8 jam.
Seleksi mean macro AP dari lima pengulangan grouped CV, lima fold per label.
Seed CV 7/21/42/84/100; estimator seed 42. Ambang utama 0,5; tambahan F1 dari rerata OOF.
Rilis v6 tetap aktif sampai v7 direview. ONNX/Android ditunda.

## 1. Audit hasil sebelum menulis kesimpulan

Baca complete.json, completion_manifest.json, identity.json, selection_frozen.json,
comparison_all.csv, paired_descriptor_cv.json, descriptor_deltas.csv dan ringkasan Optuna.
Cek checksum, kelengkapan delapan kandidat, empat studi/budget, fold identik,
parameter pilihan dan dimensi inferensi. Laporkan trial selesai/gagal, waktu,
mean/SD dan delta per seed.

Pemenang berdasarkan validasi. Bandingkan B-A, D-C, C-A, D-B.
Tampilkan kenaikan dan penurunan. Jika hasil belum lengkap, jangan mengarang angka.

## 2. Rombak laporan berdasarkan hasil

Lokasi:
C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael

- Bab 1: fokus komparasi/optimasi representasi fitur dalam konteks parfum.
  Suh acuan utama; 2048+8 eksperimen proyek, bukan otomatis penemuan metode baru.
- Bab 2: jelaskan Morgan dan delapan deskriptor dengan referensi primer.
  Suh juga membandingkan XGBoost/LightGBM; jangan menyebut hanya RF.
  Bedakan MD terpisah pada Suh dari fitur gabungan proyek.
- Bab 3: tabel 2048/2056, skenario A-D, diagram delapan cabang, kelompok tetap,
  CV/OOF, budget TPE, aturan seleksi. Angka harus cocok dengan manifest.
- Bab 4: tabel/grafik/narasi berdasarkan hasil v7 yang diaudit; pisahkan v6 historis.
  Sertakan seluruh metrik, per label, variasi seed, biaya komputasi dan Optuna.
- Bab 5: simpulan mengikuti hasil, termasuk jika fitur tambahan tidak membantu.
- Cocokkan abstrak Indonesia/Inggris, judul bila perlu, rumusan/tujuan,
  notasi, tabel, gambar, sitasi dan lampiran. Build LaTeX, cek rujukan silang.

## 3. Batas klaim

- Lima sumber relevan dengan parfum, tidak berarti setiap molekul eksklusif parfum.
- Label kosong yang dianggap nol bukan bukti sensori negatif; harmonisasi punya batasan.
- V6 B/D 1024+5 vs v7 B/D 2048+8 mengubah dua hal sekaligus, sehingga tidak
  membuktikan delapan lebih baik daripada lima pada Morgan yang sama.
- B-A menguji paket delapan pada baseline tetap; D-C membandingkan representasi
  setelah tuning masing-masing. Tidak membuktikan kontribusi tiap deskriptor.
- Test historis sudah dilihat, CV bukan nested CV; hasil eksploratori.
  Seed/fold berkorelasi: jangan mengarang signifikansi statistik.
- AP utama, accuracy saja tidak cukup. AP implementasi bukan otomatis
  AUPRC trapezoidal publikasi. Data/label/split berbeda dari Suh.
- Skor bukan intensitas, komposisi parfum atau probabilitas terkalibrasi.
- Selisih kecil/tidak stabil harus dilaporkan apa adanya.

## 4. Penutupan

Update PROJECT_CONTEXT.md, PROJECT_OVERVIEW.md, SAFE_EXECUTION_PLAN.md dan panduan
inferensi jika kandidat v7 diterima. Simpan hasil v6 sebagai riwayat.
Penggantian rilis/ONNX/Android pekerjaan terpisah.
Jangan otomatis menambah eksperimen hanya karena hasil test kurang memuaskan.
