# Kesiapan eksperimen v7 - 24 September 2026

Status: siap dibawa ke PC kampus; training penelitian belum dijalankan.

- 6.686 molekul dan 109 label tetap; train/test 5.353/1.333 tetap.
- Lima deskriptor lama identik; tiga deskriptor tambahan valid dan finite.
- Delapan kandidat B/D/E/F untuk kedua algoritma, empat studi baru 8 jam.
- 16 tes lulus: data, fitur, seleksi, tuning/resume, freeze, inferensi dan ZIP.
- Tes model memakai data sintetis kecil, bukan hasil ilmiah v7.
- Percobaan tes awal gagal pada fixture temp karena parent directory belum ada;
  bukti tersimpan di tests-initial.xml. Tes ulang dengan temp unik lulus (tests.xml).
- Ruff pemeriksaan F dan sintaks fatal lulus; git diff --check lulus.
- ZIP diperiksa checksum dan diuji --check dari folder ekstraksi mandiri.
- bash -n lulus; training Linux penuh belum dijalankan.
- Dataset, source runner v6, model v6 dan rilis aktif tidak diganti.
- Studi tetap eksploratori. Tidak ada klaim delapan lebih baik sebelum hasil tersedia.

Lihat package_validation.json untuk lokasi arsip, SHA-256 dan source yang diverifikasi.
Rencana lengkap: ../../DESCRIPTOR_COMPARISON_V7_PLAN.md.
