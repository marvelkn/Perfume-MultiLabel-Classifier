# Step 5 — Diizinkan, tertahan oleh Step 4

**Step 5 belum selesai.** Persiapan dan verifikasi ringan selesai; pembekuan jumlah trial dan pembuatan run final menunggu smoke real-data yang valid. Izin pengguna mencakup Step 4 sampai Step 5, sehingga tidak perlu meminta ulang izin kedua tahap ini setelah syarat teknis terpenuhi.

| Komponen | Status |
| --- | --- |
| Protokol, dataset, source, dan partisi | Checksum cocok; 85 berkas frozen |
| Environment | Python 3.12.14 dan 14 dependency sesuai snapshot protokol |
| Profil final | config.safe-training.yaml valid; 2 thread, 4 GiB proses, minimum 4 GiB RAM bebas, 20 menit/sesi |
| Batas tuning | Sudah dipraregistrasikan: 14.400 detik aktif per model |
| Smoke Step 4 | STOP sebelum fitting; baseline akhir 84,875°C |
| Estimasi runtime XGBoost/LightGBM | Belum tersedia |
| Jumlah trial final, run final, ledger | Belum dibuat |
| Final test | Matriks tidak dibuka; file hanya dibaca sebagai byte untuk checksum |

Urutan setelah Step 4 dan cooldown GO:

1. Verifikasi sembilan label-fit selesai per learner dan kedua estimasi runtime valid sesuai rumus pada PREREGISTRATION.md.
2. Hitung `N = min(50, floor(0.8 * 14400 / max(t_hat[xgb], t_hat[lgbm])))`. Gunakan jumlah trial yang sama bagi kedua learner; minimum 20, maksimum 50. Setiap estimasi harus memenuhi batas 960 detik per trial. Jika gate gagal, STOP dan amandemen hanya berdasarkan resource sebelum tuning; jangan memilih angka pengganti tanpa bukti.
3. Verifikasi ulang checksum dan environment, lalu buat run final baru dari dataset audited-20260904-v2 memakai safe-training. Simpan snapshot konfigurasi, source/dependency, partisi dan keputusan protokol yang tidak diubah setelah dibekukan.
4. Simpan keputusan budget dan ledger awal nol, periksa partisi run terhadap partisi frozen, lalu tutup Step 5. Step 6 dan seterusnya belum diizinkan.

**Gap operasional yang masih terbuka:** CLI saat ini membatasi durasi satu sesi, tetapi belum menegakkan akumulasi empat jam atau total percobaan lintas sesi. Controller/ledger yang memeriksa cap tersebut harus tersedia sebelum tuning Step 8/9; ledger kosong saja bukan pengaman budget. Semua percobaan COMPLETE/PRUNED/FAIL/terputus menghitung waktu aktif dan jatah percobaan sesuai praregistrasi. Baseline, smoke, cooldown, final fit, dan sensitivitas dicatat terpisah.

Bukti: [preflight.json](preflight.json), [readiness.json](readiness.json), [status.json](status.json), dan [Step 4 attempt 03](../step4_20260908/attempt-03/README.md). Tidak ada hasil model, skor test, atau angka runtime sintetis yang dipakai untuk menentukan budget.
