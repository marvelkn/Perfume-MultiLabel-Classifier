# Alignment — Step 3, 8 September 2026

Scope edit: backend asli. Frontend asli dan worktree audit dibaca sebagai referensi. Bab LaTeX tidak diubah pada Step 3.

| Komponen | Bukti terkini | Implikasi dan tindak lanjut |
| --- | --- | --- |
| Partisi Bab III | Bab3.tex baris 53–76: seed 42; 808 threshold; 4.575 fitting; fold 2.592/458/1.525 | **Selaras**; notebook Step 3 meregenerasi indeks persis seperti arsip run. |
| Resource Bab III | Bab3.tex baris 174: 30 menit dan 90°C/30 detik | **Tertinggal** dari Step 2. Sinkronkan menjadi profil smoke/training, sensor tervalidasi, preflight dingin, dan batas guard kooperatif. |
| Budget Bab III | Bab3.tex baris 170: contoh 20 trial; total budget perlu ditetapkan | **Perlu pembaruan**: 4 jam aktif/model; N sama dari estimasi smoke; cap/minimum/status trial tercatat pada PREREGISTRATION. N final tetap menunggu Step 5. |
| Sensitivitas Bab III | Bab3.tex baris 78: scaffold/seed masih sebagai opsi umum | **Perlu spesifikasi**: development tetap; seeds 42/123/2026; grouped seed 42; rounds tetap untuk seluruh sensitivitas; tidak membangun ulang outer test. |
| Rancangan scaffold awal | Audit Step 3: early-stop fold 1 tidak memiliki kedua kelas untuk apple/winey | **Sudah diselesaikan pada protokol**: gabungkan train+stop, pakai rounds utama yang dibekukan, kontrol random dengan kebijakan sama; semua train/score final memiliki dukungan kedua kelas. |
| Seleksi kandidat | `finalize` memakai best COMPLETE Optuna trial, bukan baseline | Protokol menjadikan baseline pembanding terpisah. Jika tuned kalah dari baseline, laporkan tanpa klaim peningkatan; tie trial diverifikasi eksplisit sebelum fitting. |
| Batas kumulatif tuning | ResourceGuard membatasi satu sesi; CLI belum menegakkan total aktif 4 jam atau N lintas sesi | Gate Step 5/8/9: budget freeze dan ledger/runner dengan batas kumulatif sebelum/saat sesi. Jangan menganggap guard lama sudah otomatis menegakkan budget baru. |
| Runner sensitivitas | `initialize` memakai seed manifest dan `development_partitions` utama | Gate Step 11: runner development khusus membaca partisi yang dibekukan dan fixed rounds. Jangan menjalankan `build_dataset --seed/--split` karena dapat mengubah batas test utama serta target label. |
| Frontend asli | Branch main bersih; `src/services/InferenceService.ts` tidak ada pada checkout asli | Status source aplikasi terbaru harus dibedakan dari worktree audit. Rekonsiliasi frontend di luar scope sesi ini. |
| Frontend worktree audit | `src/services/InferenceService.ts` baris 10–11 memakai Gradio; baris 139–140 hanya menyertakan skor melewati threshold | Ini konteks read-only dari worktree, bukan bukti bahwa implementasi tersebut sudah ada pada main asli. Alur seluruh skor/selected dan REST tetap pekerjaan integrasi. |
| Feature schema | Root ML `feature_spec.json` memakai chirality=false; build baru memakai true | Pertahankan API/bundle lama sampai bundle baru lolos parity. Deployment baru wajib memakai feature spec bundle terpilih; tidak mengganti kontrak aktif pada Step 3. |

## Sumber yang diperiksa

- Backend: `C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Perfume-MultiLabel-Classifier`.
- Laporan: `C:\Users\ACER\Documents\Marvel\Skripsi\Laporan Skripsi\2521_skripsi_Marvel_Kevin_Nathanael\src\2.Isi\Bab3.tex`.
- Frontend asli: `C:\Users\ACER\Documents\Marvel\Skripsi\Project Skripsi\Essenza_Frontend`, branch main.
- Referensi layanan aplikasi: `C:\Users\ACER\Documents\ChatGPT\Project Skripsi\Essenza_Frontend\src\services\InferenceService.ts`.

## Batas keputusan

Sumber kebenaran eksperimen berikutnya adalah [PREREGISTRATION.md](../../PREREGISTRATION.md) dan protocol.json yang dibekukan. Penyelarasan naskah mengikuti tahap laporan; Bab III belum dianggap sudah diperbarui. Source model/CLI belum diubah pada Step 3. Kebutuhan tooling sebelum tahap berikutnya adalah gate pelaksanaan, bukan klaim fitur yang sudah selesai.

