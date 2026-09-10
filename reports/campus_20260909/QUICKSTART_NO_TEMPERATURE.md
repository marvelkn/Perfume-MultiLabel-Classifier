# Quickstart terbaru: seluruh pipeline ML

Gunakan **WINDOWS_GUIDE.md versi full pipeline** dalam folder ini. Paket yang benar adalah `campus-transfer-full-20260909`.

Setelah clone, ekstraksi paket full dan instalasi dependensi:

```powershell
.\RUN_CAMPUS_ALL.cmd
```

Baseline, tuning kedua model, final fit, sensitivitas, evaluasi, ekspor, dan ZIP hasil berjalan otomatis. Tidak perlu smoke atau sensor suhu. Target 30 attempted trials dan cap 4 jam aktif per model; waktu total bisa lebih dari 8 jam. Hasil untuk dikirim kembali ada di `campus-results`.
