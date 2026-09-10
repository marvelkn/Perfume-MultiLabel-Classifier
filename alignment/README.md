# Pipeline aktif: perfume-five-v1

Sumber, kamus label, split, dan fitur: perfume_config.json.
Preprocessing: perfume_data.py. Notebook: ../notebooks/02_perfume_five_preprocessing.ipynb.
Training XGBoost/LightGBM dan Optuna: experiments.py. Metrik: metrics.py.

Ikuti [Windows Guide](../WINDOWS_GUIDE.md). Dataset tersimpan di data/builds/perfume-five-v1.
Jumlah fitur/label dibaca dari manifest, bukan dari angka 25 atau konfigurasi API historis.
data.py dan protocol.json menyimpan audit Suh terdahulu; bukan sumber aktif lima dataset.
Pipeline lama src dan config.yaml dipertahankan untuk model/API historis; jangan pakai entrypoint lama
untuk eksperimen revisi ini. Bundle baru belum dipasang ke API/Android.
