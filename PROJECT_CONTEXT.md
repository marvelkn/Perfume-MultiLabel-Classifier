# Project context

Sumber konteks aktif: [README.md](README.md) dan [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).
Dokumen ini menggantikan catatan lama yang memuat jumlah label, arsitektur offline/SQLite, atau hasil eksperimen yang sudah tidak sesuai.

- Molecule ML: Binary Relevance XGBoost dan LightGBM, Optuna studies terpisah.
- Feature API: RDKit online, schema versioned; mobile ONNX inference lokal.
- Explorer: katalog JSON dan AsyncStorage, taxonomy terpisah dari label molekul.
- Protokol baru: snapshot sumber, explicit identity joins, train-only label selection, threshold holdout, all-label inner CV, test freeze.
- Data historis/model lama dipertahankan. Build baru berada di direktori versioned.
- Laporan skripsi menunggu hasil eksperimen dan pengujian perangkat yang baru.
