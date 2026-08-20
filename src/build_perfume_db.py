"""
build_perfume_db.py
====================
Mengunduh dataset parfum Kaggle (Fragrantica), membersihkannya,
memetakan 'main accords' ke 25 label Leffingwell kita, lalu
mengekspornya ke dataset/perfume_db.sqlite yang siap pakai di React Native.

Cara pakai:
    pip install kaggle pandas
    # Pastikan Kaggle API key sudah di-setup: ~/.kaggle/kaggle.json
    python src/build_perfume_db.py

Output:
    dataset/perfume_db.sqlite  <- siap dipindahkan ke React Native assets
"""

import json
import os
import re
import sqlite3
import sys
import zipfile
from pathlib import Path

import pandas as pd

# -- Config --------------------------------------------------------------------
DATASET_SLUG  = "nandini1999/perfume-recommendation-dataset"
RAW_ZIP_DIR   = Path("dataset/kaggle_raw")
OUT_DB        = Path("dataset/perfume_db.sqlite")
MIN_ACCORD_SCORE = 0.15   # Buang accord yang skornya < 15% (noise)

# -- Mapping: Fragrantica Accord -> Label Leffingwell kita --------------------
# Key: substring yang muncul di Fragrantica accord (lowercase)
# Value: label Leffingwell (harus salah satu dari 25 label kita)
ACCORD_MAP: dict[str, str] = {
    # Floral group
    "floral": "floral",
    "rose":   "rose",
    # Fruity / Citrus
    "fruity":  "fruity",
    "citrus":  "citrus",
    "orange":  "citrus",
    "lemon":   "citrus",
    "bergamot":"citrus",
    "grapefruit":"citrus",
    # Woody / Earthy
    "woody":   "woody",
    "wood":    "woody",
    "cedar":   "woody",
    "sandal":  "woody",
    "vetiver": "woody",
    "earthy":  "earthy",
    "mossy":   "earthy",
    # Fresh
    "fresh":   "fresh",
    "aquatic": "fresh",
    "marine":  "fresh",
    "ozonic":  "fresh",
    "clean":   "fresh",
    # Sweet / Vanilla / Balsamic
    "sweet":    "sweet",
    "vanilla":  "vanilla",
    "caramel":  "caramellic",
    "gourmand": "sweet",
    "chocolate":"cocoa",
    "honey":    "honey",
    "balsamic": "balsamic",
    "amber":    "balsamic",
    "resin":    "balsamic",
    # Spicy / Herbal / Aromatic
    "spicy":    "spicy",
    "warm spicy":"spicy",
    "cinnamon": "spicy",
    "pepper":   "spicy",
    "herbal":   "herbal",
    "aromatic": "aromatic",
    "green":    "green",
    "mint":     "mint",
    # Musky / Powdery / Anisic
    "musky":    "musky",
    "musk":     "musky",
    "powdery":  "powdery",
    "talc":     "powdery",
    "anisic":   "anisic",
    "licorice": "anisic",
    # Smoky / Tobacco / Aldehydic
    "smoky":    "smoky",
    "tobacco":  "tobacco",
    "aldehydic":"aldehydic",
    # Winey / Waxy / Fatty
    "winey":    "winey",
    "wine":     "winey",
    "waxy":     "waxy",
    "fatty":    "fatty",
    "buttery":  "fatty",
}

# 25 label final kita (dari model XGBoost)
VALID_LABELS = {
    "floral", "fruity", "woody", "sweet", "citrus", "aromatic",
    "musky", "fresh", "spicy", "balsamic", "vanilla", "powdery",
    "earthy", "smoky", "tobacco", "anisic", "aldehydic", "rose",
    "green", "herbal", "mint", "caramellic", "cocoa", "honey",
    "winey", "fatty", "waxy"
}


# -- Step 1: Download dari Kaggle ----------------------------------------------
def download_kaggle_dataset():
    RAW_ZIP_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] Mengunduh dataset dari Kaggle...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(DATASET_SLUG, path=str(RAW_ZIP_DIR), unzip=True)
        print(f"  Dataset berhasil diunduh ke: {RAW_ZIP_DIR}/")
    except ImportError:
        print("  [ERROR] kaggle package belum terinstall.")
        print("  Jalankan: pip install kaggle")
        print("  Lalu pastikan ~/.kaggle/kaggle.json sudah ada.")
        sys.exit(1)
    except Exception as e:
        print(f"  [ERROR] Gagal download: {e}")
        print("  Pastikan ~/.kaggle/kaggle.json sudah dikonfigurasi.")
        print("  Cek: https://www.kaggle.com/settings -> API -> Create New Token")
        sys.exit(1)


# -- Step 2: Baca & Bersihkan CSV ----------------------------------------------
def load_and_clean() -> pd.DataFrame:
    print("[2/4] Membaca dan membersihkan CSV...")
    csv_files = list(RAW_ZIP_DIR.rglob("*.csv"))
    if not csv_files:
        print(f"  [ERROR] Tidak ada file CSV di {RAW_ZIP_DIR}/")
        print(f"  Isi direktori: {list(RAW_ZIP_DIR.iterdir())}")
        sys.exit(1)

    print(f"  Ditemukan: {[f.name for f in csv_files]}")
    # Ambil file CSV terbesar (biasanya yang paling lengkap)
    # Coba beberapa encoding — Fragrantica sering punya karakter aksen Prancis
    target_csv = max(csv_files, key=lambda f: f.stat().st_size)
    for enc in ["utf-8", "latin1", "cp1252", "iso-8859-1"]:
        try:
            df = pd.read_csv(target_csv, low_memory=False, encoding=enc)
            print(f"  Encoding berhasil: {enc}")
            break
        except UnicodeDecodeError:
            continue
    else:
        print("  [ERROR] Tidak ada encoding yang berhasil membaca CSV.")
        sys.exit(1)
    print(f"  Raw rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # Normalisasi nama kolom
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# -- Step 3: Petakan Accords ke Label Kita ------------------------------------
def parse_accords(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/4] Memetakan accords ke label Leffingwell...")

    # Deteksi kolom accord/notes
    accord_col = None
    for candidate in ["main_accords", "accords", "notes", "main_accord", "accord", "scent_notes"]:
        if candidate in df.columns:
            accord_col = candidate
            break
    if accord_col is None:
        matches = [c for c in df.columns if "accord" in c or "note" in c]
        accord_col = matches[0] if matches else df.columns[-1]
    print(f"  Kolom accord: '{accord_col}'")

    # Deteksi kolom lainnya
    name_col   = next((c for c in df.columns if c in ["name", "perfume_name", "title", "fragrance_name", "perfume"]), df.columns[0])
    brand_col  = next((c for c in df.columns if c in ["brand", "designer", "house", "company"]), None)
    gender_col = next((c for c in df.columns if c in ["gender", "for_gender", "for"]), None)
    rating_col = next((c for c in df.columns if c in ["rating", "average_rating", "score"]), None)

    results = []
    skipped = 0

    for _, row in df.iterrows():
        raw_accord = str(row.get(accord_col, "") or "")
        if not raw_accord or raw_accord == "nan":
            skipped += 1
            continue

        mapped: dict[str, float] = {}
        raw_lower = raw_accord.lower()

        # Format 1: JSON string
        if "{" in raw_lower:
            try:
                parsed = json.loads(raw_accord)
                for frag_key, score in parsed.items():
                    frag_key_l = str(frag_key).lower().strip()
                    for substr, leff_label in ACCORD_MAP.items():
                        if substr in frag_key_l and leff_label in VALID_LABELS:
                            mapped[leff_label] = max(mapped.get(leff_label, 0.0), float(score))
            except Exception:
                pass

        # Format 2: Teks biasa / list terpisah koma
        if not mapped:
            tokens = re.split(r"[,\n;|]+", raw_lower)
            for i, token in enumerate(tokens):
                token = token.strip()
                base_score = max(0.2, 1.0 - i * 0.12)
                for substr, leff_label in ACCORD_MAP.items():
                    if substr in token and leff_label in VALID_LABELS:
                        if leff_label not in mapped:
                            mapped[leff_label] = base_score

        if not mapped:
            skipped += 1
            continue

        mapped = {k: v for k, v in mapped.items() if v >= MIN_ACCORD_SCORE}
        if not mapped:
            skipped += 1
            continue

        sorted_accords = dict(sorted(mapped.items(), key=lambda x: x[1], reverse=True))
        top_3 = ", ".join(list(sorted_accords.keys())[:3])

        results.append({
            "brand":          str(row.get(brand_col, "") if brand_col else "Unknown").strip(),
            "name":           str(row.get(name_col, "Unknown")).strip(),
            "gender":         str(row.get(gender_col, "") if gender_col else "").strip().lower(),
            "rating":         str(row.get(rating_col, "") if rating_col else ""),
            "accords":        json.dumps(sorted_accords),
            "accords_parsed": json.dumps(sorted_accords),
            "top_accords":    top_3,
            "is_custom":      0,
        })

    print(f"  Berhasil dipetakan: {len(results):,} parfum")
    print(f"  Dilewati (no accord): {skipped:,}")
    return pd.DataFrame(results)


# -- Step 4: Export ke SQLite --------------------------------------------------
def export_to_sqlite(df: pd.DataFrame):
    print(f"[4/4] Mengeksport {len(df):,} parfum ke {OUT_DB} ...")
    OUT_DB.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(OUT_DB)
    c = conn.cursor()

    # Tabel perfumes (data bawaan dari Kaggle)
    c.execute("DROP TABLE IF EXISTS perfumes")
    c.execute("""
        CREATE TABLE perfumes (
            pid            INTEGER PRIMARY KEY AUTOINCREMENT,
            brand          TEXT,
            name           TEXT NOT NULL,
            gender         TEXT,
            rating         TEXT,
            accords        TEXT,
            accords_parsed TEXT,
            top_accords    TEXT,
            is_custom      INTEGER DEFAULT 0
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_perfumes_name ON perfumes(name)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_perfumes_brand ON perfumes(brand)")

    # Tabel user_perfumes untuk CRUD
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_perfumes (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT NOT NULL,
            accords        TEXT NOT NULL,
            accords_parsed TEXT NOT NULL,
            top_accords    TEXT,
            mode           TEXT DEFAULT 'simple',
            notes          TEXT,
            created_at     TEXT DEFAULT (datetime('now')),
            updated_at     TEXT DEFAULT (datetime('now'))
        )
    """)

    # Insert data parfum bawaan
    df.to_sql("perfumes", conn, if_exists="append", index=False)
    conn.commit()

    c.execute("SELECT COUNT(*) FROM perfumes")
    n_perfumes = c.fetchone()[0]
    size_kb = OUT_DB.stat().st_size / 1024
    conn.close()

    print(f"  perfumes     : {n_perfumes:,} baris")
    print(f"  user_perfumes: 0 baris (siap CRUD)")
    print(f"  Ukuran DB    : {size_kb:.1f} KB")
    print(f"  Output: {OUT_DB.resolve()}")
    print(f"\n  Langkah berikutnya:")
    print(f"  Salin ke: Essenza_Frontend/android/app/src/main/assets/perfume_db.sqlite")


# -- Main ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("BUILD PERFUME DATABASE (Kaggle -> SQLite)")
    print("=" * 60)
    download_kaggle_dataset()
    raw_df   = load_and_clean()
    clean_df = parse_accords(raw_df)
    export_to_sqlite(clean_df)
    print("=" * 60)
    print("SELESAI!")
    print("=" * 60)
