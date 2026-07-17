import pandas as pd
import json
import os
import sqlite3

# Path dataset
DATA_DIR = r"C:\Users\Lenovo\Documents\UMN\Semester 7\AromaML\data\fragdb"
FRAGRANCES_CSV = os.path.join(DATA_DIR, "fragrances.csv")
ACCORDS_CSV = os.path.join(DATA_DIR, "accords.csv")
DB_PATH = os.path.join(DATA_DIR, "perfume_db.sqlite")

def run_etl():
    print("Mulai ETL Pipeline untuk AromaML B2C...")
    
    # 1. Load Accords Dictionary
    print("Membaca accords.csv...")
    try:
        df_accords = pd.read_csv(ACCORDS_CSV, sep='|', on_bad_lines='skip', encoding='utf-8')
    except UnicodeDecodeError:
        df_accords = pd.read_csv(ACCORDS_CSV, sep='|', on_bad_lines='skip', encoding='latin1')
        
    accord_dict = dict(zip(df_accords['id'], df_accords['name']))
    print(f"Ditemukan {len(accord_dict)} definisi accord.")

    # 2. Load Fragrances
    print("Membaca fragrances.csv...")
    try:
        df_frag = pd.read_csv(FRAGRANCES_CSV, sep='|', on_bad_lines='skip', encoding='utf-8')
    except UnicodeDecodeError:
        df_frag = pd.read_csv(FRAGRANCES_CSV, sep='|', on_bad_lines='skip', encoding='latin1')
        
    print(f"Total parfum yang di-load: {len(df_frag)}")
    
    # Filter kolom yang dibutuhkan
    cols_to_keep = ['pid', 'brand', 'name', 'year', 'gender', 'main_photo', 'accords', 'rating']
    cols_to_keep = [c for c in cols_to_keep if c in df_frag.columns]
    df_frag = df_frag[cols_to_keep].copy()

    # Drop baris yang tidak punya data accords
    df_frag.dropna(subset=['accords'], inplace=True)

    processed_data = []
    
    print("Memproses profil aroma...")
    for idx, row in df_frag.iterrows():
        acc_str = str(row['accords'])
        acc_pairs = acc_str.split(';')
        
        parsed_accords = {}
        for pair in acc_pairs:
            if ':' in pair:
                aid, val = pair.split(':')
                accord_name = accord_dict.get(aid, aid)
                try:
                    parsed_accords[accord_name] = float(val) / 100.0
                except ValueError:
                    pass
        
        # Sort accords by value descending
        sorted_acc = dict(sorted(parsed_accords.items(), key=lambda item: item[1], reverse=True))
        
        item = row.to_dict()
        item['accords_parsed'] = json.dumps(sorted_acc)
        top_3 = list(sorted_acc.keys())[:3]
        item['top_accords'] = ", ".join(top_3)
        processed_data.append(item)
        
    df_processed = pd.DataFrame(processed_data)
    
    # 3. Simpan ke SQLite
    print(f"Menyimpan ke {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    df_processed.to_sql('perfumes', conn, if_exists='replace', index=False)
    conn.close()
    
    print("ETL Selesai! Database siap digunakan oleh FastAPI.")

if __name__ == "__main__":
    run_etl()
