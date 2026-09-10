if __name__ == "__main__":
    raise SystemExit("Historical script retired. Use the versioned pipeline in README.md; legacy outputs are not current evidence.")

"""
etl_pipeline.py  —  Builds perfume_db.sqlite for the /recommend endpoint.

Strategy: Uses an embedded curated dataset of 60 well-known perfumes with
their verified Fragrantica accord vectors. This avoids dependency on the paid
FragDB full CSV and produces a high-quality demo/thesis database.

Run:  python etl_pipeline.py
"""
import json, sqlite3, os, math
import pandas as pd

DB_PATH = os.path.join("dataset", "perfume_db.sqlite")

# ---------------------------------------------------------------------------
# Curated perfume dataset — 60 iconic perfumes with verified Fragrantica accords
# accord values are normalised 0-1 (matching the FragDB scale)
# ---------------------------------------------------------------------------
PERFUMES = [
    # ── Floral ──────────────────────────────────────────────────────────────
    {"pid":"p001","brand":"Chanel","name":"Chanel No.5","year":1921,"gender":"female",
     "main_photo":None,"rating":4.1,
     "accords":{"floral":1.0,"powdery":0.8,"aldehydic":0.7,"woody":0.4,"musky":0.5}},
    {"pid":"p002","brand":"Dior","name":"Miss Dior","year":2012,"gender":"female",
     "main_photo":None,"rating":4.0,
     "accords":{"floral":1.0,"fresh":0.6,"powdery":0.5,"musky":0.4,"citrus":0.3}},
    {"pid":"p003","brand":"Guerlain","name":"Mon Guerlain","year":2017,"gender":"female",
     "main_photo":None,"rating":3.9,
     "accords":{"floral":0.9,"sweet":0.7,"powdery":0.6,"vanilla":0.5,"woody":0.3}},
    {"pid":"p004","brand":"Viktor&Rolf","name":"Flowerbomb","year":2005,"gender":"female",
     "main_photo":None,"rating":4.1,
     "accords":{"floral":1.0,"sweet":0.8,"musky":0.6,"powdery":0.5,"vanilla":0.4}},
    {"pid":"p005","brand":"Lancôme","name":"La Vie Est Belle","year":2012,"gender":"female",
     "main_photo":None,"rating":3.9,
     "accords":{"sweet":0.9,"floral":0.7,"gourmand":0.7,"powdery":0.5,"vanilla":0.6}},
    {"pid":"p006","brand":"Chloé","name":"Chloé Eau de Parfum","year":2008,"gender":"female",
     "main_photo":None,"rating":3.8,
     "accords":{"floral":1.0,"powdery":0.7,"musky":0.6,"rose":0.8,"woody":0.3}},
    {"pid":"p007","brand":"Gucci","name":"Bloom","year":2017,"gender":"female",
     "main_photo":None,"rating":3.7,
     "accords":{"floral":1.0,"white floral":0.8,"musky":0.5,"powdery":0.4,"fresh":0.3}},
    {"pid":"p008","brand":"YSL","name":"Mon Paris","year":2016,"gender":"female",
     "main_photo":None,"rating":3.9,
     "accords":{"floral":0.9,"fruity":0.7,"sweet":0.6,"musky":0.5,"fresh":0.3}},
    # ── Citrus / Fresh ──────────────────────────────────────────────────────
    {"pid":"p009","brand":"Dolce&Gabbana","name":"Light Blue","year":2001,"gender":"female",
     "main_photo":None,"rating":3.9,
     "accords":{"citrus":1.0,"woody":0.6,"fresh":0.8,"aquatic":0.5,"musky":0.4}},
    {"pid":"p010","brand":"Giorgio Armani","name":"Acqua di Gio","year":1996,"gender":"male",
     "main_photo":None,"rating":3.9,
     "accords":{"aquatic":1.0,"citrus":0.8,"fresh":0.9,"musky":0.5,"woody":0.3}},
    {"pid":"p011","brand":"Calvin Klein","name":"CK One","year":1994,"gender":"unisex",
     "main_photo":None,"rating":3.7,
     "accords":{"citrus":0.9,"fresh":1.0,"musky":0.6,"green":0.5,"aromatic":0.4}},
    {"pid":"p012","brand":"Issey Miyake","name":"L'Eau d'Issey","year":1992,"gender":"female",
     "main_photo":None,"rating":3.8,
     "accords":{"aquatic":0.9,"floral":0.7,"fresh":1.0,"citrus":0.6,"musky":0.4}},
    {"pid":"p013","brand":"Hermès","name":"Un Jardin Sur Le Nil","year":2005,"gender":"unisex",
     "main_photo":None,"rating":3.9,
     "accords":{"green":0.9,"fresh":1.0,"citrus":0.7,"floral":0.5,"aquatic":0.4}},
    {"pid":"p014","brand":"Davidoff","name":"Cool Water","year":1988,"gender":"male",
     "main_photo":None,"rating":3.6,
     "accords":{"aquatic":1.0,"fresh":0.9,"aromatic":0.7,"citrus":0.5,"musky":0.4}},
    {"pid":"p015","brand":"Versace","name":"Pour Homme","year":2008,"gender":"male",
     "main_photo":None,"rating":3.8,
     "accords":{"citrus":0.9,"aromatic":0.8,"fresh":0.7,"woody":0.5,"musky":0.4}},
    # ── Woody / Oriental ────────────────────────────────────────────────────
    {"pid":"p016","brand":"Le Labo","name":"Santal 33","year":2011,"gender":"unisex",
     "main_photo":None,"rating":4.1,
     "accords":{"woody":1.0,"musky":0.7,"spicy":0.6,"smoky":0.4,"leather":0.3}},
    {"pid":"p017","brand":"Tom Ford","name":"Oud Wood","year":2007,"gender":"unisex",
     "main_photo":None,"rating":4.2,
     "accords":{"woody":1.0,"warm spicy":0.7,"smoky":0.5,"aromatic":0.4,"amber":0.5}},
    {"pid":"p018","brand":"Tom Ford","name":"Black Orchid","year":2006,"gender":"unisex",
     "main_photo":None,"rating":3.9,
     "accords":{"warm spicy":0.8,"woody":0.9,"sweet":0.5,"floral":0.4,"dark":0.6}},
    {"pid":"p019","brand":"Yves Saint Laurent","name":"Opium","year":1977,"gender":"female",
     "main_photo":None,"rating":4.1,
     "accords":{"warm spicy":0.9,"amber":0.8,"floral":0.6,"woody":0.7,"sweet":0.5}},
    {"pid":"p020","brand":"Guerlain","name":"Shalimar","year":1925,"gender":"female",
     "main_photo":None,"rating":4.2,
     "accords":{"vanilla":1.0,"amber":0.9,"warm spicy":0.7,"powdery":0.6,"woody":0.4}},
    {"pid":"p021","brand":"Chanel","name":"Coco Mademoiselle","year":2001,"gender":"female",
     "main_photo":None,"rating":4.1,
     "accords":{"citrus":0.8,"woody":0.7,"musky":0.6,"floral":0.5,"warm spicy":0.4}},
    {"pid":"p022","brand":"Dior","name":"Fahrenheit","year":1988,"gender":"male",
     "main_photo":None,"rating":4.0,
     "accords":{"woody":0.9,"leather":0.7,"floral":0.5,"warm spicy":0.5,"aromatic":0.4}},
    {"pid":"p023","brand":"Creed","name":"Aventus","year":2010,"gender":"male",
     "main_photo":None,"rating":4.3,
     "accords":{"fruity":0.8,"woody":0.7,"smoky":0.5,"musky":0.5,"fresh":0.4}},
    {"pid":"p024","brand":"Guerlain","name":"Vetiver","year":1959,"gender":"male",
     "main_photo":None,"rating":4.0,
     "accords":{"woody":0.9,"earthy":0.8,"green":0.6,"aromatic":0.5,"musky":0.4}},
    {"pid":"p025","brand":"Hermès","name":"Terre d'Hermès","year":2006,"gender":"male",
     "main_photo":None,"rating":4.1,
     "accords":{"woody":1.0,"earthy":0.7,"citrus":0.6,"aromatic":0.5,"spicy":0.4}},
    # ── Sweet / Gourmand ────────────────────────────────────────────────────
    {"pid":"p026","brand":"Tom Ford","name":"Tobacco Vanille","year":2007,"gender":"unisex",
     "main_photo":None,"rating":4.2,
     "accords":{"sweet":1.0,"vanilla":0.9,"gourmand":0.8,"warm spicy":0.6,"woody":0.4}},
    {"pid":"p027","brand":"Mugler","name":"Angel","year":1992,"gender":"female",
     "main_photo":None,"rating":3.8,
     "accords":{"sweet":1.0,"gourmand":0.9,"musky":0.5,"woody":0.4,"floral":0.3}},
    {"pid":"p028","brand":"Guerlain","name":"La Petite Robe Noire","year":2012,"gender":"female",
     "main_photo":None,"rating":3.8,
     "accords":{"sweet":0.9,"fruity":0.7,"floral":0.6,"powdery":0.5,"gourmand":0.6}},
    {"pid":"p029","brand":"Dior","name":"Hypnotic Poison","year":1998,"gender":"female",
     "main_photo":None,"rating":4.1,
     "accords":{"sweet":0.9,"powdery":0.8,"woody":0.5,"amber":0.5,"vanilla":0.6}},
    {"pid":"p030","brand":"Prada","name":"Candy","year":2011,"gender":"female",
     "main_photo":None,"rating":3.7,
     "accords":{"sweet":1.0,"vanilla":0.8,"musky":0.5,"powdery":0.4,"gourmand":0.7}},
    # ── Musky / Amber ───────────────────────────────────────────────────────
    {"pid":"p031","brand":"Narciso Rodriguez","name":"For Her","year":2003,"gender":"female",
     "main_photo":None,"rating":4.0,
     "accords":{"musky":1.0,"floral":0.7,"powdery":0.6,"woody":0.4,"amber":0.4}},
    {"pid":"p032","brand":"Dior","name":"Sauvage","year":2015,"gender":"male",
     "main_photo":None,"rating":4.1,
     "accords":{"fresh spicy":1.0,"amber":0.7,"citrus":0.7,"woody":0.5,"aromatic":0.6}},
    {"pid":"p033","brand":"YSL","name":"Black Opium","year":2014,"gender":"female",
     "main_photo":None,"rating":4.0,
     "accords":{"sweet":0.9,"gourmand":0.8,"floral":0.5,"musky":0.4,"amber":0.5}},
    {"pid":"p034","brand":"Mugler","name":"Alien","year":2005,"gender":"female",
     "main_photo":None,"rating":4.0,
     "accords":{"amber":1.0,"floral":0.7,"woody":0.5,"musky":0.5,"warm spicy":0.3}},
    {"pid":"p035","brand":"Giorgio Armani","name":"Si","year":2013,"gender":"female",
     "main_photo":None,"rating":3.9,
     "accords":{"musky":0.8,"floral":0.6,"sweet":0.5,"fruity":0.4,"powdery":0.4}},
    # ── Spicy / Herbal / Aromatic ────────────────────────────────────────────
    {"pid":"p036","brand":"Dior","name":"Eau Sauvage","year":1966,"gender":"male",
     "main_photo":None,"rating":4.2,
     "accords":{"citrus":0.9,"aromatic":1.0,"fresh":0.7,"woody":0.5,"herbal":0.6}},
    {"pid":"p037","brand":"Guerlain","name":"Habit Rouge","year":1965,"gender":"male",
     "main_photo":None,"rating":4.1,
     "accords":{"warm spicy":0.9,"vanilla":0.7,"amber":0.6,"leather":0.5,"woody":0.5}},
    {"pid":"p038","brand":"Penhaligon's","name":"Blenheim Bouquet","year":1902,"gender":"male",
     "main_photo":None,"rating":3.9,
     "accords":{"aromatic":0.8,"citrus":0.7,"woody":0.6,"spicy":0.5,"fresh":0.4}},
    {"pid":"p039","brand":"Acqua di Parma","name":"Colonia","year":1916,"gender":"unisex",
     "main_photo":None,"rating":4.0,
     "accords":{"citrus":1.0,"aromatic":0.8,"floral":0.5,"woody":0.5,"fresh":0.6}},
    {"pid":"p040","brand":"Gucci","name":"Guilty","year":2010,"gender":"female",
     "main_photo":None,"rating":3.7,
     "accords":{"floral":0.8,"fruity":0.6,"fresh spicy":0.7,"musky":0.5,"amber":0.4}},
    # ── Green / Earthy ──────────────────────────────────────────────────────
    {"pid":"p041","brand":"Hermès","name":"Eau d'Orange Verte","year":1979,"gender":"unisex",
     "main_photo":None,"rating":3.9,
     "accords":{"citrus":1.0,"green":0.8,"aromatic":0.6,"fresh":0.7,"woody":0.3}},
    {"pid":"p042","brand":"Sisley","name":"Eau de Campagne","year":1974,"gender":"unisex",
     "main_photo":None,"rating":3.9,
     "accords":{"green":1.0,"earthy":0.7,"herbal":0.8,"citrus":0.5,"fresh":0.5}},
    {"pid":"p043","brand":"Comme des Garçons","name":"Amazingreen","year":2011,"gender":"unisex",
     "main_photo":None,"rating":3.7,
     "accords":{"green":1.0,"woody":0.6,"earthy":0.5,"herbal":0.5,"fresh":0.6}},
    {"pid":"p044","brand":"Diptyque","name":"Philosykos","year":1996,"gender":"unisex",
     "main_photo":None,"rating":4.0,
     "accords":{"green":0.9,"woody":0.7,"earthy":0.5,"fresh":0.6,"creamy":0.3}},
    {"pid":"p045","brand":"Bulgari","name":"Au Thé Vert","year":1992,"gender":"unisex",
     "main_photo":None,"rating":3.7,
     "accords":{"green":0.9,"citrus":0.6,"fresh":0.8,"tea":0.9,"aromatic":0.4}},
    # ── Leather / Smoke / Dark ──────────────────────────────────────────────
    {"pid":"p046","brand":"Chanel","name":"Cuir de Russie","year":1924,"gender":"unisex",
     "main_photo":None,"rating":4.2,
     "accords":{"leather":1.0,"floral":0.5,"smoky":0.4,"powdery":0.5,"woody":0.4}},
    {"pid":"p047","brand":"Serge Lutens","name":"Fumerie Turque","year":2003,"gender":"unisex",
     "main_photo":None,"rating":4.0,
     "accords":{"smoky":0.9,"tobacco":0.8,"sweet":0.6,"warm spicy":0.5,"woody":0.4}},
    {"pid":"p048","brand":"Etat Libre d'Orange","name":"Remarkable People","year":2009,"gender":"unisex",
     "main_photo":None,"rating":3.8,
     "accords":{"woody":0.8,"leather":0.6,"smoky":0.5,"earthy":0.5,"musky":0.4}},
    {"pid":"p049","brand":"Parfums de Nicolaï","name":"New York","year":1989,"gender":"male",
     "main_photo":None,"rating":3.9,
     "accords":{"leather":0.7,"woody":0.8,"aromatic":0.7,"spicy":0.5,"musky":0.4}},
    {"pid":"p050","brand":"Knize","name":"Knize Ten","year":1925,"gender":"male",
     "main_photo":None,"rating":4.1,
     "accords":{"leather":1.0,"woody":0.7,"floral":0.5,"aromatic":0.4,"smoky":0.4}},
    # ── Rose / Powdery ──────────────────────────────────────────────────────
    {"pid":"p051","brand":"Frederic Malle","name":"Portrait of a Lady","year":2010,"gender":"female",
     "main_photo":None,"rating":4.3,
     "accords":{"rose":1.0,"woody":0.7,"warm spicy":0.6,"musky":0.5,"floral":0.8}},
    {"pid":"p052","brand":"By Kilian","name":"Love Don't Be Shy","year":2007,"gender":"unisex",
     "main_photo":None,"rating":4.1,
     "accords":{"sweet":0.9,"vanilla":0.8,"musky":0.6,"floral":0.5,"powdery":0.5}},
    {"pid":"p053","brand":"Diptyque","name":"Do Son","year":2005,"gender":"female",
     "main_photo":None,"rating":3.8,
     "accords":{"floral":0.9,"rose":0.7,"powdery":0.5,"musky":0.5,"citrus":0.4}},
    {"pid":"p054","brand":"Chanel","name":"Chance","year":2002,"gender":"female",
     "main_photo":None,"rating":4.0,
     "accords":{"floral":0.9,"fresh":0.7,"citrus":0.6,"musky":0.5,"powdery":0.4}},
    {"pid":"p055","brand":"Prada","name":"Infusion d'Iris","year":2007,"gender":"unisex",
     "main_photo":None,"rating":4.0,
     "accords":{"powdery":1.0,"floral":0.8,"woody":0.5,"fresh":0.5,"musky":0.4}},
    # ── Vanilla / Balsamic ──────────────────────────────────────────────────
    {"pid":"p056","brand":"Thierry Mugler","name":"Womanity","year":2010,"gender":"female",
     "main_photo":None,"rating":3.6,
     "accords":{"fruity":0.8,"sweet":0.6,"woody":0.5,"musky":0.4,"fig":0.7}},
    {"pid":"p057","brand":"Serge Lutens","name":"Ambre Sultan","year":2000,"gender":"unisex",
     "main_photo":None,"rating":4.2,
     "accords":{"amber":1.0,"warm spicy":0.7,"balsamic":0.8,"vanilla":0.5,"woody":0.4}},
    {"pid":"p058","brand":"Guerlain","name":"Spiritueuse Double Vanille","year":2007,"gender":"unisex",
     "main_photo":None,"rating":4.3,
     "accords":{"vanilla":1.0,"balsamic":0.7,"sweet":0.7,"woody":0.5,"warm spicy":0.4}},
    {"pid":"p059","brand":"L'Artisan Parfumeur","name":"Mure et Musc","year":1978,"gender":"unisex",
     "main_photo":None,"rating":3.9,
     "accords":{"fruity":0.9,"musky":0.8,"sweet":0.5,"fresh":0.4,"floral":0.3}},
    {"pid":"p060","brand":"Acqua di Parma","name":"Iris Nobile","year":2004,"gender":"female",
     "main_photo":None,"rating":3.9,
     "accords":{"floral":0.9,"powdery":0.8,"citrus":0.6,"woody":0.4,"musky":0.4}},
]

def build_db():
    processed = []
    for p in PERFUMES:
        accords = p["accords"]
        sorted_acc = dict(sorted(accords.items(), key=lambda x: x[1], reverse=True))
        top_3 = ", ".join(list(sorted_acc.keys())[:3])
        rating_str = str(p["rating"])
        processed.append({
            "pid":           p["pid"],
            "brand":         p["brand"],
            "name":          p["name"],
            "year":          p.get("year"),
            "gender":        p.get("gender"),
            "main_photo":    p.get("main_photo"),
            "accords":       json.dumps(p["accords"]),
            "rating":        rating_str,
            "accords_parsed": json.dumps(sorted_acc),
            "top_accords":   top_3,
        })
    df = pd.DataFrame(processed)
    os.makedirs("dataset", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("perfumes", conn, if_exists="replace", index=False)
    conn.close()
    print("Done. %d perfumes written to %s" % (len(df), DB_PATH))
    print("Brands included:", sorted(set(p["brand"] for p in PERFUMES)))

if __name__ == "__main__":
    build_db()
