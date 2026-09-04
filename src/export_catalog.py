"""Reproducible JSON catalog export from an explicitly supplied normalized CSV snapshot."""
import argparse
import json
import math
from pathlib import Path
import pandas as pd
from .artifacts import digest,write_json

def export_catalog(source,output,source_url,source_version):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError("Choose a new catalog directory")
    if not source_url.strip() or not source_version.strip():raise ValueError("Source URL and snapshot version are required")
    frame=pd.read_csv(source,dtype=str,keep_default_na=False)
    required={"pid","brand","name","gender","rating","accords"}
    if not required<=set(frame.columns):raise ValueError("Expected normalized columns: "+", ".join(sorted(required)))
    records=[];ids=set()
    for row in frame.to_dict("records"):
        pid=int(row["pid"]);rating=float(row["rating"])
        accords=json.loads(row["accords"])
        if pid in ids or not row["name"].strip() or not math.isfinite(rating) or not 0<=rating<=5:
            raise ValueError("Invalid catalog identity/name/rating")
        if not isinstance(accords,dict) or not accords or any(not isinstance(k,str) or not k.strip() or not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or not 0<=v<=1 for k,v in accords.items()):
            raise ValueError("Accords must be an explicit JSON object of finite 0..1 strengths")
        ids.add(pid)
        top=sorted(accords,key=lambda key:(-accords[key],key))[:3]
        records.append({"pid":pid,"brand":row["brand"].strip(),"name":row["name"].strip(),"gender":row["gender"].strip(),
                        "rating":str(rating),"accords":dict(sorted(accords.items())),"top_accords":", ".join(top)})
    if not records:raise ValueError("Empty catalog")
    output.mkdir(parents=True)
    write_json(output/"perfumes.json",sorted(records,key=lambda row:row["pid"]))
    write_json(output/"catalog_manifest.json",{"version":1,"source_url":source_url,"source_version":source_version,
        "source_sha256":digest(source),"sha256":digest(output/"perfumes.json"),"records":len(records),
        "accords":sorted({key for row in records for key in row["accords"]}),
        "taxonomy":"source catalog accords, independent of molecule ML labels","transformation":"explicit schema; no guessed note weights or substring label mapping"})
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    p.add_argument("--source-url",required=True);p.add_argument("--source-version",required=True)
    a=p.parse_args();print(export_catalog(a.input,a.output,a.source_url,a.source_version))
if __name__=="__main__":main()
