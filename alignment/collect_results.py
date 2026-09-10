"""Collect complete or interrupted campus results without copying the environment."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import zipfile
from .data import ROOT

def collect(root=ROOT):
    root = Path(root).resolve()
    run = root / "runs/perfume-five-grouped-v2"
    if not run.exists():
        raise FileNotFoundError("No grouped-v2 run results yet.")
    if (run / ".alignment.lock").exists():
        raise RuntimeError("Stop/wait for training before collecting a consistent result archive.")
    folders = [run, root/"data/builds/perfume-five-grouped-v2", root/"alignment", root/"src",
               root/"reports/grouped_v2_20260910"]
    files = {p for folder in folders for p in folder.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc",)}
    files.update(root/n for n in ("requirements_training.txt","config.yaml","WINDOWS_GUIDE.md",
                                "RUN_PERFUME_FIVE.cmd","RUN_PERFUME_FIVE.sh",
                                "COLLECT_CAMPUS_RESULTS.cmd","COLLECT_CAMPUS_RESULTS.sh"))
    files.add(root/"reports/campus_20260909/full/amendment.json")
    files.add(root/"notebooks/02_perfume_five_preprocessing.ipynb")
    target = root/"campus-results"
    target.mkdir(exist_ok=True)
    path = target/("perfume-grouped-v2-results-"+datetime.now().strftime("%Y%m%d-%H%M%S-%f")+".zip")
    manifest = {}
    with zipfile.ZipFile(path,"x",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            content = p.read_bytes()
            name = p.relative_to(root).as_posix()
            manifest[name] = hashlib.sha256(content).hexdigest()
            z.writestr(name,content)
        z.writestr("RESULTS_MANIFEST.json",json.dumps({
            "complete":(run/"complete.json").exists(),"sha256":manifest},indent=2))
    with zipfile.ZipFile(path) as z:
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
    path.with_suffix(".zip.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest()+"  "+path.name+"\n")
    return path

if __name__=="__main__":
    print(collect())
