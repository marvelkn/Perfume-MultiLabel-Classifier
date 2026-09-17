"""Portable v6 ZIP with prepared data, historical evidence, and verified hashes."""
from pathlib import Path
import hashlib
import json
import zipfile
from .data import ROOT

FOLDERS=('alignment','src','data/builds/perfume-five-grouped-v2','data/snapshots/perfume-five-v1','v4_seed','v5_seed','v6_reference','reports/grouped_v6_20260916')
FILES=('config.yaml','requirements_training.txt','WINDOWS_GUIDE_V6.md','LINUX_GUIDE_V6.md','RUN_GROUPED_V6.cmd','RUN_GROUPED_V6.sh','COLLECT_GROUPED_V6_RESULTS.cmd','COLLECT_GROUPED_V6_RESULTS.sh','README.md','SAFE_EXECUTION_PLAN.md','IMPLEMENTATION_STATUS.md','notebooks/02_perfume_five_preprocessing.ipynb','reports/campus_20260909/full/amendment.json')

def paths(root):
    found={p for name in FOLDERS for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.tmp') and p.name!='package_validation.json'}
    found.update(root/name for name in FILES)
    for p in found:
        if not p.is_file(): raise FileNotFoundError(p)
    return found

def build(root=ROOT):
    root=Path(root).resolve()
    target=root/'.local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v6.zip'
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.zip.tmp')
    hashes={}
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(paths(root)):
            name=path.relative_to(root).as_posix(); value=path.read_bytes()
            hashes[name]=hashlib.sha256(value).hexdigest(); archive.writestr(name,value)
        archive.writestr('CAMPUS_PACKAGE_MANIFEST.json',json.dumps({'protocol':'perfume-five-grouped-v6','dataset_included':True,'five_raw_sources_included':True,'fresh_optuna_studies':4,'candidates':8,'tuning_hours_per_study':8,'tuning_hours_total':32,'files':hashes},indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None: raise ValueError('ZIP corrupt.')
        for name,h in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=h: raise ValueError(name)
    temporary.replace(target)
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.zip.sha256').write_text(f'{digest}  {target.name}\n',encoding='ascii')
    result={'archive':str(target),'sha256':digest,'bytes':target.stat().st_size,'files':len(hashes)}
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__': build()
