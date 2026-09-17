"""Collect frozen complete or interrupted v6 results after the worker stops."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import zipfile
from .data import ROOT
from .package_campus_v6 import paths
from . import experiments as core
from .experiments_v6 import verify_files

def collect(root=ROOT):
    root=Path(root).resolve(); run=root/'runs/perfume-five-grouped-v6'
    if not run.exists(): raise FileNotFoundError('Belum ada run v6.')
    if (run/'.alignment.lock').exists(): raise RuntimeError('Tunggu training selesai. Setelah crash, gunakan launcher lagi untuk memulihkan lock.')
    complete=(run/'complete.json').exists() and (run/'completion_manifest.json').exists()
    if complete: verify_files(run,core.read(run/'completion_manifest.json')['files'])
    files=paths(root)|{p for p in run.rglob('*') if p.is_file() and p.suffix!='.tmp'}
    target=root/'campus-results'; target.mkdir(exist_ok=True)
    path=target/('perfume-grouped-v6-results-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.zip')
    temporary=path.with_suffix('.zip.tmp'); hashes={}
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):
            name=p.relative_to(root).as_posix(); content=p.read_bytes()
            hashes[name]=hashlib.sha256(content).hexdigest(); z.writestr(name,content)
        z.writestr('RESULTS_MANIFEST.json',json.dumps({'protocol':'perfume-five-grouped-v6','complete':complete,'sha256':hashes},indent=2))
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None: raise ValueError('ZIP hasil rusak.')
        for n,h in hashes.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h: raise ValueError(n)
    temporary.replace(path)
    path.with_suffix('.zip.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+f'  {path.name}\n',encoding='ascii')
    print(path)
    return path

if __name__=='__main__': collect()
