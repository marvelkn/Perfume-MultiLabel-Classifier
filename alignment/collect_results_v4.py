"""Collect complete or interrupted symmetric grouped-v4 results."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import zipfile

from .data import ROOT


def collect(root=ROOT):
    root = Path(root).resolve()
    run = root / "runs/perfume-five-grouped-v4"
    if not run.exists():
        raise FileNotFoundError("Belum ada hasil run grouped-v4.")
    if (run / ".alignment.lock").exists():
        raise RuntimeError("Training masih aktif. Tunggu atau hentikan proses sebelum koleksi.")
    folders = (
        run,
        root / "data/builds/perfume-five-grouped-v2",
        root / "alignment",
        root / "src",
        root / "reports/grouped_v4_20260912",
        root / "v4_seed",
    )
    files = {
        path for folder in folders for path in folder.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    files.update(root / name for name in (
        "requirements_training.txt",
        "config.yaml",
        "WINDOWS_GUIDE_V4.md",
        "RUN_GROUPED_V4.cmd",
        "RUN_GROUPED_V4.sh",
        "COLLECT_GROUPED_V4_RESULTS.cmd",
        "COLLECT_GROUPED_V4_RESULTS.sh",
    ))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Result inputs missing: {missing}")

    target = root / "campus-results"
    target.mkdir(exist_ok=True)
    archive_path = target / (
        "perfume-grouped-v4-results-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".zip"
    )
    manifest = {}
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            content = path.read_bytes()
            name = path.relative_to(root).as_posix()
            manifest[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
        archive.writestr("RESULTS_MANIFEST.json", json.dumps({
            "protocol": "perfume-five-grouped-v4",
            "complete": (run / "complete.json").exists(),
            "sha256": manifest,
        }, indent=2))
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP hasil rusak.")
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Checksum hasil tidak cocok: {name}")
    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{checksum}  {archive_path.name}\n", encoding="ascii"
    )
    return archive_path


if __name__ == "__main__":
    print(collect())
