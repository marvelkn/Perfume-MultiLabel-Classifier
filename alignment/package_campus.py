"""Build and verify the portable Windows/Linux campus package."""
from pathlib import Path
import hashlib
import json
import zipfile

from .data import ROOT


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def build(root=ROOT):
    root = Path(root).resolve()
    destination = root / ".local-tools/campus-transfer-full-20260909/perfume-campus-grouped-v2.zip"
    temporary = destination.with_suffix(".zip.tmp")
    folders = [
        "alignment",
        "src",
        "data/builds/perfume-five-grouped-v2",
        "data/snapshots/perfume-five-v1",
        "reports/grouped_v2_20260910",
    ]
    files = {
        path
        for folder in folders
        for path in (root / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    files.update(root / name for name in (
        "config.yaml",
        "requirements_training.txt",
        "WINDOWS_GUIDE.md",
        "RUN_PERFUME_FIVE.cmd",
        "RUN_PERFUME_FIVE.sh",
        "COLLECT_CAMPUS_RESULTS.cmd",
        "COLLECT_CAMPUS_RESULTS.sh",
        "README.md",
        "SAFE_EXECUTION_PLAN.md",
        "IMPLEMENTATION_STATUS.md",
        "notebooks/02_perfume_five_preprocessing.ipynb",
        "reports/campus_20260909/full/amendment.json",
    ))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Package inputs missing: {missing}")

    hashes = {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            content = path.read_bytes()
            name = path.relative_to(root).as_posix()
            hashes[name] = digest_bytes(content)
            archive.writestr(name, content)
        archive.writestr("CAMPUS_PACKAGE_MANIFEST.json", json.dumps({
            "protocol": "perfume-five-grouped-v2",
            "supported_systems": ["Windows", "Linux"],
            "dataset_included": True,
            "raw_five_sources_included": True,
            "virtual_environment_included": False,
            "files": hashes,
        }, indent=2))

    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt campus ZIP.")
        for name, expected in hashes.items():
            if digest_bytes(archive.read(name)) != expected:
                raise ValueError(f"ZIP checksum mismatch: {name}")
    temporary.replace(destination)
    checksum = digest_bytes(destination.read_bytes())
    checksum_path = destination.with_suffix(".zip.sha256")
    checksum_path.write_text(f"{checksum}  {destination.name}\n", encoding="ascii")
    return {"archive": str(destination), "sha256": checksum, "files": len(hashes),
            "bytes": destination.stat().st_size, "supported_systems": ["Windows", "Linux"]}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
