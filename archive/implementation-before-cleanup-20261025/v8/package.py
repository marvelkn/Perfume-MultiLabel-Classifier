"""Package only V8 preparation, original data, and audit evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from .prepare import ROOT, CONFIG, LOCK, digest, read_json, verify

BASENAME = "perfume-campus-v8-preml-m2048-d8"
FILES = [
    "RUN_PREML_V8.cmd", "RUN_PREML_V8.sh",
    "WINDOWS_GUIDE_V8.md", "LINUX_GUIDE_V8.md", "V8_PREML_PLAN.md",
    "requirements_preml_v8.txt", "requirements_notebook_v8.txt",
    "v8/__init__.py", "v8/prepare.py", "v8/test_preml.py",
    "v8/protocol.json", "v8/source_lock.json", "v8/package.py",
    "notebooks/04_suh_v8_preml.ipynb",
    "data/reference/suh2025/author_code.ipynb",
    "data/reference/suh2025/Dataset1_data_preparation.xlsx",
    "data/reference/suh2025/Dataset2_final_w_molblock.csv",
    "data/reference/suh2025/download_manifest.json",
    "data/reference/suh2025/license.json",
]


def build(target=None):
    manifest = verify()
    output = ROOT / read_json(CONFIG)["output"]
    target = Path(target) if target else (
        ROOT / ".local-tools/campus-transfer-full-20260909" / (BASENAME + ".zip"))
    if target.exists() or target.with_suffix(".zip.tmp").exists():
        raise FileExistsError("Archive exists; use a new --output instead of overwriting.")
    paths = {n: ROOT/n for n in FILES}
    for p in output.iterdir():
        if p.is_file():
            paths["v8_reference/preml/" + p.name] = p
    for name in ["v8_validation.json", "notebook_overview.png", "notebook_label_distribution.png"]:
        path = ROOT / "reports/v8_preml_20261001" / name
        if path.is_file():
            paths["v8_reference/validation/" + name] = path
    files = {}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, path in sorted(paths.items()):
            data = path.read_bytes()
            files[name] = hashlib.sha256(data).hexdigest()
            info = zipfile.ZipInfo(BASENAME + "/" + name, date_time=(2026,10,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = ((0o755 if name.endswith(".sh") else 0o644) << 16)
            z.writestr(info, data)
        receipt = {
            "id": manifest["id"], "scope": "PREML_ONLY",
            "dataset_included": True, "raw_archive": "Author's workbook includes ten-source rows.",
            "feature_dimensions": 2056, "training_enabled": False, "ready_for_ml": False,
            "original_notebook": "Reference only; runner executes reviewed feature helpers, not its training cells.",
            "v8_reference": "Evidence from preparation machine; lab run recomputes separate data/builds.",
            "files": files,
        }
        z.writestr(BASENAME+"/CAMPUS_PACKAGE_MANIFEST.json",
                   json.dumps(receipt, indent=2, ensure_ascii=False))
    with zipfile.ZipFile(temp) as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f"Corrupt member: {bad}")
        for name, expected in files.items():
            if hashlib.sha256(z.read(BASENAME+"/"+name)).hexdigest() != expected:
                raise ValueError(f"Archive content differs: {name}")
        if any("/.venv" in name or "/runs/" in name for name in z.namelist()):
            raise ValueError("Unexpected runtime/model history in archive.")
    temp.replace(target)
    result = {"archive": str(target), "bytes": target.stat().st_size,
              "sha256": digest(target), "file_count": len(files)}
    target.with_suffix(".zip.sha256").write_text(
        result["sha256"] + "  " + target.name + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build(args.output)
