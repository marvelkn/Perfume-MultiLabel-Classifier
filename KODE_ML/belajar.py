"""Tampilkan source lokal yang benar-benar diimpor notebook; bukan salinan snippet."""
import ast
import hashlib
import inspect
from pathlib import Path
from IPython.display import Markdown, display

KODE = Path(__file__).resolve().parent
SUMBER_DITAMPILKAN = []

def tampilkan_sumber(objek):
    objek = inspect.unwrap(objek)
    path = Path(inspect.getsourcefile(objek)).resolve()
    if not path.is_relative_to(KODE):
        raise ValueError("Penjelasan utama harus memakai source di KODE_ML: " + str(path))
    lines, first = inspect.getsourcelines(objek)
    source = "".join(lines)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    relative = path.relative_to(KODE).as_posix()
    SUMBER_DITAMPILKAN.append({"file": relative, "name": getattr(objek, "__name__", ""),
                              "first_line": first, "sha256": sha})
    display(Markdown(f"**Source aktual:** `KODE_ML/{relative}`, baris {first}–{first+len(lines)-1}. "
                     f"Kode di bawah dibaca dari objek yang diimpor. SHA-256: `{sha}`.\n\n```python\n{source}\n```"))

def tampilkan_awal(modul):
    """Tampilkan import/konstanta sebelum fungsi pertama, termasuk ALIASES asli."""
    path = Path(inspect.getsourcefile(modul)).resolve()
    if not path.is_relative_to(KODE):
        raise ValueError("Source berada di luar KODE_ML.")
    text = path.read_text(encoding="utf-8-sig")
    nodes = ast.parse(text).body
    first = min((n.lineno for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef))), default=len(text.splitlines())+1)
    source = "\n".join(text.splitlines()[:first-1])
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    display(Markdown(f"**Import dan konstanta aktual:** `KODE_ML/{path.relative_to(KODE).as_posix()}`. "
                     f"SHA-256: `{sha}`.\n\n```python\n{source}\n```"))
