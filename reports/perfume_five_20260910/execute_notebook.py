"""Execute plain-Python notebook cells with captured rich display outputs."""
import base64
import contextlib
import io
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
import nbformat
import psutil

ROOT = Path(__file__).resolve().parents[2]
if Path.cwd().resolve() != ROOT:
    raise RuntimeError("Run this script from the ML repository root.")
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
PATH = ROOT / "notebooks/02_perfume_five_preprocessing.ipynb"
notebook = nbformat.read(PATH, as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.execution_count, cell.outputs = None, []
namespace = {"__name__": "__preprocessing_audit__"}
active_outputs = []

def display(*objects):
    for obj in objects:
        data = {"text/plain": repr(obj)}
        if hasattr(obj, "savefig"):
            buffer = io.BytesIO()
            obj.savefig(buffer, format="png", dpi=115, bbox_inches="tight")
            data["image/png"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        elif type(obj).__module__.startswith("PIL.") and hasattr(obj, "save"):
            buffer = io.BytesIO()
            obj.save(buffer, format="PNG")
            data["image/png"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        elif hasattr(obj, "_repr_html_"):
            value = obj._repr_html_()
            if value:
                data["text/html"] = value
        active_outputs.append(nbformat.v4.new_output("display_data", data=data))

namespace["display"] = display
started, executed, failure, timings = time.monotonic(), 0, None, []
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    executed += 1
    print(f"Executing cell {executed}...", flush=True)
    cell.execution_count = executed
    active_outputs = cell.outputs
    stream = io.StringIO()
    cell_started = time.monotonic()
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            exec(compile(cell.source, f"preprocessing:cell{executed}", "exec"), namespace)
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        active_outputs.append(nbformat.v4.new_output(
            "error", ename=type(exc).__name__, evalue=str(exc),
            traceback=traceback.format_exc().splitlines()))
    if stream.getvalue():
        cell.outputs.insert(0, nbformat.v4.new_output(
            "stream", name="stdout", text=stream.getvalue()))
    timings.append({"cell": executed, "seconds": time.monotonic() - cell_started})
    if failure:
        break
info = psutil.Process().memory_info()
result = {
    "status": "PASS" if failure is None else "FAIL",
    "completed_at": datetime.now().astimezone().isoformat(),
    "execution_mode": "fresh project Python process; sequential plain-Python cells and rich-output capture; no interactive kernel",
    "python_executable": sys.executable,
    "code_cells_executed": executed,
    "code_cells_total": sum(c.cell_type == "code" for c in notebook.cells),
    "seconds": time.monotonic() - started,
    "peak_process_working_set_bytes": getattr(info, "peak_wset", info.rss),
    "cell_timings": timings, "failure": failure,
    "training_started": False, "test_arrays_deserialized": False,
}
notebook.metadata["audit_execution"] = result
nbformat.validate(notebook)
nbformat.write(notebook, PATH)
(OUT / "execution.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
if failure:
    raise SystemExit(1)
