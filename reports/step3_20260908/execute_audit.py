"""Execute the audit's plain-Python cells sequentially in the project environment."""
import contextlib
import io
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import nbformat
import psutil

ROOT = Path(__file__).resolve().parents[2]
if Path.cwd().resolve() != ROOT:
    raise RuntimeError("Run from the original ML repository root")
out = Path(__file__).resolve().parent
path = out / "audit_notebook.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
namespace = {"__name__": "__audit__"}
started = time.monotonic()
failure = None
executed = 0
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    executed += 1
    cell.execution_count = executed
    cell.outputs = []
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            exec(compile(cell.source, f"audit_notebook:cell{executed}", "exec"), namespace)  # noqa: S102 - Reviewed local notebook only.
    except BaseException as exc:  # noqa: BLE001 - Persist notebook errors/interrupts before exiting.
        failure = {"type":type(exc).__name__, "message":str(exc)}
        cell.outputs.append(nbformat.v4.new_output(
            "error", ename=type(exc).__name__, evalue=str(exc),
            traceback=traceback.format_exc().splitlines()))
    if stream.getvalue():
        cell.outputs.insert(0, nbformat.v4.new_output(
            "stream", name="stdout", text=stream.getvalue()))
    if failure:
        break
info = psutil.Process().memory_info()
result = {
    "completed_at":datetime.now().astimezone().isoformat(),
    "status":"PASS" if failure is None else "FAIL",
    "execution_mode":"fresh project Python process, sequential plain-Python notebook cells; no interactive kernel",
    "code_cells_executed":executed,
    "seconds":time.monotonic()-started,
    "peak_audit_process_working_set_bytes":getattr(info, "peak_wset", info.rss),
    "memory_scope":"audit Python process only; excludes launcher and other applications",
    "system_available_ram_bytes_after":psutil.virtual_memory().available,
    "cpu_temperature_c":None,
    "cpu_temperature_reason":"No verified live sensor session during this lightweight audit",
    "training_started":False, "test_arrays_deserialized":False, "failure":failure
}
notebook.metadata["audit_execution"] = result
nbformat.validate(notebook)
nbformat.write(notebook, path)
(out / "audit-execution.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result))
if failure:
    raise SystemExit(1)
