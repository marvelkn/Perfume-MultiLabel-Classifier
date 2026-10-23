from pathlib import Path
import nbformat, sys, base64, json
from nbclient import NotebookClient
from jupyter_client import KernelManager

path = Path("notebooks/04_suh_v8_preml.ipynb")
nb = nbformat.read(path, as_version=4)
km = KernelManager(kernel_name="python3")
km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
try:
    NotebookClient(nb, km=km, timeout=180,
                   resources={"metadata": {"path": str(Path.cwd())}}).execute()
finally:
    if km.has_kernel:
        km.shutdown_kernel(now=True)
nbformat.validate(nb)
nbformat.write(nb, path)
out = Path("reports/v8_preml_20261001")
out.mkdir(parents=True, exist_ok=True)
images = []
for cell in nb.cells:
    for result in cell.get("outputs", []):
        data = result.get("data", {})
        if "image/png" in data:
            name = ["notebook_overview.png", "notebook_label_distribution.png"][len(images)]
            target = out/name
            target.write_bytes(base64.b64decode(data["image/png"]))
            images.append(str(target))
print(json.dumps({"notebook_executed": str(path),
                  "code_cells": sum(c.cell_type == "code" for c in nb.cells),
                  "images": images}))
