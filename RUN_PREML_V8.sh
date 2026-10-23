#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
echo "V8 - Audit dataset Suh dan fitur Morgan 2048 + 8. Tidak training."
if [[ ! -x .venv-v8/bin/python ]]; then
  command -v python3.12 >/dev/null || { echo "Python 3.12 belum tersedia. Baca LINUX_GUIDE_V8.md."; exit 1; }
  python3.12 -m venv .venv-v8
fi
if ! .venv-v8/bin/python -B -c "from v8.prepare import validate_environment; validate_environment()" >/dev/null 2>&1; then
  .venv-v8/bin/python -m pip install -r requirements_preml_v8.txt
fi
set +e
.venv-v8/bin/python -B -m v8.prepare run
status=$?
set -e
if [[ $status -eq 2 ]]; then
  echo "AUDIT SELESAI. STOP sebelum ML; baca data/builds/suh-ten-v8-m2048-d8-preml/HASIL_AUDIT_V8.md"
  exit 2
fi
echo "Error teknis. Simpan pesan error; jangan timpa hasil." >&2
exit 1
