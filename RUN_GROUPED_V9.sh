#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUTF8=1
if [[ ! -x .venv/bin/python ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv
  elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; assert sys.version_info[:2] == (3, 12)'; then
    python3 -m venv .venv
  else
    echo "Python 3.12 diperlukan. Baca LINUX_GUIDE_V9.md."
    exit 1
  fi
fi
if [[ ! -f .venv/v9_dependencies_ready.txt ]]; then
  .venv/bin/python -m pip install -r requirements_campus_v9.txt
  .venv/bin/python -u -B -m alignment.experiments_v9 --check
  printf 'V9 dependencies verified\n' > .venv/v9_dependencies_ready.txt
fi
echo "V9: 8 kandidat; 64 jam tuning total, ditambah validasi dan fitting final."
echo "CPU 6 thread, model berurutan. Jalankan command yang sama untuk resume."
.venv/bin/python -u -B -m alignment.experiments_v9
.venv/bin/python -u -B -m alignment.collect_results_v9
echo "SELESAI. Bawa ZIP terbaru dari folder campus-results."
