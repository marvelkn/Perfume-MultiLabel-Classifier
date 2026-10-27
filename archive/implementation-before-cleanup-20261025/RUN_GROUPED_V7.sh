#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Ikuti instalasi sekali di LINUX_GUIDE_V7.md."
  exit 1
fi
.venv/bin/python -u -B -m alignment.experiments_v7
.venv/bin/python -u -B -m alignment.collect_results_v7
echo "SELESAI. Bawa ZIP terbaru dari folder campus-results."
