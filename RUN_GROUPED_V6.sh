#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Ikuti instalasi sekali di LINUX_GUIDE_V6.md."
  exit 1
fi
.venv/bin/python -u -B -m alignment.experiments_v6
.venv/bin/python -u -B -m alignment.collect_results_v6
echo "SELESAI. Bawa pulang ZIP hasil terbaru dari folder campus-results."
