#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Buat .venv dan instal requirements_training.txt sesuai WINDOWS_GUIDE_V5.md."
  exit 1
fi

if [[ ! -f "data/builds/perfume-five-grouped-v2/dataset_manifest.json" ]]; then
  .venv/bin/python -B -m alignment.perfume_data
fi

.venv/bin/python -B -m alignment.experiments_v5 --check
.venv/bin/python -B -m alignment.experiments_v5
.venv/bin/python -B -m alignment.collect_results_v5

echo "SELESAI. Ambil ZIP perfume-grouped-v5-results terbaru dari folder campus-results."
