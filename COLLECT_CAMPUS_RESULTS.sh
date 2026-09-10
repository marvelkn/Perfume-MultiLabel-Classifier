#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Environment .venv tidak ditemukan."
  exit 1
fi

.venv/bin/python -B -m alignment.collect_results
