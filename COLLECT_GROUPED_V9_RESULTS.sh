#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python -u -B -m alignment.collect_results_v9
echo "ZIP hasil ada di campus-results."
