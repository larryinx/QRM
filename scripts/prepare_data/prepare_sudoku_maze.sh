#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_DIR}"

python -m qrm.data_prep.build_sudoku_dataset \
  --output-dir "${REPO_DIR}/data/sudoku-extreme-1k-aug-1000" \
  --subsample-size 1000 \
  --num-aug 1000

python -m qrm.data_prep.build_maze_dataset \
  --output-dir "${REPO_DIR}/data/maze-30x30-hard-1k"