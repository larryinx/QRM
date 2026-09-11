#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_DIR}"

# Variant R: 1k subsample with rating histogram matched to the test set.
python -m qrm.data_prep.build_sudoku_dataset \
  --output-dir "${REPO_DIR}/data/sudoku-extreme-1k-aug-1000-rating" \
  --subsample-size 1000 \
  --num-aug 1000 \
  --sample-mode match_rating \
  --match-reference test \
  --rating-bin-width 1

# Variant B: 1k subsample with blank-count histogram matched to the test set.
python -m qrm.data_prep.build_sudoku_dataset \
  --output-dir "${REPO_DIR}/data/sudoku-extreme-1k-aug-1000-blank" \
  --subsample-size 1000 \
  --num-aug 1000 \
  --sample-mode match_blank \
  --match-reference test \
  --blank-bin-width 1

# Variant RB: 1k subsample with joint (rating, blank) histogram matched
# to the test set. Rating uses width-5 buckets to keep the joint dense.
python -m qrm.data_prep.build_sudoku_dataset \
  --output-dir "${REPO_DIR}/data/sudoku-extreme-1k-aug-1000-rating-blank" \
  --subsample-size 1000 \
  --num-aug 1000 \
  --sample-mode match_joint \
  --match-reference test \
  --rating-bin-width 5 \
  --blank-bin-width 1

# Variant F: full CSV pool, no subsampling, no augmentation.
python -m qrm.data_prep.build_sudoku_dataset \
  --output-dir "${REPO_DIR}/data/sudoku-extreme-full" \
  --num-aug 0
