#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_DIR}"

MODE="${1:-full}"

case "$MODE" in
  full)
    OUTPUT_DIR="./data/arc1concept-aug-1000"
    ;;
  no_color)
    OUTPUT_DIR="./data/arc1concept-aug-1000-no-color"
    ;;
  no_aug)
    OUTPUT_DIR="./data/arc1concept-aug-1000-no-aug"
    ;;
  shared)
    OUTPUT_DIR="./data/arc1concept-aug-1000-shared"
    ;;
  *)
    echo "Usage: $0 {full|no_color|no_aug|shared}"
    exit 1
    ;;
esac

echo "Preparing ARC-AGI-1 dataset with identifier mode: $MODE"
echo "Output directory: $OUTPUT_DIR"

python -m qrm.data_prep.build_arc_dataset \
  --input-file-prefix ./data/kaggle/combined/arc-agi \
  --output-dir "$OUTPUT_DIR" \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --identifier-mode "$MODE"
