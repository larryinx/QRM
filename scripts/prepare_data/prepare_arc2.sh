#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_DIR}"

case "$MODE" in
  full)
    OUTPUT_DIR="./data/arc2concept-aug-1000"
    ;;
  no_color)
    OUTPUT_DIR="./data/arc2concept-aug-1000-no-color"
    ;;
  no_aug)
    OUTPUT_DIR="./data/arc2concept-aug-1000-no-aug"
    ;;
  shared)
    OUTPUT_DIR="./data/arc2concept-aug-1000-shared"
    ;;
  *)
    echo "Usage: $0 {full|no_color|no_aug|shared}"
    exit 1
    ;;
esac

echo "Preparing ARC-AGI-2 dataset with identifier mode: $MODE"
echo "Output directory: $OUTPUT_DIR"

python -m qrm.data_prep.build_arc_dataset \
  --input-file-prefix ./data/kaggle/combined/arc-agi \
  --output-dir "$OUTPUT_DIR" \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2 \
  --identifier-mode "$MODE"
