#!/usr/bin/env bash
# Record per-recursion-step accuracy and halting statistics for one checkpoint.
#
#   bash scripts/analysis/run_trm_analysis.sh <ckpt_dir> [<output_dir>]
#
# Works for TRM and TRM + FSQ checkpoints. Set SAVE_PREDS=1 to also dump
# per-sample predictions (large). MAX_SAMPLES and BATCH_SIZE as in
# run_fsq_entropy.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT_DIR="${1:?usage: $0 <ckpt_dir> [<output_dir>]}"
OUTPUT_DIR="${2:-${CKPT_DIR}/analysis/trm}"

EXTRA=()
if [ "${SAVE_PREDS:-0}" = "1" ]; then EXTRA+=(--save_preds); fi
if [ -n "${MAX_SAMPLES:-}" ]; then EXTRA+=(--max_samples "${MAX_SAMPLES}"); fi

python -m qrm.analyze \
    --ckpt_dir "${CKPT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --analyzer trm \
    --batch_size "${BATCH_SIZE:-768}" \
    "${EXTRA[@]}"

echo "Per-step results written to ${OUTPUT_DIR}/"
