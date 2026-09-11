#!/usr/bin/env bash
# Evaluate one Sudoku / Maze checkpoint (TRM, TRM + FSQ or QRM) on the test split.
#
#   bash scripts/eval/eval_checkpoint.sh results/<run>/checkpoint-12345
#   bash scripts/eval/eval_checkpoint.sh results/<run>/final_checkpoint
#
# Writes <ckpt_dir>/test_metrics_compile.json (skipped if it already exists).
# Set EVAL_USE_COMPILE=0 to evaluate without torch.compile (writes test_metrics.json).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT_DIR="${1:?usage: $0 <ckpt_dir>}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

if [ "${EVAL_USE_COMPILE:-1}" = "1" ]; then
    python -m qrm.test --ckpt_dir "${CKPT_DIR}" --eval_use_compile
else
    python -m qrm.test --ckpt_dir "${CKPT_DIR}"
fi

# Summarize every evaluated checkpoint under results/ into results/pretty_table.txt
python -m qrm.pretty_print $( [ "${EVAL_USE_COMPILE:-1}" = "1" ] && echo --compile )
