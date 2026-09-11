#!/usr/bin/env bash
# ARC-AGI evaluation for a TRM / TRM + FSQ run: per-checkpoint pass@K, then
# aggregated voting across checkpoints, then a summary table.
#
#   bash scripts/eval/eval_arc.sh results/<run>                 # all checkpoint-* dirs in the run
#   bash scripts/eval/eval_arc.sh results/<run> checkpoint-4000 # a single checkpoint
#
# NPROC_PER_NODE > 1 shards the test set across GPUs with torchrun.
# Outputs: <ckpt>/arc_metrics.json, <ckpt>/arc_results/submission.json,
#          <run>/arc_aggregate_ckpt<N>/{arc_metrics.json,submission.json},
#          results/arc_pretty_table.txt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXP_DIR="${1:?usage: $0 <run_dir> [<checkpoint name>]}"
ONLY_CKPT="${2:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

if [ -n "${ONLY_CKPT}" ]; then
    CKPTS=("${EXP_DIR}/${ONLY_CKPT}")
else
    CKPTS=("${EXP_DIR}"/checkpoint-*)
fi

for CKPT in "${CKPTS[@]}"; do
    echo "== ${CKPT} =="
    if [ "${NPROC_PER_NODE}" -gt 1 ]; then
        MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
        torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
            -m qrm.arc_eval_multi_gpu --ckpt_dir "${CKPT}"
    else
        python -m qrm.arc_eval --ckpt_dir "${CKPT}"
    fi
done

python -m qrm.arc_aggregate --exp_dir "${EXP_DIR}"
python -m qrm.arc_pretty_print
