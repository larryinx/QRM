#!/usr/bin/env bash
# Run the FSQ entropy analyzer on one checkpoint.
#
#   bash scripts/analysis/run_fsq_entropy.sh <ckpt_dir> [<output_dir>]
#
# <ckpt_dir> is a checkpoint-* or final_checkpoint directory of a run
# trained with model.use_fsq=True (TRM + FSQ) or a qrm_* model. The
# checkpoint's parent must contain the config.yaml written by qrm.train.
#
# Environment overrides: ANALYZE_ROUNDS (default "1,8,16"), MAX_SAMPLES
# (default 1024), BATCH_SIZE (default 768), FSQ_SAMPLING ("true"|"false",
# default "false" = greedy quantization regardless of training setting).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT_DIR="${1:?usage: $0 <ckpt_dir> [<output_dir>]}"
OUTPUT_DIR="${2:-${CKPT_DIR}/analysis/fsq_entropy}"

python -m qrm.analyze \
    --ckpt_dir "${CKPT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --analyzer fsq_entropy \
    --analyze_rounds "${ANALYZE_ROUNDS:-1,8,16}" \
    --max_samples "${MAX_SAMPLES:-1024}" \
    --batch_size "${BATCH_SIZE:-768}" \
    --fsq_sampling_inference "${FSQ_SAMPLING:-false}"

echo "Summary written to ${OUTPUT_DIR}/fsq_entropy_summary.json"
