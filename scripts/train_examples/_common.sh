# Shared setup for the example launch scripts. Source it; do not run it.
#
# Environment variables you may set before running an example:
#   NPROC_PER_NODE   number of GPUs (default 1 -> plain python; >1 -> torchrun)
#   MASTER_PORT      torchrun rendezvous port (default: random in 29500-30499)
#   WANDB_PROJECT    Weights & Biases project (default "qrm")
#   WANDB_MODE       "online", "offline" (default) or "disabled"
#
# Every example accepts extra Hydra overrides as arguments, e.g.
#   bash scripts/train_examples/trm_sudoku.sh dataset.global_batch_size=384 train.epochs=20000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"   # dataset paths in config/dataset/*.yaml are relative to the repo root

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-qrm}"
export WANDB_MODE="${WANDB_MODE:-offline}"

run_train() {
    if [ "${NPROC_PER_NODE}" -gt 1 ]; then
        MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
        torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" -m qrm.train "$@"
    else
        python -m qrm.train "$@"
    fi
}
