#!/usr/bin/env bash
# QRM (FSQ + MCTS) on Maze-Hard: the configuration we launched on 8 GPUs (exact-match reward,
# optimizer step every 8 iterations). Not completed; starting point only.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
run_train model=qrm_att_maze dataset=maze train=maze \
    model.fsq_levels=[8,5,5,5] model.fsq_top_k=4 \
    model.fsq_residual_mode=fixed model.fsq_residual_weight=0.3 \
    model.em_weight=1.0 model.num_iterations=64 train.gradient_accumulation_steps=8 \
    train.lr=2e-4 train.puzzle_emb_lr=2e-4 "$@"
