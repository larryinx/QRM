#!/usr/bin/env bash
# QRM (FSQ + MCTS) on Sudoku-Extreme, MLP-T block: the best run in the repository (88.8 exact accuracy).
# Branching factor 4, 64 search iterations per puzzle batch, one optimizer step every 4 iterations.
# Attention block: model=qrm_att_sudoku model.fsq_pre_norm=True (85.8). Our runs used NPROC_PER_NODE=4.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_train model=qrm_mlp_t_sudoku dataset=sudoku train=sudoku \
    model.fsq_levels=[8,5,5,5] model.fsq_top_k=4 \
    model.fsq_residual_mode=fixed model.fsq_residual_weight=0.3 model.fsq_pre_norm=False \
    model.num_iterations=64 train.gradient_accumulation_steps=4 \
    train.lr=2e-4 train.puzzle_emb_lr=2e-4 "$@"
