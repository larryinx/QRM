#!/usr/bin/env bash
# QRM model in chain mode: branching factor 1, one optimizer step per step, 16 steps.
# This degenerates the tree search to a single quantized chain and is useful as a control
# (78.7 on Sudoku with the MLP-T block, 79.2 on Maze). Pass model=qrm_att_maze dataset=maze train=maze for Maze.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_train model=qrm_mlp_t_sudoku dataset=sudoku train=sudoku \
    model.fsq_levels=[8,5,5,5] model.fsq_top_k=1 \
    model.fsq_residual_mode=fixed model.fsq_residual_weight=0.3 model.fsq_pre_norm=True \
    model.num_iterations=16 train.gradient_accumulation_steps=1 "$@"
