#!/usr/bin/env bash
# TRM baseline on Sudoku-Extreme (attention block). For the MLP-T variant pass model=trm_mlp_t_sudoku.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_train model=trm_att_sudoku dataset=sudoku train=sudoku "$@"
