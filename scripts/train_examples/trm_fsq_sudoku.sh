#!/usr/bin/env bash
# TRM + FSQ on Sudoku-Extreme, attention block: the recipe that reached 83.0 exact accuracy
# (residual mix 0.3, pre-tanh norm, reconstruction loss, exact-match up-weighting, greedy quantization).
# For the MLP-T variant the best setting was simpler: model=trm_mlp_t_sudoku without
# fsq_pre_norm / fsq_recon_weight / weighted_em_loss (84.4).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_train model=trm_att_sudoku dataset=sudoku train=sudoku \
    model.use_fsq=True model.fsq_levels=[8,5,5,5] \
    model.fsq_residual_mode=fixed model.fsq_residual_weight=0.3 \
    model.fsq_pre_norm=True model.fsq_recon_weight=0.5 \
    model.weighted_em_loss=True model.weighted_em_weight=1 \
    model.fsq_sampling_training=False model.fsq_sampling_inference=False "$@"
