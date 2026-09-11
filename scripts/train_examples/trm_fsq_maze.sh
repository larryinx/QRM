#!/usr/bin/env bash
# TRM + FSQ on Maze-Hard (residual mix 0.3, reconstruction loss, greedy quantization).
# Plain quantization with sampling (model.use_fsq=True model.fsq_sampling_training=True
# model.fsq_sampling_inference=True, nothing else) also matches the TRM baseline on Maze.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_train model=trm_att_maze dataset=maze train=maze \
    model.use_fsq=True model.fsq_levels=[8,5,5,5] \
    model.fsq_residual_mode=fixed model.fsq_residual_weight=0.3 model.fsq_recon_weight=0.5 \
    model.fsq_sampling_training=False model.fsq_sampling_inference=False "$@"
