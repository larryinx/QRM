#!/usr/bin/env bash
# TRM + FSQ on ARC-AGI-1. This is the configuration we launched (8 GPUs); the run was not
# completed, so treat it as a starting point rather than a tuned recipe.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_train model=trm_att_arc_agi dataset=arc_agi_1 train=arc_agi \
    model.use_fsq=True model.fsq_levels=[8,5,5,5] \
    model.fsq_residual_mode=fixed model.fsq_residual_weight=0.3 model.fsq_recon_weight=0.5 \
    model.weighted_em_loss=True model.weighted_em_weight=1 \
    model.fsq_sampling_training=False model.fsq_sampling_inference=False "$@"
