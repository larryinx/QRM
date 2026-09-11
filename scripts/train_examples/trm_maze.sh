#!/usr/bin/env bash
# TRM baseline on Maze-Hard. Our runs used NPROC_PER_NODE=2-4.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_train model=trm_att_maze dataset=maze train=maze "$@"
