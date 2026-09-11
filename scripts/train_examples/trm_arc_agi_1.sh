#!/usr/bin/env bash
# TRM baseline on ARC-AGI-1 (+ ConceptARC). Our runs used NPROC_PER_NODE=4.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_train model=trm_att_arc_agi dataset=arc_agi_1 train=arc_agi "$@"
