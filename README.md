# Quantized Recursive Model: Advancing Looped Reasoning with Quantized Tree Search

QRM extends the [Tiny Recursive Model](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
(TRM) with a discrete bottleneck on the latent state that is carried
between reasoning steps. The bottleneck is finite scalar quantization
(FSQ); its stochastic top-k variant yields several candidate next states
per step, and QRM trains by expanding those candidates into a search tree
with Monte-Carlo tree search (UCT), supervising every node with a
reward- and depth-weighted loss. The repository contains:

- a TRM re-implementation on the Hugging Face `Trainer` that reproduces
  the paper's numbers on Sudoku-Extreme, Maze-Hard, ARC-AGI-1 and ARC-AGI-2;
- TRM + FSQ, a single-chain model with the quantized hand-off;
- QRM, FSQ plus tree search, with the analysis tooling used to study
  the quantizer.

Sequence-level exact accuracy on Sudoku-Extreme (TRM-MLP block, 50k
epochs): TRM 81.3, TRM + FSQ 84.4, QRM 88.8. See
[docs/07_experiments.md](docs/07_experiments.md) for all results,
including the replication tables and the runs that were not completed.

## Install

Requires Python 3.10+ and a CUDA GPU. The project is managed with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/larryinx/QRM.git && cd QRM
uv sync                 # .venv with torch 2.9 and all dependencies
uv sync --group dev     # + pytest, ruff, black, isort, pre-commit
```

Use `uv run <cmd>` or `source .venv/bin/activate`. See the note in
`pyproject.toml` to pin a different CUDA build of torch.

## Prepare data

```bash
uv run bash scripts/prepare_data/prepare_sudoku_maze.sh   # Sudoku-Extreme (1k x 1000 aug) and Maze-Hard
uv run bash scripts/prepare_data/prepare_arc1.sh full     # ARC-AGI-1 + ConceptARC
uv run bash scripts/prepare_data/prepare_arc2.sh full     # ARC-AGI-2 + ConceptARC
```

The raw ARC-AGI json files are bundled under `data/kaggle/combined/`;
Sudoku and Maze are downloaded from the Hugging Face Hub. Details and
variants: [docs/04_datasets.md](docs/04_datasets.md).

## Train

Training is driven by Hydra configs: `model` picks the architecture
(`trm_*` or `qrm_*`), `dataset` the data, `train` the schedule, and any
key can be overridden on the command line.

```bash
# TRM baseline
uv run python -m qrm.train model=trm_att_sudoku dataset=sudoku train=sudoku

# TRM + FSQ (residual mix 0.3, greedy quantization)
uv run python -m qrm.train model=trm_mlp_t_sudoku dataset=sudoku train=sudoku \
    model.use_fsq=True model.fsq_residual_mode=fixed model.fsq_residual_weight=0.3

# QRM: FSQ + tree search (branching 4, 64 iterations, update every 4)
NPROC_PER_NODE=4 uv run bash scripts/train_examples/qrm_sudoku.sh
```

`scripts/train_examples/` has one script per task and method (TRM,
TRM + FSQ, QRM on Sudoku, Maze, ARC-AGI-1, ARC-AGI-2) plus a chain-mode
control. Set `NPROC_PER_NODE` for multi-GPU (`torchrun`), `WANDB_PROJECT`
/ `WANDB_MODE` for logging, and append overrides as arguments. Runs are
written to `results/<run_name>/` and resume automatically. Full guide:
[docs/03_training_guide.md](docs/03_training_guide.md).

## Evaluate

```bash
uv run bash scripts/eval/eval_checkpoint.sh results/<run>/final_checkpoint   # Sudoku / Maze
uv run bash scripts/eval/eval_arc.sh results/<run>                           # ARC: pass@K + voting
```

## Analyze

```bash
uv run bash scripts/analysis/run_fsq_entropy.sh results/<run>/final_checkpoint
uv run python scripts/analysis/fsq_entropy_report.py --manifest scripts/analysis/fsq_entropy_manifest.example.json
```

The FSQ analyzer records codebook utilization, per-position code
entropy, per-dimension level histograms and quantization error at
chosen recursion steps; the report script compares checkpoints. What we
learned from it is written up in
[docs/06_fsq_entropy_analysis.md](docs/06_fsq_entropy_analysis.md).

## Method in one paragraph

TRM refines an answer state `z_H` and a reasoning state `z_L` for a
fixed number of cycles and detaches them between supervision steps. QRM
quantizes `z_L` at the hand-off with FSQ (`[8,5,5,5]` levels, 1000
codes), mixes it with the continuous state (`alpha = 0.3`), and, instead
of committing to the nearest code, ranks codes by a Gibbs distribution
over quantization levels and keeps the top `k`. Each of the `k`
candidates gets its own answer and its own supervised reward (token
accuracy against the label), so a step becomes a node expansion. During
training one search tree per puzzle is grown with UCT for
`num_iterations` expansions; every node contributes a cross-entropy term
weighted by `(0.2 + 0.8 * reward) * (1 - alpha * depth / max_depth)`,
plus a reconstruction term between the continuous and quantized latents.
At inference QRM runs a greedy quantized chain. There is no halting
head. See [docs/01_overview.md](docs/01_overview.md).

## Results summary

| Task | TRM (ours) | TRM + FSQ | QRM |
|---|---|---|---|
| Sudoku-Extreme, MLP-T block | 81.3 | 84.4 | **88.8** |
| Sudoku-Extreme, attention block | 77.5 | 83.0 | **85.8** |
| Maze-Hard, attention block | 85.1 | 85.4 | not completed |
| ARC-AGI-1, pass@2 (voted) | 45.1 (paper 44.6) | not completed | not completed |
| ARC-AGI-2, pass@2 (voted) | 7.4 (paper 7.8) | not completed | not completed |

Rows within a task are trained under the same budget; TRM numbers on
Sudoku are therefore lower than the full-budget replication (84.5 MLP,
81.7 attention). The tree-search runs on Maze and ARC-AGI were launched
but not finished for lack of GPU time; their configurations are kept in
`scripts/train_examples/`.

## Repository layout

```
src/qrm/            models (trm, qrm), layers (StochasticFSQ), losses, mcts, trainers, data, analyzers, CLI entry points
config/             Hydra groups: model/, dataset/, train/, base.yaml
scripts/
  prepare_data/     dataset builders
  train_examples/   launch scripts for every task x method
  eval/             checkpoint and ARC evaluation
  analysis/         FSQ entropy analysis, report generation, dataset checks
data/kaggle/        raw ARC-AGI json (tracked)
docs/               design, codebase, training, datasets, analysis, experiments
```

## Documentation

[docs/README.md](docs/README.md) indexes the design overview, codebase
guide, training guide, dataset notes, analysis tooling, the FSQ codebook
study, the experiment log and the design notes.

## Acknowledgements and license

The TRM implementation follows
[SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
(MIT), which in turn builds on the
[Hierarchical Reasoning Model](https://github.com/sapientinc/HRM).
Finite scalar quantization follows Mentzer et al., "Finite Scalar
Quantization: VQ-VAE Made Simple". ARC-AGI data is from the
[ARC Prize](https://arcprize.org/) releases; Sudoku-Extreme and Maze-Hard
are the datasets published with HRM. This repository is released under
the Apache License 2.0 (see `LICENSE`).
