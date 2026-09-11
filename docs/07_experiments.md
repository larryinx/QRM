# Experiments and results

All accuracies are sequence-level exact match on the test split (in
percent), except ARC, which reports pass@2. "TRM budget" runs use the
configs in `config/train/`: 50k epochs on Sudoku and Maze, 100k on ARC.
Several of the FSQ and QRM sweeps below were trained for shorter
budgets; the TRM baseline in each table was trained under the same
budget as the rows it is compared with.

## 1. TRM replication

The starting point was a strict replication of TRM inside the
Hugging Face `Trainer`, including the dual optimizer, EMA, stablemax
loss, augmentation-inverted ARC evaluation and checkpoint voting.

| Task | Model | Ours | TRM paper |
|---|---|---|---|
| Sudoku-Extreme | TRM-MLP | 84.53 | 87.4 |
| Sudoku-Extreme | TRM-Att | 81.72 | 74.7 |
| Maze-Hard | TRM-Att | 86.46 | 85.3 |
| Maze-Hard | TRM-MLP | 0.00 | 0.0 |
| ARC-AGI-1 | TRM-Att | 44.12 single / 45.12 voted | 44.6 |
| ARC-AGI-2 | TRM-Att | 7.08 single / 7.36 voted | 7.8 |

All models: `T=3, n=6` (Sudoku) or `n=4` (Maze, ARC), 2 layers, hidden
size 512.

## 2. TRM + FSQ on Sudoku and Maze

### 2.1 Pure quantization hurts Sudoku, not Maze

First runs with `alpha = 1.0` (the quantized latent fully replaces
`z_L`), `fsq_levels=[8,5,5,5]`:

| Task / model | TRM (same budget) | FSQ greedy (F/F) | FSQ sampled (T/T) |
|---|---|---|---|
| Sudoku TRM-Att | 77.5 | 32.4 | 46.1 |
| Sudoku TRM-MLP | 81.3 | 61.8 | 66.2 (5 levels: [8,5,5,5,5]) |
| Maze TRM-Att | 85.1 | 81.2 | 85.4 |

Training with sampling and evaluating greedily (or the reverse) collapses
to 0 on Sudoku; keep the two settings equal.

### 2.2 Residual mixing recovers it

`fsq_residual_mode=fixed`, `fsq_residual_weight=0.3`:

| Task / model | TRM | FSQ res0.3 F/F | Note |
|---|---|---|---|
| Sudoku TRM-MLP | 81.3 | **84.4** | best TRM-MLP + FSQ |
| Sudoku TRM-Att | 77.5 | 76.9 | parity |
| Sudoku TRM-Att | 77.5 | **83.0** | + `fsq_pre_norm` + `fsq_recon_weight=0.5` + `weighted_em_loss` (w=1) |

The full attention recipe is

```
model=trm_att_sudoku dataset=sudoku train=sudoku model.use_fsq=True \
  model.fsq_levels=[8,5,5,5] model.fsq_residual_mode=fixed model.fsq_residual_weight=0.3 \
  model.fsq_pre_norm=True model.fsq_recon_weight=0.5 \
  model.weighted_em_loss=True model.weighted_em_weight=1 \
  model.fsq_sampling_training=False model.fsq_sampling_inference=False
```

Other things we tried on Sudoku: `learned_scalar` residual (81.4 with
the same recipe and sampling; alpha converges to about 0.35 to 0.39),
six-dimensional levels `[8,3,3,3,3,3]` (81.0), `[8,5,5,5,5]` (worse for
attention, 8.7 with alpha=1), the attention residual over all codes
(`fsq_residual_mode=attention`, 82.0 for MLP with `[3,3,3,3]`; retired
because it never beat the scalar blend), exact-match up-weighting alone
(71.1), and matched-distribution training subsets (79.9 on
`sudoku_rating`, 72.7 on `sudoku_rating_blank`). Why the residual works
is discussed in [06_fsq_entropy_analysis.md](06_fsq_entropy_analysis.md).

On Maze the plain quantized model with sampling (85.4) is already at
parity with TRM (85.1); adding pre-norm and the residual gives 84.6 with
sampling but 43 without it (the normalized codes drift when the model is
trained greedily), so on Maze we keep sampling on when pre-norm is on.

## 3. QRM: tree search on Sudoku and Maze

`model=qrm_*` configs. `fsq_top_k` is the branching factor, `num_iterations`
the search budget, `train.gradient_accumulation_steps` the optimizer
cadence (0 = one update per search).

| Task / model | Setting | Exact acc |
|---|---|---|
| Sudoku TRM-MLP | TRM baseline | 81.3 |
| Sudoku QRM-MLP | chain: `fsq_top_k=1 num_iterations=16 accum=1`, pre-norm | 78.7 |
| Sudoku QRM-MLP | tree: `fsq_top_k=4 num_iterations=64 accum=1`, lr 2e-4 | 84.9 |
| Sudoku QRM-MLP | tree: `fsq_top_k=4 num_iterations=64 accum=4`, lr 2e-4 | **88.8** |
| Sudoku QRM-Att | tree: `fsq_top_k=4 num_iterations=64 accum=4`, lr 2e-4, pre-norm | **85.8** |
| Maze QRM-Att | chain: `fsq_top_k=1 num_iterations=16 accum=1`, pre-norm | 79.2 |

All QRM rows use `fsq_residual_mode=fixed fsq_residual_weight=0.3`,
UCT with `c_uct=1.414`, depth penalty `alpha=0.5`, token-accuracy
reward. The best row is

```
model=qrm_mlp_t_sudoku dataset=sudoku train=sudoku \
  model.fsq_levels=[8,5,5,5] model.fsq_top_k=4 \
  model.fsq_residual_mode=fixed model.fsq_residual_weight=0.3 model.fsq_pre_norm=False \
  model.num_iterations=64 train.gradient_accumulation_steps=4 \
  train.lr=2e-4 train.puzzle_emb_lr=2e-4
```

Two observations. Updating the weights every 4 search iterations (so
that later expansions in a tree use newer weights than earlier ones)
beats one update per completed search, 88.8 versus 84.9. And the
degenerate chain (`fsq_top_k=1`) is *below* the TRM + FSQ chain of
section 2 because it drops the halting head and uses the tree loss
without any search benefit; the gain comes from the search, not from
the loss.

Other QRM settings we ran on Sudoku without improving on the above:
branching factor 2 with `alpha=0` and `c_uct=0.5` at several cadences,
learning rate 5e-4, and `em_weight=1.0` rewards.

## 4. ARC-AGI and Maze with tree search: not completed

TRM + FSQ runs on ARC-AGI-1/2 (residual 0.2 to 0.3, with and without
pre-norm and exact-match weighting) and QRM tree-search runs on
ARC-AGI-1/2 and Maze (`fsq_top_k=4 num_iterations=64`, cadence 4 to 16,
`em_weight` 0.5 to 1.0, `alpha` 0.1 to 0.5) were configured and
launched but did not finish within our GPU budget, and we do not report
partial numbers for them. The example scripts in
`scripts/train_examples/` reproduce those launch settings so the runs can
be completed on adequate hardware (8 GPUs for ARC).

## 5. Puzzle embeddings on ARC-AGI-1

| Identifier mode | Embedding rows | pass@2 |
|---|---|---|
| `full` (TRM default) | 876,406 | 45 |
| `no_aug` | 961 | 25 |
| `full`, low rank 64 (`puzzle_emb_rank=64`) | 876,406 x 64 | 39 |

We kept `full` for parity with TRM and report the embedding table as a
separate parameter count (about 449M buffer entries versus about 7M
network parameters); see [08_design_notes.md](08_design_notes.md).
