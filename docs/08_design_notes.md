# Design notes

Decisions that are not obvious from the code, with the reasoning behind
them.

## Where to quantize

FSQ sits on `z_L` after the last inner iteration and before the last
`z_H` update, so the answer hypothesis is always computed from a
(partly) discrete reasoning state, and the discrete state is what gets
carried to the next step. Quantizing `z_H` instead would bottleneck the
answer rather than the reasoning; quantizing inside the inner loop would
multiply the cost by `L_cycles`. The FSQ input is left attached to the
graph so the straight-through gradient reaches the reasoning block; only
the reconstruction target is detached.

## Joint top-k instead of per-dimension top-k

The paper describes stochastic quantization per scalar. Taking the top
`k` levels of each of `C` dimensions and combining them yields `k^C`
candidates. The implementation instead scores every code in the
implicit codebook with the summed per-dimension log probabilities and
keeps the best `fsq_top_k` codes. This is exact for the joint
distribution, gives exactly `M` children per node, and is why the
codebook should stay in the low thousands (`[8,5,5,5]` = 1000; a
warning fires above 100k).

## Why the residual mix

`project_out(code)` has rank at most `C = len(fsq_levels)`. With
`alpha = 1` the 512-dimensional state that crosses a supervision step is
confined to a rank-4 subspace, and on Sudoku the model cannot recover
from that. With `alpha = 0.3` the continuous path carries the bulk of
the information and the discrete path acts as a correction that still
defines the branching. The empirical picture (converging code usage,
sharpening within a recursion) is in
[06_fsq_entropy_analysis.md](06_fsq_entropy_analysis.md).

## Why UCT rather than PUCT

Priors from the FSQ log probabilities are almost uniform at temperature
1 (top-1 probability around 0.01 over 1000 codes), so PUCT's prior term
carries little information. UCT with unvisited-first selection was
adopted as the default; PUCT stays available through
`search_rule=puct` for ablations.

## Backing up all children

An expansion evaluates `M` children with supervised rewards at once.
Treating that as `M` simulations (`N += M` on every ancestor) keeps the
visit counts consistent with the amount of evidence gathered, and makes
the exploration term shrink at the right rate. Backing up only the best
child (a DeepSearch-style rule) was tried in design and rejected for
the same reason.

## Removing adaptive computation time

QRM drops TRM's halting head and its two BCE terms. The tree has a fixed
depth, the depth penalty in the node weights favours shallow correct
answers, and the exported carry is the best leaf by `Q`. Analyses of
HRM/TRM showed that a fixed budget of 16 steps performs about as well as
ACT, which made the simplification cheap.

## Search budget, tree depth and optimizer cadence

Three quantities are easy to confuse:

| Quantity | Config key | Meaning |
|---|---|---|
| depth | `model.halt_max_steps` | maximum node depth; also the inference chain length |
| budget | `model.num_iterations` | forward calls (expansions) per puzzle batch |
| cadence | `train.gradient_accumulation_steps` | forward calls per optimizer update; 0 means `num_iterations` |

With cadence equal to the budget, a whole tree is built with fixed
weights. With a shorter cadence the weights move while the tree
persists, so stored node values were produced by older weights. The
second option gave the best Sudoku results (88.8 versus 84.9).

## Hugging Face `Trainer` as the driver

Reusing the `Trainer` gives checkpoints, resume, DDP, logging and
`torch.compile` for free, at the cost of a few workarounds: batching
happens inside the dataset (`per_device_train_batch_size=1`), the
learning-rate schedule is applied by a callback because two optimizers
share it, the puzzle-embedding optimizer works on buffers, and during a
QRM search the batches the `Trainer` fetches are discarded (the model
reuses the cached batch until the search ends). `train.py` scales
`max_steps` by the accumulation factor so the number of examples seen
matches TRM.

## Puzzle embeddings on ARC

TRM learns one embedding vector per augmented puzzle (`full` identifier
mode, 876k rows on ARC-AGI-1) and updates them with SignSGD. This table
dominates the parameter count and ties the model to the exact
augmentations seen in training. We evaluated coarser identifiers
(`no_color`, `no_aug`, `shared`) and low-rank factorizations
(`puzzle_emb_rank`); they cut the table by orders of magnitude but cost
accuracy (`no_aug` 25%, rank 64 39%, versus 45% for `full`). We chose
to keep `full` for comparability with TRM, to report the embedding table
separately from the roughly 7M network parameters (`train.py` prints
both), and to leave a learned puzzle encoder as future work.

## ARC evaluation offline

TRM evaluates ARC by voting across checkpoints during training. Here
each checkpoint is evaluated once (`arc_eval`), its evaluator state is
saved, and `arc_aggregate` merges the states and votes, sorting
candidates by vote count and mean halt confidence. The result is
numerically the same as TRM's online voting and lets any subset of
checkpoints be aggregated after the fact. `arc_eval_multi_gpu` shards
the 400-puzzle test set across ranks and merges the shards on rank 0.

## Things we would do next

- Binary levels (`fsq_levels=[2]*k`) or a non-saturating bound, since
  trained models already use each FSQ dimension as a binary feature.
- An effective-rank tracker for `z_L` across recursion steps.
- Normalizing the tree loss by the sum of node weights so its scale is
  independent of `fsq_top_k`.
- Finishing the ARC-AGI and Maze tree-search runs.
