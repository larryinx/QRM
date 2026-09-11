# QRM: idea and method

QRM (Quantized Recursive Model) starts from the Tiny Recursive Model (TRM)
and changes one thing about how it reasons: the latent state that is
handed from one supervision step to the next passes through a discrete
bottleneck. Making that hand-off discrete gives the recursion a small set
of distinct "next states" to choose from, which in turn makes it possible
to search over reasoning trajectories with a tree instead of committing to
a single chain.

The project has three layers, each usable on its own:

1. **TRM**, re-implemented on top of the Hugging Face `Trainer`
   (`model=trm_*` configs). This is a faithful replication of the
   original TRM and is the baseline for everything else.
2. **TRM + FSQ** (`model=trm_* model.use_fsq=True`): the same model with
   a finite-scalar-quantization layer on the latent hand-off, trained as a
   single chain with TRM's loss.
3. **QRM** (`model=qrm_*`): FSQ plus Monte-Carlo tree search over the
   quantized candidates during training, with a depth- and reward-weighted
   tree loss and no adaptive-computation-time halting.

This document explains the method. [02_codebase.md](02_codebase.md) maps
it to the code and [03_training_guide.md](03_training_guide.md) shows
how to run each layer.

## 1. Background: recursive reasoning with a tiny network

TRM keeps two latent tensors of shape `[B, S, D]`: an answer hypothesis
`z_H` (the paper's `y`) and a reasoning state `z_L` (the paper's `z`),
where `S` is the sequence length plus a prefix of puzzle-embedding slots
and `D` is the hidden size (512 here). One supervision step runs
`H_cycles` outer cycles; each outer cycle refines `z_L` for `L_cycles`
inner iterations with the input injected, then refines `z_H` once from
`z_L`. Only the last outer cycle carries gradient; the earlier ones run
under `no_grad` to "warm up" the state.

```
for h in range(H_cycles - 1):            # no grad
    for l in range(L_cycles):
        z_L = f(z_L, z_H + x)
    z_H = f(z_H, z_L)
for l in range(L_cycles):                # with grad
    z_L = f(z_L, z_H + x)
z_H = f(z_H, z_L)                        # <- QRM inserts FSQ before this line
logits = lm_head(z_H)
```

`f` is one shared two-layer block (attention + SwiGLU, or the MLP-T
variant that mixes across the sequence axis). After each supervision step
the carry `(z_H, z_L)` is detached and fed to the next step, so the model
is trained with truncated backpropagation across steps and full
backpropagation within the last cycle. TRM adds a halting head and a
binary cross-entropy halt loss; a step budget of `halt_max_steps` (16)
bounds the recursion.

## 2. Quantized hand-off: TRM + FSQ

QRM inserts a quantizer between the final `z_L` refinement and the final
`z_H` update:

```
z_L      = f(z_L, z_H + x)               # last inner iteration
z_q      = FSQ(z_L)                      # discrete bottleneck
z_L'     = alpha * z_q + (1 - alpha) * z_L
z_H      = f(z_H, z_L')
```

The quantizer is finite scalar quantization (FSQ): a linear projection
from `D` down to `C = len(fsq_levels)` scalars, each squashed with `tanh`
and rounded to one of `L_i` uniformly spaced levels, then projected back
up to `D`. The implicit codebook has `prod(fsq_levels)` entries; the
default `[8, 5, 5, 5]` gives 1000 codes over four scalars. There is no
learned codebook and no commitment loss, so nothing can "collapse" in the
VQ-VAE sense. Gradients pass through the rounding with a straight-through
estimator.

Two additions came out of the empirical study in
[06_fsq_entropy_analysis.md](06_fsq_entropy_analysis.md):

- **Residual mixing** (`fsq_residual_mode`, `fsq_residual_weight`). The
  quantized path has rank at most `C`, so replacing a 512-dimensional
  state by it between steps is a severe bottleneck. Mixing with the
  continuous state at `alpha = 0.3` is the configuration that works on
  Sudoku; `alpha = 1.0` (pure quantization) is the backward-compatible
  default and works on Maze. `learned_scalar` mode learns `alpha` through
  a sigmoid.
- **Pre-tanh normalization** (`fsq_pre_norm`). Standardizing each of the
  `C` scalars across positions before `tanh` avoids saturation, which
  otherwise makes every dimension behave like a binary feature.

A reconstruction term `MSE(z_q, stopgrad(z_L))` (`fsq_recon_weight` in
TRM, `lambda_recon` in QRM) keeps the two paths aligned.

## 3. Stochastic top-k quantization

Deterministic rounding gives one next state. To obtain several, QRM
replaces nearest-level rounding by a Gibbs distribution over levels. For
scalar `u` and level `q_k` with temperature `sigma`,

```
d_k = (u - q_k)^2 / sigma^2,      p_k = softmax_k(-d_k)
```

The implementation computes this per dimension, sums the log
probabilities over the `C` dimensions to get a log probability for every
one of the `prod(levels)` codes, and takes the top `fsq_top_k` codes per
position. (Ranking joint codes rather than taking the top-k per
dimension avoids a `k^C` blow-up of candidates.) Three modes use this
ranking:

| Mode | Used by | Output |
|---|---|---|
| all top-k | QRM training (tree expansion) | `M = fsq_top_k` candidates per position |
| greedy top-1 | TRM + FSQ, QRM inference | 1 candidate |
| sample from the renormalized top-k | `fsq_sampling_*` in TRM, `do_sampling` in QRM | 1 candidate |

Each candidate `m` gets its own final `z_H` and its own logits, so one
forward pass of QRM returns logits of shape `[B, M, N, V]`.

## 4. Tree search over quantized states

With `M` discrete children per step, a supervision step becomes a node
expansion. QRM builds one search tree per puzzle in the batch during
training:

- **Node**: a detached carry `(z_H, z_L)`, its depth, and running
  statistics (visit count `N`, reward sum `W`, `Q = W / N`).
- **One `forward()` call = one search iteration**: select a node from
  each tree, run the recursion once for the whole batch of selected
  nodes, quantize to `M` children, score every child, attach the
  children, and back up their rewards. `num_iterations` calls make up one
  search.
- **Selection** uses UCT, `Q + c_uct * sqrt(ln N_parent / N_child)`,
  with unvisited children scored `+inf` and `Q` clipped to `[0, 1]`.
  PUCT (priors from the FSQ log probabilities) is kept for ablation.
- **Reward** needs no rollout: because every node's `z_H` decodes to an
  answer, the child's reward is its token accuracy (optionally mixed with
  exact match, `em_weight` / `token_acc_weight`) against the label.
- **Backup** treats each of the `M` children as one simulation: each
  child gets `N += 1, W += R`, every ancestor gets `N += M, W += sum R`.
- **Depth** is bounded by `halt_max_steps`; at the last level a node is
  expanded once and then closed (`terminal_selection_mode=skip_saturated`).
- After the search, the best leaf by `Q` becomes the carry for the next
  puzzle, and the tree is released to free GPU memory.

At inference time there is no tree: QRM runs a fixed chain of
`halt_max_steps` greedy (or sampled) quantized steps, exactly like
TRM + FSQ without the halting head.

## 5. Training objective

Every expanded child contributes a supervised term, weighted by how good
it is and how deep it sits:

```
w(mu)    = (0.2 + 0.8 * R(mu)) * (1 - alpha_depth * d(mu) / D_tree)
L_tree   = sum over children  w(mu) * CE(logits(mu), y)
L_total  = lambda_tree * L_tree
         + lambda_recon * MSE(mean_m z_q^(m), stopgrad(z_L))
         + lambda_div   * L_div
```

`CE` is the stablemax cross-entropy TRM uses, normalized per sequence.
The depth penalty (`alpha`, default 0.5, `D_tree = halt_max_steps`)
favours shallow correct answers. `L_div` is an optional energy-score
term that pushes candidates apart (`lambda_div`, off in the recipes that
worked). QRM drops TRM's halt losses entirely: the fixed tree depth and
the depth weighting play the role of early stopping.

Because the Hugging Face `Trainer` calls `training_step` once per
gradient-accumulation micro-step, QRM maps one search iteration to one
micro-step. The optimizer cadence (`train.gradient_accumulation_steps`)
can equal the search budget (`num_iterations`, the "on-policy" schedule)
or be shorter, in which case the weights update several times while a
tree persists ("off-policy"). The best Sudoku results used the
off-policy schedule with updates every 4 iterations.

## 6. What the evidence says so far

- TRM replication matches the paper on Sudoku-Extreme, Maze-Hard,
  ARC-AGI-1 and ARC-AGI-2 (see [07_experiments.md](07_experiments.md)).
- Pure quantization (`alpha = 1`) is neutral on Maze and harmful on
  Sudoku; the residual mix recovers and slightly exceeds the TRM
  baseline at equal budget (TRM-Att 77.5 to 83.0, TRM-MLP 81.3 to 84.4).
- QRM with tree search reaches 88.8 exact accuracy on Sudoku-Extreme
  with the MLP variant and 85.8 with the attention variant, above every
  TRM + FSQ chain we trained.
- Tree-search training on ARC-AGI-1/2 and Maze was set up but not
  completed for lack of GPU time; the scripts and configs are in place.

## 7. Relation to the paper

The manuscript "Quantized Recursive Model: Advancing Recursive Reasoning
with Quantized Tree Search" describes the same method. Differences
between the paper's notation and the code:

| Paper | Code |
|---|---|
| element-wise stochastic top-k (Algorithm 2) | joint top-k over codes (`StochasticFSQ.compute_joint_log_probs`) |
| reward = exact-match indicator | `em_weight * EM + token_acc_weight * token_acc`, default token accuracy |
| `z^(s+1,0) = Q(sg[z^(s,T)])` | FSQ input is not detached; only the reconstruction target is |
| single quantization loss | reconstruction loss plus optional diversity loss and residual mixing |
| MCTS with UCT | UCT default, PUCT available, all-children backup |
