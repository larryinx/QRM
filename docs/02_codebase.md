# Codebase guide

Everything lives under `src/qrm/`. The package is a Hugging Face
`transformers`-style implementation: models subclass `PreTrainedModel`,
training goes through a `Trainer` subclass, and experiments are
composed with Hydra configs.

```
src/qrm/
  train.py                 Hydra entry point: python -m qrm.train model=... dataset=... train=...
  test.py                  Evaluate one checkpoint on the test split
  analyze.py               Inference-time diagnostics (TRM rounds, FSQ entropy)
  arc_eval.py              ARC: per-checkpoint inference and pass@K
  arc_eval_multi_gpu.py    ARC: same, sharded across GPUs with torchrun
  arc_aggregate.py         ARC: vote across checkpoints (TRM's aggregated voting)
  arc_pretty_print.py      ARC: summary table over results/
  pretty_print.py          Sudoku/Maze: summary table over results/
  models/trm/              TRMConfig, TRMInner, TRMForPuzzleSolving
  models/qrm/              QRMConfig, QRMInner, QRMForPuzzleSolving
  layers/                  attention, SwiGLU, RMSNorm, sparse puzzle embedding, StochasticFSQ, FSQAttention
  losses/                  stablemax cross-entropy; QRM tree and diversity losses
  mcts/                    QRMNode, QRMSearchScorer (UCT/PUCT), QRMMCTSManager
  trainers/                TRMTrainer (dual optimizer, EMA, eval loop), QRMTrainer
  optim/                   SignSGD for sparse puzzle-embedding buffers
  data/puzzle_dataset.py   TRMIterableDataset (internal batching, resumable)
  data_prep/               dataset builders for Sudoku, Maze, ARC
  evaluators/arc.py        ARC pass@K evaluator with augmentation inversion
  analyzers/               BaseAnalyzer, TRMAnalyzer, FSQEntropyAnalyzer
  tests/                   unit and smoke tests (pytest)
config/
  base.yaml                defaults for every key; select model/dataset/train groups
  model/*.yaml             trm_att_*, trm_mlp_t_*, qrm_att_*, qrm_mlp_t_*
  dataset/*.yaml           dataset paths
  train/*.yaml             epochs, learning rates, EMA
```

## 1. Data

`TRMIterableDataset` reads the `.npy` files produced by `data_prep`
(`inputs`, `labels`, `puzzle_identifiers`, `puzzle_indices`,
`group_indices`) and yields **whole batches**: it groups examples by
puzzle, samples `global_batch_size` rows per step, splits them across
ranks, and returns a dict with `input_ids`, `labels`,
`puzzle_identifiers`. The `Trainer` therefore runs with
`per_device_train_batch_size=1` and a pass-through collator. Label
positions that are not part of the answer carry `IGNORE_LABEL_ID` (-100).

The dataset is resumable: `_iters` (epochs consumed) is stored in each
checkpoint and recomputed from the global step on resume, so data order
is reproducible even with worker processes.

## 2. TRM model (`models/trm/modeling_trm.py`)

`TRMInner` holds the parameters:

- `embed_tokens`, optional learned positions or RoPE, and
  `puzzle_emb` (a `CastedSparseEmbedding` whose weights are buffers, not
  parameters, so they can be updated by the dedicated SignSGD optimizer);
- `L_level`, a `ReasoningModule` of `L_layers` blocks; each block is
  attention + SwiGLU with post-RMSNorm, or an MLP over the sequence axis
  when `mlp_t=True`;
- `lm_head` and `q_head` (halt logits);
- `H_init` / `L_init` buffers used to reset the carry;
- optionally `fsq` (a `StochasticFSQ`) plus the residual weight when
  `use_fsq=True`.

`_reasoning_cycles` implements the recursion of
[01_overview.md](01_overview.md) and can stop before the final `z_H`
update (`return_z_L_before_last_z_H=True`). `TRMInner.forward` runs the
cycles, applies FSQ and the residual blend if enabled, computes the final
`z_H`, and returns detached carries plus logits and halt logits.

`TRMForPuzzleSolving` is the `PreTrainedModel` wrapper. It owns the
runtime carry (`_carry`: inner carry, per-sample `steps`, `halted`
flags, and the batch currently being processed). Each `forward` call is
one supervision step for the whole batch:

1. samples flagged `halted` are reset to `H_init` / `L_init` and take
   the incoming batch rows; the others keep their cached rows;
2. one inner forward;
3. halting: a sample halts at `halt_max_steps` or when the halt logit is
   positive (`no_ACT_continue=True`), with random minimum-step
   exploration during training;
4. loss = stablemax CE + 0.5 * halt BCE (+ continue BCE if enabled)
   (+ `fsq_recon_weight` * reconstruction MSE); `weighted_em_loss`
   multiplies the CE of sequences that are not exactly correct by
   `1 + weighted_em_weight`.

Metrics (`accuracy`, `exact_accuracy`, `q_halt_accuracy`, `steps`) are
counted only for samples that halted in this call.

## 3. QRM model (`models/qrm/modeling_qrm.py`)

`QRMInner` composes a `TRMInner` (it does not subclass it) and adds a
`StochasticFSQ`, the residual weight, and its own `lm_head`. Its forward
runs `_reasoning_cycles` up to the last `z_L`, quantizes to `M`
candidates, expands `z_H` to `[B, M, S, D]`, computes the final `z_H`
for every candidate in one batched call, and returns logits
`[B, M, N, V]`, the continuous `z_L`, the quantized candidates, their
codebook indices, and optional PUCT priors. The returned carry is
"multi-candidate" and must be reduced to `[B, S, D]` by the caller
(`select_best_carry`).

`QRMForPuzzleSolving` has two paths:

- **Training** (`_forward_train`): owns a `QRMMCTSManager` with one tree
  per batch element. Each call selects a node per tree, batches the
  selected carries, expands, computes rewards / weights / losses, syncs
  the children into the trees, backs up rewards, and exports the best
  leaf's carry. When `steps` reaches `num_iterations` the search ends:
  metrics such as `max_depth` are recorded, the trees are released, and
  `steps` is reset to zero so the next call starts a new puzzle batch.
  While a search is in progress the incoming batch is ignored and the
  cached `current_data` is reused.
- **Inference** (`_forward_inference`): a single greedy or sampled
  candidate per step for `halt_max_steps` steps, no tree, TRM-style
  metrics.

There is no `halted` vector in `QRMCarry`; a new sequence starts when
`steps == 0`. The halt head inherited from `TRMInner` is unused.

## 4. StochasticFSQ (`layers/stochastic_fsq.py`)

```
project_in (D -> C)  ->  [pre_norm]  ->  bound_soft (tanh, no rounding)
  -> per-dim Gibbs log-probs  ->  joint log-probs over prod(levels) codes
  -> topk(fsq_top_k)  ->  {all k | sample | top-1}
  -> straight-through codes  ->  project_out (C -> D)
```

Shapes: input `[B, N, D]`, output `[B, M, N, D]` plus indices
`[B, M, N]`. `return_diagnostics=True` also returns the top-k indices
and log probabilities, the bounded latent, and the full joint log
probabilities; the analyzers and the PUCT priors use these. Quantization
runs in float32 regardless of the model dtype.

## 5. Losses (`losses/`)

- `stablemax_cross_entropy` (from TRM): per-token CE with the stablemax
  transform, masked by `IGNORE_LABEL_ID`.
- `qrm_tree_loss(logits [B,M,N,V], labels [B,N], weights [B,M])`: per
  candidate, sequence-normalized CE times its weight, summed over batch
  and candidates. The magnitude grows with `M`.
- `qrm_diversity_loss`: negative mean pairwise distance between
  candidates (`diversity_beta`, `diversity_seq_aggregation`).

The reconstruction MSE lives in the model forward for both TRM and QRM.

## 6. Search (`mcts/`)

- `QRMNode`: carry, depth, batch index, parent, prior, immediate reward,
  token accuracy, exact match, `visit_count`, `reward_sum`, `expanded`,
  `terminal_closed`, children; `q_value = W / N` or the immediate reward
  when unvisited.
- `QRMSearchScorer`: `score_child` (UCT or PUCT), `select_child`,
  `backpropagate` (single path).
- `QRMMCTSManager`: `initialize_roots`, `select_node` (returns `None`
  when a tree has no expandable node, in which case the root is
  re-expanded), `sync_children_from_expansion`,
  `backpropagate_all_children`, `get_best_leaf`, and `release`, which
  breaks parent/child reference cycles so CUDA memory is freed
  immediately rather than waiting for Python's cyclic collector.

`tests/test_qrm_mcts.py`, `tests/test_mcts_unit.py` and `tests/test_mcts_expand.py` exercise
this logic on CPU with a fake carry.

## 7. Trainers (`trainers/`)

`TRMTrainer` reproduces the original TRM optimization inside the
Hugging Face loop:

- **Dual optimizer.** `AdamAtan2` over `model.parameters()`, and
  `DistributedCastedSparseEmbeddingSignSGD` over the puzzle-embedding
  buffers (the buffers are not parameters, so `zero_grad` does not touch
  them and the callback zeros them explicitly). On ARC this keeps
  embedding-optimizer memory at about 1.8 GB instead of 5.4 GB with Adam.
- **Schedule.** A cosine schedule with warmup and `lr_min_ratio`,
  applied to both optimizers by `DualOptimizerCallback` before each
  step; the Hugging Face scheduler is a no-op placeholder.
- **Loss scaling.** Loss is divided by the local batch size, so DDP's
  mean-reduction reproduces TRM's `loss / global_batch_size`.
- **EMA.** `EMAHelper` keeps shadow weights (`ema_rate`); at evaluation
  the weights are swapped in place so the compiled model is reused.
- **Evaluation loop.** For each test batch, a fresh carry and a
  `while True` loop until `all_finish`; metrics are summed and divided
  by `count`, and reduced to rank 0 under DDP.
- **Checkpoints** add `ema_state.pt`, `signsgd_optimizer.pt` and
  `dataset_state.pt` next to the usual Hugging Face files; `train.py`
  auto-resumes from the latest `checkpoint-*` in the output directory
  and writes a `final_checkpoint/` at the end.

`QRMTrainer` overrides `create_optimizer` (the puzzle embedding lives at
`model.model.trm_inner.puzzle_emb`), `training_step` (divides the loss by
`gradient_accumulation_steps` and stages metrics until a search
finishes), and `log` (depth-bucketed metrics such as `accuracy_d1_8`
are divided by their own counts).

## 8. Entry points and config

`train.py` composes `config/base.yaml` with the three groups and reads
the model type from the model config name (`trm_*` or `qrm_*`, see
`MODEL_REGISTRY`). It derives `max_steps` and `eval_steps` from
`train.epochs`, `train.eval_interval`, the dataset size, and the
gradient-accumulation factor, so the number of training *examples* seen
is the same for TRM and QRM. Outputs go to
`results/<run_name>/`, where `run_name` encodes the config names and any
command-line overrides (abbreviated by `utils/train.KEY_MAPPING`); a job
id from SLURM / SGE / PBS is prefixed when present. The merged config is
saved as `config.yaml` and reread by `test.py`, `analyze.py` and the ARC
scripts.

Training uses `torch.compile` (inductor) and `report_to=["wandb"]`; set
`WANDB_MODE=disabled` or `WANDB_MODE=offline` if you do not want to log
to Weights & Biases.

Important config keys (all under `model:` unless noted):

| Key | Meaning | Default |
|---|---|---|
| `H_cycles`, `L_cycles`, `L_layers` | recursion depth and block count | 3, 6, 2 |
| `halt_max_steps` | supervision steps per puzzle; QRM tree depth | 16 |
| `mlp_t` | MLP-T block instead of attention | False |
| `puzzle_emb_ndim`, `puzzle_emb_len`, `puzzle_emb_rank` | puzzle embedding size, prefix length, optional low rank | 512, 16, null |
| `use_fsq` | enable FSQ in TRM | False (QRM: always True) |
| `fsq_levels`, `fsq_top_k`, `fsq_temperature` | codebook shape, candidates, Gibbs temperature | [8,5,5,5], 8 (QRM configs: 4), 1.0 |
| `fsq_residual_mode`, `fsq_residual_weight` | `fixed` / `learned_scalar` / `attention` (TRM only), alpha | fixed, 1.0 |
| `fsq_pre_norm` | standardize before tanh | False |
| `fsq_sampling_training`, `fsq_sampling_inference` | TRM: sample instead of greedy | False |
| `fsq_recon_weight` | TRM reconstruction MSE weight | 0.0 |
| `weighted_em_loss`, `weighted_em_weight` | TRM: up-weight non-exact sequences | False, 1.0 |
| `lambda_tree`, `lambda_recon`, `lambda_div`, `alpha` | QRM loss weights and depth penalty | 1.0, 0.1, 0.01, 0.5 |
| `em_weight`, `token_acc_weight` | QRM reward mix | 0.0, 1.0 |
| `num_iterations` | QRM search iterations per puzzle batch | 64 |
| `search_rule`, `c_uct`, `c_puct`, `q_normalize`, `terminal_selection_mode` | QRM selection rule | uct, 1.414, 1.0, True, skip_saturated |
| `do_sampling` | QRM inference sampling | False |
| `train.gradient_accumulation_steps` | QRM optimizer cadence; 0 means `num_iterations` | 0 |
| `train.epochs`, `train.eval_interval`, `train.lr`, `train.puzzle_emb_lr`, `train.weight_decay`, `train.ema` | schedule | per `config/train/*.yaml` |

## 9. Evaluation and ARC pipeline

`test.py --ckpt_dir <ckpt> [--eval_use_compile]` rebuilds the model
from the saved config, loads the checkpoint (EMA weights included), runs
the multi-step evaluation loop, and writes `test_metrics.json`.

ARC needs pass@K over augmentation-inverted predictions and voting across
checkpoints, so it has its own pipeline that reproduces TRM's online
"aggregated voting" offline:

1. `arc_eval.py --ckpt_dir <ckpt>` (or `torchrun ... -m
   qrm.arc_eval_multi_gpu`) writes `arc_eval_state.pt`,
   `arc_metrics.json` and `arc_results/submission.json` per checkpoint;
2. `arc_aggregate.py --exp_dir <run>` merges the saved states of all
   checkpoints, votes, and writes `arc_aggregate_ckpt<N>/`;
3. `arc_pretty_print.py` tabulates every `arc_metrics.json` under
   `results/`.

## 10. Analyzers

`analyze.py --ckpt_dir <ckpt> --analyzer {trm,fsq_entropy}` loads a
checkpoint without `torch.compile`, runs the recursion on the test split
and records per-step diagnostics. See
[05_analysis_tools.md](05_analysis_tools.md).

## 11. torch.compile and other engineering notes

- Compilation gives a 2 to 3x speed-up on these small models. Two
  patterns keep the graph unbroken: `all_finish` is returned as a tensor
  rather than a Python bool, and carry resets use `torch.where` rather
  than data-dependent control flow.
- Compiled parameters carry an `_orig_mod.` prefix; `_clean_param_name`
  strips it when matching EMA shadows and checkpoints.
- Evaluation with EMA swaps weights in place instead of deep-copying the
  model so the compiled graph is reused (`eval_use_compile`).
- The analyzers run uncompiled because `return_diagnostics` changes the
  return arity of the FSQ layer.
- Mixed precision is off in `TrainingArguments`; the model itself runs
  its forward in `forward_dtype` (bfloat16) and casts to float32 where
  needed (quantization, losses).

## 12. Known limitations

- The tree loss sums over `M` candidates, so its scale depends on
  `fsq_top_k`; learning rates were tuned per setting.
- `QRMInner` keeps the unused `lm_head` and `q_head` of the wrapped
  `TRMInner` in the parameter list.
- During a QRM search the `Trainer` still fetches a batch per
  micro-step; those batches are discarded. Data consumption per optimizer
  step therefore differs from TRM by the accumulation factor, which
  `train.py` compensates when computing `max_steps`.
- ARC training uses one embedding per augmented puzzle (about 449M
  buffer entries on ARC-AGI-1 in `full` identifier mode); see
  [08_design_notes.md](08_design_notes.md).
