# Training and evaluation guide

## 1. Environment

The project is managed with [uv](https://docs.astral.sh/uv/). Python
3.10 or newer and a CUDA GPU are required for training.

```bash
git clone https://github.com/larryinx/QRM.git && cd QRM
uv sync                 # creates .venv and installs the package with its dependencies
uv sync --group dev     # additionally installs pytest, ruff, black, isort, pre-commit
```

Run commands either through `uv run ...` or after `source .venv/bin/activate`.
`torch==2.9.0` is installed from PyPI (CUDA 12 wheels on Linux); the
comment at the bottom of `pyproject.toml` shows how to pin a different
CUDA build through a PyTorch index.

Training logs to Weights & Biases (`report_to=["wandb"]` is fixed in
`train.py`). Set `WANDB_PROJECT` to choose a project, and `WANDB_MODE`
to `online`, `offline` or `disabled`. The example scripts default to
`offline`.

## 2. Data

```bash
uv run bash scripts/prepare_data/prepare_sudoku_maze.sh     # Sudoku-Extreme and Maze-Hard
uv run bash scripts/prepare_data/prepare_arc1.sh full       # ARC-AGI-1
uv run bash scripts/prepare_data/prepare_arc2.sh full       # ARC-AGI-2
uv run python scripts/analysis/check_dataset.py data/sudoku-extreme-1k-aug-1000
```

Dataset paths in `config/dataset/*.yaml` are relative to the repository
root, so launch training from the root. See
[04_datasets.md](04_datasets.md) for builder options.

## 3. Launching a run

`python -m qrm.train` composes three config groups and accepts Hydra
overrides:

```bash
uv run python -m qrm.train model=<model> dataset=<dataset> train=<train> [key=value ...]
```

| Group | Choices |
|---|---|
| `model` | `trm_att_sudoku`, `trm_mlp_t_sudoku`, `trm_att_maze`, `trm_mlp_t_maze`, `trm_att_arc_agi`, `trm_att_arc_agi_lowrank{,16,32,128}`, `qrm_att_sudoku`, `qrm_mlp_t_sudoku`, `qrm_att_maze`, `qrm_att_arc_agi` |
| `dataset` | `sudoku`, `sudoku_rating`, `sudoku_blank`, `sudoku_rating_blank`, `sudoku_full`, `maze`, `arc_agi_1{,_no_color,_no_aug,_shared}`, `arc_agi_2{,_no_color,_no_aug,_shared}` |
| `train` | `sudoku`, `maze`, `arc_agi` |

The model group decides the model class: `trm_*` selects TRM (with FSQ
when `model.use_fsq=True`), `qrm_*` selects QRM with tree search. The
example scripts under `scripts/train_examples/` wrap the twelve
combinations we ran plus a chain-mode control:

| Script | What it trains |
|---|---|
| `trm_sudoku.sh`, `trm_maze.sh`, `trm_arc_agi_1.sh`, `trm_arc_agi_2.sh` | TRM baselines |
| `trm_fsq_sudoku.sh`, `trm_fsq_maze.sh`, `trm_fsq_arc_agi_1.sh`, `trm_fsq_arc_agi_2.sh` | TRM + FSQ (single chain) |
| `qrm_sudoku.sh`, `qrm_maze.sh`, `qrm_arc_agi_1.sh`, `qrm_arc_agi_2.sh` | QRM (FSQ + MCTS) |
| `qrm_chain_mode.sh` | QRM with branching factor 1 (control) |

Each script sources `_common.sh`, which picks `python` or `torchrun`
from `NPROC_PER_NODE`, and forwards any extra arguments as overrides:

```bash
uv run bash scripts/train_examples/trm_sudoku.sh
NPROC_PER_NODE=4 uv run bash scripts/train_examples/qrm_sudoku.sh
NPROC_PER_NODE=8 uv run bash scripts/train_examples/trm_fsq_arc_agi_1.sh dataset.global_batch_size=768
```

The Sudoku and Maze scripts reproduce reported numbers. The ARC-AGI
FSQ and QRM scripts and the Maze QRM script reproduce the settings we
launched but did not finish; they are starting points, not tuned
recipes.

### Multi-GPU

`torchrun` sets `RANK`, `WORLD_SIZE` and `LOCAL_RANK`; `train.py` reads
them, places each process on `cuda:LOCAL_RANK`, and splits
`dataset.global_batch_size` across processes. The global batch size,
not the per-device one, is the configured quantity. ARC runs used 4 to 8
GPUs; QRM ARC runs additionally set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce
fragmentation from the per-tree carries.

### Outputs and run names

A run writes to `results/<run_name>/`:

```
results/<run_name>/
  config.yaml            merged Hydra config (+ model_type), read by test/analyze/arc_eval
  training_args.yaml
  checkpoint-<step>/     model, optimizer, scheduler, trainer state, ema_state.pt, signsgd_optimizer.pt, dataset_state.pt
  final_checkpoint/      same, written at the end of training
```

`run_name` is `<model>__<dataset>__<train>` followed by abbreviated
overrides (for example `fsq--True__flevels--[8,5,5,5]__fres_w--0.3`), and
prefixed by the scheduler job id when one is present. Because the name
can contain brackets and commas, quote it when passing it back.

### Resuming

If the output directory already contains `checkpoint-*`, training
resumes from the latest one automatically. To continue a run under a
different command (for example more epochs), point it at the directory:

```bash
uv run python -m qrm.train model=trm_att_sudoku dataset=sudoku train=sudoku \
    train.epochs=80000 "train.output_dir='results/<run_name>'"
```

### Budget and schedule keys

`train.epochs` and `train.eval_interval` are expressed in dataset
epochs, as in TRM; `train.py` converts them to optimizer steps
(`steps_per_epoch = total_groups * mean_puzzle_examples / global_batch_size`).
For QRM the step counts are divided by the gradient-accumulation factor
so the number of examples seen matches TRM.

## 4. Choosing FSQ and search settings

Recipes that worked (details in [07_experiments.md](07_experiments.md)):

- **TRM + FSQ, Sudoku, attention**: `fsq_residual_weight=0.3`,
  `fsq_pre_norm=True`, `fsq_recon_weight=0.5`, `weighted_em_loss=True`,
  greedy quantization (both `fsq_sampling_*` False).
- **TRM + FSQ, Sudoku, MLP-T**: `fsq_residual_weight=0.3`, greedy, nothing else.
- **TRM + FSQ, Maze**: either plain `use_fsq=True` with sampling on for
  both training and inference, or residual 0.3 with `fsq_recon_weight=0.5`
  greedy. Keep `fsq_sampling_training` and `fsq_sampling_inference` equal.
- **QRM, Sudoku**: `fsq_top_k=4`, `num_iterations=64`,
  `train.gradient_accumulation_steps=4`, `train.lr=2e-4`,
  `train.puzzle_emb_lr=2e-4`, residual 0.3.

Knobs worth knowing when exploring further:

| Knob | Effect |
|---|---|
| `model.fsq_top_k` | branching factor; memory and the tree-loss scale grow with it |
| `model.num_iterations` | expansions per puzzle batch; 64 with depth 16 and branching 4 |
| `train.gradient_accumulation_steps` | 0 = update after each full search; 4 updated the weights during the search and worked best |
| `model.em_weight`, `model.token_acc_weight` | reward mix; token accuracy alone is the default |
| `model.alpha` | depth penalty; 0 removes the preference for shallow answers |
| `model.c_uct`, `model.search_rule` | exploration constant; `puct` uses FSQ priors |
| `model.fsq_temperature` | flattens or sharpens the Gibbs distribution over codes |

## 5. Evaluation

Sudoku and Maze checkpoints (TRM, TRM + FSQ, QRM):

```bash
uv run bash scripts/eval/eval_checkpoint.sh results/<run_name>/final_checkpoint
# or directly
uv run python -m qrm.test --ckpt_dir results/<run_name>/checkpoint-65104 --eval_use_compile
uv run python -m qrm.pretty_print --compile      # table over all evaluated checkpoints
```

Metrics land in `<ckpt>/test_metrics_compile.json` (`test_metrics.json`
without `--eval_use_compile`). `exact_accuracy` is the number reported
in the paper.

ARC-AGI (TRM and TRM + FSQ checkpoints):

```bash
uv run bash scripts/eval/eval_arc.sh results/<run_name>            # every checkpoint, then voting
NPROC_PER_NODE=4 uv run bash scripts/eval/eval_arc.sh results/<run_name>
```

This runs `qrm.arc_eval` (or `qrm.arc_eval_multi_gpu`) per checkpoint,
`qrm.arc_aggregate` to vote across checkpoints, and
`qrm.arc_pretty_print` for a pass@K table. `pass@2` is the reported
metric. The ARC evaluator currently instantiates the TRM model class,
so QRM checkpoints cannot be evaluated on ARC with it yet.

## 6. Tests

```bash
uv sync --group dev
uv run pytest                              # CPU tests; GPU tests are skipped without CUDA
uv run pytest src/qrm/tests/test_mcts_unit.py -q
```

`test_stochastic_fsq.py` compares the quantizer with the reference
implementation in `vector_quantize_pytorch`; `test_mcts_unit.py`
exercises selection and backup with a fake carry; `test_qrm_mcts.py` and
`test_mcts_expand.py` run a tiny QRM end to end on a GPU.
