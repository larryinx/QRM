# Datasets

All builders live in `src/qrm/data_prep/` and write the layout that
`TRMIterableDataset` expects: one directory per dataset with `train/`
and `test/` subdirectories holding `all__inputs.npy`,
`all__labels.npy`, `all__puzzle_identifiers.npy`,
`all__puzzle_indices.npy`, `all__group_indices.npy` and a
`dataset.json` with `seq_len`, `vocab_size`,
`num_puzzle_identifiers` and related metadata. The wrapper scripts in
`scripts/prepare_data/` call the builders with the settings used in our
experiments.

```bash
bash scripts/prepare_data/prepare_sudoku_maze.sh    # Sudoku-Extreme 1k x 1000 aug, Maze-Hard 1k
bash scripts/prepare_data/prepare_arc1.sh full      # ARC-AGI-1 (+ ConceptARC), identifier mode
bash scripts/prepare_data/prepare_arc2.sh full      # ARC-AGI-2 (+ ConceptARC)
bash scripts/prepare_data/prepare_sudoku_variants.sh  # rating / blank matched Sudoku subsamples
```

Run them from the repository root inside the uv environment
(`uv run bash ...` or after `source .venv/bin/activate`).

## Sudoku-Extreme

Source: `sapientinc/sudoku-extreme` on the Hugging Face Hub (downloaded
automatically). The default training set is a uniform subsample of 1000
puzzles, each augmented 1000 times with digit permutations and
band/stack shuffles that preserve validity; the test set is the full
422,786-puzzle evaluation split. Sequence length 81, vocabulary 11
(pad plus digits 0 to 9), one shared puzzle identifier.

```bash
python -m qrm.data_prep.build_sudoku_dataset \
    --output-dir data/sudoku-extreme-1k-aug-1000 \
    --subsample-size 1000 --num-aug 1000 [--seed 42]
```

Variants built by `prepare_sudoku_variants.sh` change how the 1000
training puzzles are chosen (`--sample-mode`):

| Config name | Directory | Selection |
|---|---|---|
| `sudoku` | `sudoku-extreme-1k-aug-1000` | uniform |
| `sudoku_rating` | `...-rating` | rating histogram matched to the test set |
| `sudoku_blank` | `...-blank` | blank-count histogram matched to the test set |
| `sudoku_rating_blank` | `...-rating-blank` | joint histogram, rating bins of 5 |
| `sudoku_full` | `sudoku-extreme-full` | all 3.83M puzzles, no augmentation |

In our runs none of the matched subsamples beat the uniform one, so the
default stays `sudoku`. (`sudoku_att` in `config/dataset/` points to an
internal re-build of the same recipe that one reported QRM-Att run used;
no builder is provided for it, so use `sudoku`.)

## Maze-Hard

Source: `sapientinc/maze-30x30-hard-1k`. 1000 training mazes, 30x30
grids (sequence length 900), vocabulary 6, one shared identifier.
The build is deterministic.

```bash
python -m qrm.data_prep.build_maze_dataset --output-dir data/maze-30x30-hard-1k
```

## ARC-AGI-1 and ARC-AGI-2

The raw JSON files are bundled under `data/kaggle/combined/`:
`arc-agi_{training,evaluation}_{challenges,solutions}.json` (ARC-AGI-1),
`arc-agi_{training2,evaluation2}_*.json` (ARC-AGI-2), and
`arc-agi_concept_*.json` (ConceptARC, training only). The builder
applies dihedral transforms, translations and colour permutations
(`--num-aug 1000`, de-duplicated per puzzle), pads grids to 30x30
(sequence length 900, vocabulary 12), and stores an
`identifiers.json` mapping plus the full augmentation string of every
row so that predictions can be inverted at evaluation time.

```bash
python -m qrm.data_prep.build_arc_dataset \
    --input-file-prefix data/kaggle/combined/arc-agi \
    --output-dir data/arc1concept-aug-1000 \
    --subsets training evaluation concept \
    --test-set-name evaluation \
    --identifier-mode full
# ARC-AGI-2: --subsets training2 evaluation2 concept --test-set-name evaluation2
```

`--identifier-mode` controls how many puzzle embeddings the model
learns:

| Mode | Identifier | Count (ARC-AGI-1) | Config |
|---|---|---|---|
| `full` | puzzle, transform and colour permutation | 876,406 | `arc_agi_1` |
| `no_color` | puzzle and transform | 7,680 | `arc_agi_1_no_color` |
| `no_aug` | puzzle | 961 | `arc_agi_1_no_aug` |
| `shared` | one embedding for everything | 1 | `arc_agi_1_shared` |

`full` matches TRM and is what the reported numbers use. Alternatives
such as `no_aug` (25% pass@2 versus 45% for `full` on ARC-AGI-1) or a
low-rank embedding (`model.puzzle_emb_rank=64`, 39%) reduce the
embedding table at a cost in accuracy; see
[08_design_notes.md](08_design_notes.md).

Two properties of the ARC split are worth knowing when reading results:
the demonstration pairs of test puzzles are part of the training set
(as in TRM and HRM), so evaluation is not zero-shot; and the ARC-AGI-2
training set contains 382 of the 400 ARC-AGI-1 evaluation puzzles, so a
model trained on ARC-AGI-2 cannot be fairly evaluated on ARC-AGI-1.
