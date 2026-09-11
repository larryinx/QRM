"""Sanity-check a dataset directory produced by ``qrm.data_prep``.

    python scripts/analysis/check_dataset.py data/sudoku-extreme-1k-aug-1000 [--split train]

Prints the metadata, array shapes and dtypes, verifies that the puzzle
and group index arrays are consistent with the number of rows, and, for
Sudoku datasets (seq_len 81, vocab 11), validates every row:

* each label is a complete, valid Sudoku grid (rows, columns and boxes
  each contain the nine digits once);
* each input agrees with its label on every non-blank cell;
* the number of duplicate (input, label) pairs inside each augmentation
  group is reported (an unseeded builder can produce a few).

Token convention for Sudoku: 0 = pad, 1 = blank, 2..10 = digits 1..9.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ARRAYS = [
    "all__inputs.npy",
    "all__labels.npy",
    "all__puzzle_identifiers.npy",
    "all__puzzle_indices.npy",
    "all__group_indices.npy",
]
SUDOKU_BLANK = 1


def load_split(root: str, split: str) -> dict:
    d = os.path.join(root, split)
    if not os.path.isdir(d):
        sys.exit(f"missing split directory: {d}")
    with open(os.path.join(d, "dataset.json")) as f:
        meta = json.load(f)
    arrays = {}
    for name in ARRAYS:
        p = os.path.join(d, name)
        arrays[name] = np.load(p, mmap_mode="r") if os.path.exists(p) else None
    return {"meta": meta, "arrays": arrays}


def check_indices(arrays: dict) -> list[str]:
    problems = []
    n = arrays["all__inputs.npy"].shape[0]
    pi = arrays["all__puzzle_indices.npy"]
    gi = arrays["all__group_indices.npy"]
    if pi is None or gi is None:
        return ["puzzle/group index arrays missing"]
    if pi[0] != 0 or pi[-1] != n or np.any(np.diff(pi) < 0):
        problems.append(f"puzzle_indices must start at 0, end at {n} and be non-decreasing")
    if gi[0] != 0 or gi[-1] != len(pi) - 1 or np.any(np.diff(gi) < 0):
        problems.append("group_indices must start at 0, end at num_puzzles and be non-decreasing")
    ids = arrays["all__puzzle_identifiers.npy"]
    if ids is not None and ids.shape[0] != len(pi) - 1:
        problems.append("puzzle_identifiers length must equal the number of puzzles")
    return problems


def valid_sudoku_solutions(labels: np.ndarray) -> np.ndarray:
    n = labels.shape[0]
    grids = labels.reshape(n, 9, 9).astype(np.int64)
    digits_ok = ((grids >= 2) & (grids <= 10)).all(axis=(1, 2))
    target = np.arange(2, 11)

    def all_unique(a):  # a: [n, 9, 9]; every row must be a permutation of 2..10
        return (np.sort(a, axis=-1) == target).all(axis=-1).all(axis=-1)

    rows_ok = all_unique(grids)
    cols_ok = all_unique(grids.transpose(0, 2, 1))
    boxes = grids.reshape(n, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(n, 9, 9)
    return digits_ok & rows_ok & cols_ok & all_unique(boxes)


def inputs_match_labels(inputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    blank = inputs == SUDOKU_BLANK
    return ((inputs == labels) | blank).all(axis=1)


def duplicate_pairs_per_group(inputs, labels, puzzle_indices, group_indices) -> np.ndarray:
    dups = np.zeros(len(group_indices) - 1, dtype=np.int64)
    for g in range(len(group_indices) - 1):
        lo, hi = puzzle_indices[group_indices[g]], puzzle_indices[group_indices[g + 1]]
        if hi <= lo:
            continue
        rows = np.concatenate([inputs[lo:hi], labels[lo:hi]], axis=1)
        dups[g] = (hi - lo) - len({r.tobytes() for r in rows})
    return dups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--max_rows", type=int, default=None, help="only validate the first N rows")
    args = parser.parse_args()

    data = load_split(args.dataset_dir, args.split)
    meta, arrays = data["meta"], data["arrays"]

    print(f"== {args.dataset_dir} [{args.split}] ==")
    print(json.dumps(meta, indent=2))
    for name, arr in arrays.items():
        print(f"{name:32s} {'missing' if arr is None else f'{arr.shape} {arr.dtype}'}")

    problems = check_indices(arrays)
    for p in problems:
        print(f"PROBLEM: {p}")

    inputs, labels = arrays["all__inputs.npy"], arrays["all__labels.npy"]
    is_sudoku = meta.get("seq_len") == 81 and meta.get("vocab_size") == 11
    if is_sudoku:
        n = inputs.shape[0] if args.max_rows is None else min(args.max_rows, inputs.shape[0])
        valid = np.zeros(n, dtype=bool)
        consistent = np.zeros(n, dtype=bool)
        step = 100_000
        for lo in range(0, n, step):
            hi = min(lo + step, n)
            valid[lo:hi] = valid_sudoku_solutions(np.asarray(labels[lo:hi]))
            consistent[lo:hi] = inputs_match_labels(np.asarray(inputs[lo:hi]), np.asarray(labels[lo:hi]))
        print(f"valid label grids:            {valid.sum()}/{n}")
        print(f"inputs consistent with labels: {consistent.sum()}/{n}")
        if not valid.all() or not consistent.all():
            problems.append("invalid Sudoku rows found")
        if args.max_rows is None and args.split == "train":
            dups = duplicate_pairs_per_group(
                np.asarray(inputs), np.asarray(labels),
                np.asarray(arrays["all__puzzle_indices.npy"]), np.asarray(arrays["all__group_indices.npy"]),
            )
            print(f"duplicate (input, label) pairs per group: mean {dups.mean():.2f}, "
                  f"max {dups.max()}, groups with duplicates {(dups > 0).sum()}/{dups.size}")

    print("RESULT:", "OK" if not problems else f"{len(problems)} problem(s)")
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
