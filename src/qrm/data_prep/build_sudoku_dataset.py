import csv
import json
import os
from enum import Enum
from typing import Optional

import numpy as np
from argdantic import ArgParser
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from tqdm import tqdm

from qrm.data_prep.common import PuzzleDatasetMetadata

cli = ArgParser()


class SampleMode(str, Enum):
    """Sampling strategy for building the training subset.

    uniform: plain random subsample (legacy behavior).
    match_rating: stratified match on the reference rating histogram.
    match_blank: stratified match on the reference blank-count histogram.
    match_joint: stratified match on the joint (rating, blank) histogram.
    """

    uniform = "uniform"
    match_rating = "match_rating"
    match_blank = "match_blank"
    match_joint = "match_joint"


class MatchReference(str, Enum):
    """Which CSV provides the target distribution for matched sampling."""

    test = "test"
    train = "train"


class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-extreme-full"

    subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_aug: int = 0

    seed: int = 42
    sample_mode: SampleMode = SampleMode.uniform
    match_reference: MatchReference = MatchReference.test
    rating_bin_width: int = 1
    blank_bin_width: int = 1


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))

    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def _read_csv_records(set_name: str, config: DataProcessConfig):
    """Read `{set_name}.csv` into parallel lists of puzzles, solutions,
    ratings, and sources, applying the `min_difficulty` filter."""
    inputs = []
    labels = []
    ratings = []
    sources = []

    path = hf_hub_download(
        config.source_repo, f"{set_name}.csv", repo_type="dataset"
    )
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            if (config.min_difficulty is not None) and (
                int(rating) < config.min_difficulty
            ):
                continue
            assert len(q) == 81 and len(a) == 81

            inputs.append(
                np.frombuffer(
                    q.replace(".", "0").encode(), dtype=np.uint8
                ).reshape(9, 9)
                - ord("0")
            )
            labels.append(
                np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9)
                - ord("0")
            )
            ratings.append(int(rating))
            sources.append(source)

    return inputs, labels, ratings, sources


def _bin_key(rating, blank, mode, rating_bin_width, blank_bin_width):
    if mode == SampleMode.match_rating:
        return rating // rating_bin_width
    if mode == SampleMode.match_blank:
        return blank // blank_bin_width
    if mode == SampleMode.match_joint:
        return (rating // rating_bin_width, blank // blank_bin_width)
    raise ValueError(f"not a matched sample mode: {mode}")


def _records_bin_keys(inputs, ratings, mode, rating_bin_width, blank_bin_width):
    return [
        _bin_key(
            r,
            int((inp == 0).sum()),
            mode,
            rating_bin_width,
            blank_bin_width,
        )
        for inp, r in zip(inputs, ratings)
    ]


def _largest_remainder_allocate(target_hist, pool_bins, total):
    """Largest-remainder allocation from `target_hist` onto `pool_bins`.

    `target_hist` is a dict[bin_key, int] giving the reference counts.
    `pool_bins` is a dict[bin_key, list[int]] where each list holds the
    pool indices that fall into that bin. Returns `(alloc, stats)` such
    that `sum(alloc.values()) == total` and `alloc[k] <= len(pool_bins[k])`
    for every key. Target mass on bins with no pool support is dropped
    (recorded in `stats`); residual overflow from rounding / capping is
    redistributed to high-target bins with headroom.
    """
    # Drop target bins that have no pool support.
    valid_keys = [k for k in target_hist if len(pool_bins.get(k, [])) > 0]
    if not valid_keys:
        raise ValueError("no target bins have pool support")

    dropped = [
        {"bin": str(k), "target_count": int(target_hist[k])}
        for k in target_hist
        if len(pool_bins.get(k, [])) == 0
    ]

    valid_total = sum(target_hist[k] for k in valid_keys)
    alloc_raw = {k: total * target_hist[k] / valid_total for k in valid_keys}
    alloc = {k: int(np.floor(alloc_raw[k])) for k in valid_keys}
    remainders = {k: alloc_raw[k] - alloc[k] for k in valid_keys}

    # Largest-remainder distribution of the rounding leftover.
    remaining = total - sum(alloc.values())
    if remaining > 0:
        sorted_keys = sorted(valid_keys, key=lambda k: (-remainders[k], k))
        for k in sorted_keys[:remaining]:
            alloc[k] += 1

    # Cap at pool availability; redistribute overflow to bins with headroom.
    redistributed = 0
    converged = False
    for _ in range(100):
        overflow = 0
        for k in valid_keys:
            cap = len(pool_bins[k])
            if alloc[k] > cap:
                overflow += alloc[k] - cap
                alloc[k] = cap
        if overflow == 0:
            converged = True
            break
        redistributed += overflow

        headroom = [k for k in valid_keys if alloc[k] < len(pool_bins[k])]
        if not headroom:
            raise RuntimeError(
                f"pool exhausted; cannot place {overflow} samples"
            )
        # Prefer high-target bins with headroom.
        headroom.sort(key=lambda k: (-target_hist[k], k))

        placed = 0
        for k in headroom:
            room = len(pool_bins[k]) - alloc[k]
            take = min(room, overflow - placed)
            alloc[k] += take
            placed += take
            if placed == overflow:
                break
    if not converged:
        raise RuntimeError("allocation did not converge in 100 iterations")

    stats = {
        "dropped_target_bins": dropped,
        "redistributed_total": int(redistributed),
    }
    return alloc, stats


def _matched_subsample(train_records, reference_records, config):
    """Draw `config.subsample_size` pool indices so the selected subset's
    bin histogram matches the reference histogram."""
    train_inputs, _, train_ratings, _ = train_records
    ref_inputs, _, ref_ratings, _ = reference_records

    train_keys = _records_bin_keys(
        train_inputs,
        train_ratings,
        config.sample_mode,
        config.rating_bin_width,
        config.blank_bin_width,
    )
    ref_keys = _records_bin_keys(
        ref_inputs,
        ref_ratings,
        config.sample_mode,
        config.rating_bin_width,
        config.blank_bin_width,
    )

    target_hist = {}
    for k in ref_keys:
        target_hist[k] = target_hist.get(k, 0) + 1

    pool_bins = {}
    for i, k in enumerate(train_keys):
        pool_bins.setdefault(k, []).append(i)

    alloc, alloc_stats = _largest_remainder_allocate(
        target_hist, pool_bins, config.subsample_size
    )

    sampled = []
    for k in sorted(alloc.keys()):
        n = alloc[k]
        if n == 0:
            continue
        pool_arr = np.array(pool_bins[k], dtype=np.int64)
        chosen = np.random.choice(len(pool_arr), size=n, replace=False)
        sampled.extend(pool_arr[chosen].tolist())

    sampled_arr = np.array(sampled, dtype=np.int64)
    np.random.shuffle(sampled_arr)

    manifest = {
        "sample_mode": config.sample_mode.value,
        "match_reference": config.match_reference.value,
        "rating_bin_width": config.rating_bin_width,
        "blank_bin_width": config.blank_bin_width,
        "subsample_size": config.subsample_size,
        "seed": config.seed,
        "reference_n": len(ref_inputs),
        "pool_n": len(train_inputs),
        "num_bins_target": len(target_hist),
        "num_bins_pool": len(pool_bins),
        "per_bin_target": {
            str(k): int(v) for k, v in sorted(target_hist.items())
        },
        "per_bin_alloc": {
            str(k): int(v) for k, v in sorted(alloc.items())
        },
        "per_bin_pool_size": {
            str(k): len(v) for k, v in sorted(pool_bins.items())
        },
        "allocation_stats": alloc_stats,
    }

    return sampled_arr.tolist(), manifest


def convert_subset(set_name, records, config, reference_records=None):
    inputs, labels, _, _ = records

    # If subsample_size is specified for the training set,
    # sample the desired number of examples (uniform or matched).
    if set_name == "train" and config.subsample_size is not None:
        if config.sample_mode == SampleMode.uniform:
            total_samples = len(inputs)
            if config.subsample_size < total_samples:
                indices = np.random.choice(
                    total_samples, size=config.subsample_size, replace=False
                )
                inputs = [inputs[i] for i in indices]
                labels = [labels[i] for i in indices]
        else:
            if reference_records is None:
                raise ValueError(
                    "matched sampling requires reference_records"
                )
            indices, manifest = _matched_subsample(
                records, reference_records, config
            )
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]
            os.makedirs(config.output_dir, exist_ok=True)
            with open(
                os.path.join(config.output_dir, "sample_manifest.json"), "w"
            ) as f:
                json.dump(manifest, f, indent=2)
    elif set_name == "train" and config.sample_mode != SampleMode.uniform:
        raise ValueError(
            f"sample_mode={config.sample_mode.value} requires --subsample-size"
        )

    # Generate dataset
    num_augments = config.num_aug if set_name == "train" else 0

    results = {
        k: []
        for k in [
            "inputs",
            "labels",
            "puzzle_identifiers",
            "puzzle_indices",
            "group_indices",
        ]
    }
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            # Push puzzle (only single example)
            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        # Push group
        results["group_indices"].append(puzzle_id)

    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)

        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1

    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=10 + 1,  # PAD + "0" ... "9"
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"],
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    np.random.seed(config.seed)

    train_records = _read_csv_records("train", config)
    test_records = _read_csv_records("test", config)

    if config.sample_mode != SampleMode.uniform:
        reference_records = (
            test_records
            if config.match_reference == MatchReference.test
            else train_records
        )
    else:
        reference_records = None

    convert_subset(
        "train", train_records, config, reference_records=reference_records
    )
    convert_subset("test", test_records, config)


if __name__ == "__main__":
    cli()
