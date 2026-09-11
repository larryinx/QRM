import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

from qrm.data_prep.common import (
    PuzzleDatasetMetadata,
    dihedral_transform,
    inverse_dihedral_transform,
)

cli = ArgParser()


class IdentifierMode(str, Enum):
    """Puzzle identifier mode for controlling embedding granularity.

    full: base_id|||t{trans}|||color_perm (default, ~876K identifiers for ARC1)
    no_color: base_id|||t{trans} (removes color permutation, ~8.6K identifiers)
    no_aug: base_id only (removes all augmentation, ~960 identifiers)
    shared: all puzzles share one embedding (1 identifier)
    """

    full = "full"
    no_color = "no_color"
    no_aug = "no_aug"
    shared = "shared"


class DataProcessConfig(BaseModel):
    input_file_prefix: str
    output_dir: str
    subsets: List[str]
    test_set_name: str
    test_set_name2: str = "your_test_set"
    seed: int = 42
    num_aug: int = 1000
    puzzle_identifiers_start: int = 1  # start > 1 to handle multiple datasets
    identifier_mode: IdentifierMode = IdentifierMode.full


ARCMaxGridSize = 30
ARCAugmentRetriesFactor = 5

PuzzleIdSeparator = "|||"


@dataclass
class ARCPuzzle:
    id: str  # Embedding ID (for embedding lookup, granularity depends on mode)
    full_aug_string: str  # Full augmentation string (for inverse_aug, always complete)
    examples: List[Tuple[np.ndarray, np.ndarray]]


def arc_grid_to_np(grid: List[List[int]]):
    arr = np.array(grid)

    # Shape check
    assert arr.ndim == 2
    assert arr.shape[0] <= ARCMaxGridSize and arr.shape[1] <= ARCMaxGridSize
    # Element check
    assert np.all((arr >= 0) & (arr <= 9))
    return arr.astype(np.uint8)


def np_grid_to_seq_translational_augment(
    inp: np.ndarray, out: np.ndarray, do_translation: bool
):
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    # Compute random top-left pad
    if do_translation:
        pad_r = np.random.randint(
            0, ARCMaxGridSize - max(inp.shape[0], out.shape[0]) + 1
        )
        pad_c = np.random.randint(
            0, ARCMaxGridSize - max(inp.shape[1], out.shape[1]) + 1
        )
    else:
        pad_r = pad_c = 0

    # Pad grid
    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        grid = np.pad(
            grid + 2,
            (
                (pad_r, ARCMaxGridSize - pad_r - nrow),
                (pad_c, ARCMaxGridSize - pad_c - ncol),
            ),
            constant_values=0,
        )

        # Add <eos>
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARCMaxGridSize:
            grid[eos_row, pad_c:eos_col] = 1
        if eos_col < ARCMaxGridSize:
            grid[pad_r:eos_row, eos_col] = 1

        result.append(grid.flatten())

    return result


def grid_hash(grid: np.ndarray):
    assert grid.ndim == 2
    assert grid.dtype == np.uint8

    buffer = [x.to_bytes(1, byteorder="big") for x in grid.shape]
    buffer.append(grid.tobytes())

    return hashlib.sha256(b"".join(buffer)).hexdigest()


def puzzle_hash(puzzle: dict):
    # Hash the puzzle for checking equivalence
    hashes = []
    for example_type, example in puzzle.items():
        for input, label in example.examples:
            hashes.append(f"{grid_hash(input)}|{grid_hash(label)}")

    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def aug(name: str, mode: IdentifierMode = IdentifierMode.full):
    """Generate augmentation for a puzzle.

    Args:
        name: Base puzzle name.
        mode: Identifier mode controlling embedding granularity.

    Returns:
        Tuple of (embedding_id, full_aug_string, transform_fn):
        - embedding_id: String used for embedding lookup (granularity depends on mode)
        - full_aug_string: Full augmentation string for inverse_aug (always complete)
        - transform_fn: Function to apply the augmentation to a grid
    """
    # Augment plan
    trans_id = np.random.randint(0, 8)
    mapping = np.concatenate(
        [
            np.arange(0, 1, dtype=np.uint8),
            np.random.permutation(np.arange(1, 10, dtype=np.uint8)),
        ]
    )  # Permute colors, Excluding "0" (black)

    # Full augmentation string (always complete for inverse_aug)
    full_aug_string = f"{name}{PuzzleIdSeparator}t{trans_id}{PuzzleIdSeparator}{''.join(str(x) for x in mapping)}"

    # Embedding ID depends on mode
    if mode == IdentifierMode.full:
        embedding_id = full_aug_string
    elif mode == IdentifierMode.no_color:
        embedding_id = f"{name}{PuzzleIdSeparator}t{trans_id}"
    elif mode == IdentifierMode.no_aug:
        embedding_id = name
    elif mode == IdentifierMode.shared:
        embedding_id = "__shared_puzzle_embedding__"
    else:
        embedding_id = full_aug_string  # Default to full

    def _map_grid(grid: np.ndarray):
        return dihedral_transform(mapping[grid], trans_id)

    return embedding_id, full_aug_string, _map_grid


def inverse_aug(name: str):
    """Inverse the "aug" function.

    Supports multiple identifier formats:
    - Full: base_id|||t{trans}|||color_perm (e.g., "puzzle|||t3|||0612397845")
    - No color: base_id|||t{trans} (e.g., "puzzle|||t3")
    - No aug / shared: base_id (e.g., "puzzle" or "__shared_puzzle_embedding__")
    """
    if PuzzleIdSeparator not in name:
        return name, lambda x: x

    parts = name.split(PuzzleIdSeparator)

    if len(parts) == 3:
        # Full format: base_id|||t{trans}|||color_perm
        base_name, trans_str, perm = parts
        trans_id = int(trans_str[1:])  # Remove "t" letter
        inv_perm = np.argsort(list(perm)).astype(np.uint8)

        def _map_grid_full(grid: np.ndarray):
            return inv_perm[inverse_dihedral_transform(grid, trans_id)]

        return base_name, _map_grid_full

    elif len(parts) == 2:
        # No color format: base_id|||t{trans}
        base_name, trans_str = parts
        trans_id = int(trans_str[1:])  # Remove "t" letter

        def _map_grid_no_color(grid: np.ndarray):
            return inverse_dihedral_transform(grid, trans_id)

        return base_name, _map_grid_no_color

    else:
        # Unexpected format, return identity
        return name.split(PuzzleIdSeparator)[0], lambda x: x


def convert_single_arc_puzzle(
    results: dict,
    name: str,
    puzzle: dict,
    aug_count: int,
    dest_mapping: Dict[str, Tuple[str, str]],
    identifier_mode: IdentifierMode = IdentifierMode.full,
):
    """Convert a single ARC puzzle with augmentations.

    Args:
        results: Dictionary to store converted puzzles.
        name: Base puzzle name.
        puzzle: Raw puzzle data with "train" and "test" examples.
        aug_count: Number of augmentations to generate.
        dest_mapping: Mapping from example type to (split, set) destination.
        identifier_mode: Identifier mode for embedding granularity.
    """
    # Convert original (no augmentation)
    dests = set(dest_mapping.values())

    # Determine embedding_id for original (no augmentation) based on identifier_mode
    # Original has identity transform (t0) and identity color permutation (0123456789)
    original_full_aug_string = (
        f"{name}{PuzzleIdSeparator}t0{PuzzleIdSeparator}0123456789"
    )
    if identifier_mode == IdentifierMode.full:
        original_embedding_id = original_full_aug_string
    elif identifier_mode == IdentifierMode.no_color:
        original_embedding_id = f"{name}{PuzzleIdSeparator}t0"
    elif identifier_mode == IdentifierMode.no_aug:
        original_embedding_id = name
    elif identifier_mode == IdentifierMode.shared:
        original_embedding_id = "__shared_puzzle_embedding__"
    else:
        original_embedding_id = original_full_aug_string

    converted = {
        dest: ARCPuzzle(original_embedding_id, original_full_aug_string, [])
        for dest in dests
    }
    for example_type, examples in puzzle.items():
        # Map to target split
        dest = dest_mapping[example_type]
        converted[dest].examples.extend(
            [
                (arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"]))
                for example in examples
            ]
        )

    group = [converted]

    # Augment
    if aug_count > 0:
        hashes = {puzzle_hash(converted)}

        for _trial in range(ARCAugmentRetriesFactor * aug_count):
            embedding_id, full_aug_string, _map_grid = aug(name, identifier_mode)

            # Check duplicate
            augmented = {
                dest: ARCPuzzle(
                    embedding_id,
                    full_aug_string,
                    [
                        (_map_grid(input), _map_grid(label))
                        for (input, label) in puzzle_data.examples
                    ],
                )
                for dest, puzzle_data in converted.items()
            }
            h = puzzle_hash(augmented)
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)

            if len(group) >= aug_count + 1:
                break

        if len(group) < aug_count + 1:
            print(f"[Puzzle {name}] augmentation not full, only {len(group)}")

    # Append
    for dest in dests:
        # Convert the examples
        dest_split, dest_set = dest

        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])
        results[dest_split][dest_set].append([converted[dest] for converted in group])


def load_puzzles_arcagi(config: DataProcessConfig):
    train_examples_dest = ("train", "all")
    test_examples_map = {
        config.test_set_name: [(1.0, ("test", "all"))],
        config.test_set_name2: [(1.0, ("test", "all"))],
        "_default": [(1.0, ("train", "all"))],
    }

    test_puzzles = {}
    results = {}

    total_puzzles = 0
    for subset_name in config.subsets:
        # Load all puzzles in this subset
        with open(
            f"{config.input_file_prefix}_{subset_name}_challenges.json", "r"
        ) as f:
            puzzles = json.load(f)

        sols_filename = f"{config.input_file_prefix}_{subset_name}_solutions.json"
        if os.path.isfile(sols_filename):
            with open(sols_filename, "r") as f:
                sols = json.load(f)

                for puzzle_id in puzzles.keys():
                    for idx, sol_grid in enumerate(sols[puzzle_id]):
                        puzzles[puzzle_id]["test"][idx]["output"] = sol_grid
        else:
            # Fill with dummy
            print(f"{subset_name} solutions not found, filling with dummy")

            for puzzle_id, puzzle in puzzles.items():
                for example in puzzle["test"]:
                    example.setdefault("output", [[0]])

        # Shuffle puzzles
        puzzles = list(puzzles.items())
        np.random.shuffle(puzzles)

        # Assign by fraction
        for idx, (name, puzzle) in enumerate(puzzles):
            fraction = idx / len(puzzles)
            test_examples_dest = None
            for f, dest in test_examples_map.get(
                subset_name, test_examples_map["_default"]
            ):
                if fraction < f:
                    test_examples_dest = dest
                    break

            assert test_examples_dest is not None

            if test_examples_dest[0] == "test":
                test_puzzles[name] = puzzle

            convert_single_arc_puzzle(
                results,
                name,
                puzzle,
                config.num_aug,
                {"train": train_examples_dest, "test": test_examples_dest},
                identifier_mode=config.identifier_mode,
            )
            total_puzzles += 1

    print(f"Total puzzles: {total_puzzles}")
    print(f"Identifier mode: {config.identifier_mode.value}")
    return results, test_puzzles


def convert_dataset(config: DataProcessConfig):
    np.random.seed(config.seed)

    # Read dataset
    data, test_puzzles = load_puzzles_arcagi(config)

    # Map global puzzle identifiers
    # SHARED mode: start from 0 (no blank needed, like maze/sudoku)
    # Other modes: start from 1 (0 is blank for padding)
    if config.identifier_mode == IdentifierMode.shared:
        num_identifiers = 0
    else:
        num_identifiers = config.puzzle_identifiers_start  # 0 is blank, start at 1
    identifier_map = {}
    for split_name, split in data.items():
        for subset_name, subset in split.items():
            for group in subset:
                for puzzle in group:
                    if puzzle.id not in identifier_map:
                        identifier_map[puzzle.id] = num_identifiers
                        num_identifiers += 1
    if config.identifier_mode == IdentifierMode.shared:
        print(f"Total puzzle IDs (SHARED mode, no blank): {num_identifiers}")
    else:
        print(f"Total puzzle IDs (including <blank>): {num_identifiers}")

    # Save
    for split_name, split in data.items():
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)

        # Translational augmentations
        enable_translational_augment = split_name == "train"

        # Statistics
        total_examples = 0
        total_puzzles = 0
        total_groups = 0

        for subset_name, subset in split.items():  # "all" is the only subset
            # Construct subset
            results = {
                k: []
                for k in [
                    "inputs",
                    "labels",
                    "puzzle_identifiers",
                    "puzzle_aug_strings",  # Full augmentation strings for inverse_aug
                    "puzzle_indices",
                    "group_indices",
                ]
            }
            results["puzzle_indices"].append(0)
            results["group_indices"].append(0)

            example_id = 0
            puzzle_id = 0

            for group in subset:
                for puzzle in group:
                    # Push puzzle
                    no_aug_id = np.random.randint(0, len(puzzle.examples))
                    for _idx_ex, (inp, out) in enumerate(puzzle.examples):
                        inp, out = np_grid_to_seq_translational_augment(
                            inp,
                            out,
                            do_translation=enable_translational_augment
                            and _idx_ex != no_aug_id,
                        )

                        results["inputs"].append(inp)
                        results["labels"].append(out)
                        example_id += 1

                        total_examples += 1

                    results["puzzle_indices"].append(example_id)
                    results["puzzle_identifiers"].append(identifier_map[puzzle.id])
                    results["puzzle_aug_strings"].append(puzzle.full_aug_string)

                    puzzle_id += 1
                    total_puzzles += 1

                # Push group
                results["group_indices"].append(puzzle_id)
                total_groups += 1

            for k, v in results.items():
                if k in {"inputs", "labels"}:
                    v = np.stack(v, 0)
                elif k == "puzzle_aug_strings":
                    # Save strings as object array
                    v = np.array(v, dtype=object)
                else:
                    v = np.array(v, dtype=np.int32)

                np.save(
                    os.path.join(
                        config.output_dir, split_name, f"{subset_name}__{k}.npy"
                    ),
                    v,
                )

        # Metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=ARCMaxGridSize * ARCMaxGridSize,
            vocab_size=10 + 2,  # PAD + EOS + "0" ... "9"
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / total_puzzles,
            total_puzzles=total_puzzles,
            sets=list(split.keys()),
            identifier_mode=config.identifier_mode.value,
        )

        # Save metadata as JSON.
        with open(
            os.path.join(config.output_dir, split_name, "dataset.json"), "w"
        ) as f:
            json.dump(metadata.model_dump(), f)

    # Save IDs mapping
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        ids_mapping = {v: k for k, v in identifier_map.items()}
        json.dump([ids_mapping.get(i, "<blank>") for i in range(num_identifiers)], f)

    # Save Test Puzzles
    with open(os.path.join(config.output_dir, "test_puzzles.json"), "w") as f:
        json.dump(test_puzzles, f)


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
